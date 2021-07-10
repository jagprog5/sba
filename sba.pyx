# cython: overflowcheck=False, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
np.import_array()

cimport cython
from cython.parallel cimport prange
from cpython cimport array, Py_buffer, PyObject_CheckBuffer, PyBUF_WRITABLE, PyBUF_FORMAT, PyBUF_ND, PyBUF_STRIDES
from cpython.object cimport Py_EQ, Py_NE, Py_LT, Py_LE, Py_GE, Py_GT
from libc.stdlib cimport rand, srand, RAND_MAX, qsort
from libc.string cimport memset, memcpy, memmove
from libc.limits cimport INT_MAX, INT_MIN

'''
There's two different ways of handling memory allocations when doing operations. It's not apparent which is better, so I'm leaving both as an option for now.
ALLOC_THEN_SHRINK = True:
    Allocate the maximum size required for the result, based off of the lengths of the operands. Then, shrink the result to the actb ual used size when complete.
    For example, if A is of length 2, and B is of length 3, then A AND B will have at max a length of 2.
    This strategy would allocate an array of length 2, then shrink it down when the AND op is complete to free up the excess.
ALLOC_THEN_SHRINK = False:
    Realloc up from a length of 0 until the operation is done, according to SBA.lengthen_if_needed
'''
cdef bint ALLOC_THEN_SHRINK = False

'''
Always shorten the array when an op is complete.
turn_off function ignores this, and only shortens when needed.
'''
cdef bint STRICT_SHORTEN = False

cdef extern from "math.h":
    float floorf(float) nogil
    float log10f(float) nogil
    int roundf(float) nogil

cdef extern from "time.h":
    void* time(void*) nogil

cdef extern from "object.h":
    int Py_IS_TYPE(void*, void*) nogil

cdef extern from "numpy/arrayobject.h":
    void *PyArray_malloc(int nbytes) nogil
    void PyArray_free(void *ptr) nogil
    void *PyArray_realloc(void *ptr, int nbytes) nogil
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags) nogil
    void *PyArray_DATA(void *arr) nogil

class SBAException(Exception):
    pass

cdef bint do_sba_verify = True 

@cython.final
cdef class SBA:
    @staticmethod
    cdef inline void c_verify_input(bint enable):
        do_sba_verify = enable
    
    @staticmethod
    def verify_input(bint enable):
        SBA.c_verify_input(enable)

    cdef inline int raise_if_viewing(self) nogil except -1:
        if self.views > 0:
            raise SBAException("Buffer is still being viewed, or is not owned!")
    
    cdef inline void range_indices(self, int start_inclusive, int stop_inclusive) nogil:
        '''
        sets the indices' values  
        ensure start <= stop  
        [start, start + 1, ..., stop - 1, stop]
        '''
        cdef int i = 0
        cdef int val = start_inclusive
        while val <= stop_inclusive:
            self.indices[i] = val
            i += 1
            val += 1
    
    cdef inline void lengthen_if_needed(self, int length_override = -1) nogil:
        '''
        lengthen the indices if the capacity has been reached.  
        if length_override is specified, then use it as the current length instead of the SBA's length.
        '''
        if (self.len.len if length_override == -1 else length_override) >= self.cap: # ==
            self.cap = self.cap + (self.cap >> 1) + 1; # cap = cap * 1.5 + 1, estimate for golden ratio (idk python uses the same strat)
            self.indices = <int*>PyArray_realloc(self.indices, sizeof(self.indices[0]) * self.cap)
    
    cdef inline void shorten(self) nogil:
        self.cap = self.len.len
        self.indices = <int*>PyArray_realloc(self.indices, sizeof(self.indices[0]) * self.cap)

    cdef inline void shorten_if_needed(self, bint ignore_strict_shorten = False) nogil:
        if (STRICT_SHORTEN and not ignore_strict_shorten) or self.len.len < self.cap >> 1:
            self.shorten()
    
    @staticmethod
    cdef SBA c_range(int start_inclusive, int stop_inclusive):
        if start_inclusive > stop_inclusive:
            raise SBAException("start must be <= stop")
        cdef SBA ret = SBA.__new__(SBA)
        # ret.views = 0 # implicit since initialization guarantees 0s in memory. Keep this in mind for all factory methods.
        ret.cap = stop_inclusive - start_inclusive + 1
        ret.indices = <int*>PyArray_malloc(sizeof(ret.indices[0]) * ret.cap)
        ret.range_indices(start_inclusive, stop_inclusive)
        ret.len.len = ret.cap
        return ret

    @staticmethod
    def range(int start_inclusive, int stop_inclusive):
        return SBA.c_range(start_inclusive, stop_inclusive)

    @staticmethod
    cdef SBA capacity(int cap, bint default):
        '''
        if default is True:  
            initalizes the indices such that they are descending from 0 to cap-1.  
        else:  
            allocate the capacity but leave it uninitialized, and sets this SBA's length to 0.
        '''
        if cap < 0:
            raise SBAException("cap must be non-negative!")
        cdef SBA ret = SBA.__new__(SBA)
        ret.cap = cap
        ret.indices = <int*>PyArray_malloc(sizeof(ret.indices[0]) * ret.cap)
        if default:
            ret.len.len = ret.cap
            ret.range_indices(0, ret.len.len - 1)
        return ret
    
    @staticmethod
    def length(int len=0):
        return SBA.capacity(len, True)
    
    @staticmethod
    def iterable(obj, filter = None, *, bint reverse = False, verify = None):
        cdef SBA ret = SBA.__new__(SBA)
        cdef int ln = <int>len(obj)
        if filter is None:
            # iterable is a sparse array
            if do_sba_verify if verify is None else verify:
                for i in range(ln):
                    if type(obj[i]) is not int:
                        raise TypeError("Indices must be ints.")
                    if obj[i] > INT_MAX or obj[i] < INT_MIN:
                        raise SBAException("Indices ust be in c int range.")
                for i in range(ln - 1):
                    if obj[i] >= obj[i + 1]:
                        raise SBAException("Indices must be in ascending order, with no duplicates.")
            ret.cap = ln
            ret.indices = <int*>PyArray_malloc(sizeof(ret.indices[0]) * ret.cap)
            ret.len.len = ret.cap
            for i in range(ln):
                ret.indices[i] = obj[i]
        else:
            # iterable is a dense array
            i = 0
            while i < ln:
                if filter(obj[ln - i - 1 if reverse else i]):
                    ret.lengthen_if_needed()
                    ret.indices[ret.len.len] = i
                    ret.len.len += 1
                i += 1
        return ret

    @staticmethod
    def buffer(const_numeric[:] buf, filter = None, *, bint copy = True, bint reverse = False, verify = None):
        cdef SBA ret = SBA.__new__(SBA)
        cdef int ln = <int>buf.shape[0] 
        cdef int i
        if filter is None:
            ret.len.len = ln
            if do_sba_verify if verify is None else verify:
                i = 0
                while i < ln - 1:
                    if buf[i] >= buf[i + 1]:
                        raise SBAException("Indices must be in ascending order, with no duplicates.")
                    i += 1
            if copy:
                ret.cap = ret.len.len
                ret.indices = <int*>PyArray_malloc(sizeof(ret.indices[0]) * ret.cap)
                i = 0
                while i < ln:
                    ret.indices[i] = <int>buf[i]
                    i += 1
            else:
                if const_numeric is not int:
                    raise SBAException("Buffer type must be c int when creating SBA by reference to buffer.")
                ret.views = 1 # lock-out changing mem
                # ret.cap not set since it should not be used in this SBA
                ret.indices = <int*>&buf[0]
        else:
            i = 0
            while i < ln:
                if filter(buf[ln - i - 1 if reverse else i]):
                    ret.lengthen_if_needed()
                    ret.indices[ret.len.len] = i
                    ret.len.len += 1
                i += 1
        return ret
    
    @staticmethod
    cdef SBA encode_linear(float input, int num_on_bits, int length):
        if num_on_bits > length:
            raise SBAException("The number of ON bits can't exceed the length of the array.")
        cdef SBA ret = SBA.capacity(num_on_bits, False)
        ret.len.len = num_on_bits
        cdef int start_offset = <int>roundf((length - num_on_bits) * input)
        for i in range(num_on_bits):
            ret.indices[i] = start_offset + i
        return ret
    
    @staticmethod
    cdef SBA encode_periodic(float input, float period, int num_on_bits, int length):
        if input < 0:
            input *= -<float>1
        if num_on_bits > length:
            raise SBAException("The number of ON bits can't exceed the length of the array.")
        cdef SBA ret = SBA.capacity(num_on_bits, False)
        ret.len.len = num_on_bits
        cdef float progress = input / period
        progress = progress - <int>progress
        cdef int start_offset = <int>roundf(progress * length)
        cdef int index_wrap
        cdef int index_leading
        if start_offset + num_on_bits > length:
            index_wrap = start_offset + num_on_bits - length
            index_leading = index_wrap
            while index_wrap > 0:
                index_wrap -= 1
                ret.indices[index_wrap] = index_wrap
            while index_leading < num_on_bits:
                ret.indices[index_leading] = length - (num_on_bits - index_leading)
                index_leading += 1
        else:
            for i in range(num_on_bits):
                ret.indices[i] = start_offset + i
        return ret

    @staticmethod
    def encode(float input, int num_on_bits, int size, float period = 0):
        if period == 0:
            return SBA.encode_linear(input, num_on_bits, size)
        else:
            return SBA.encode_periodic(input, period, num_on_bits, size)

    def __getbuffer__(SBA self, Py_buffer *buffer, int flags):
        # Buffer protocol. Read-only
        if buffer == NULL:
            raise BufferError("Buffer is NULL")
        if flags & PyBUF_WRITABLE:
            memset(buffer, 0, sizeof(buffer[0]))
            raise BufferError("Buffer is read-only.")
        
        buffer.buf = self.indices
        buffer.obj = self
        buffer.itemsize = sizeof(self.indices[0])
        buffer.len = buffer.itemsize * self.len.len
        buffer.readonly = 1
        buffer.ndim = 1
        buffer.suboffsets = NULL
        # buffer.internal = NULL

        if flags & PyBUF_FORMAT:
            buffer.format = 'i'
        else:
            buffer.format = NULL

        if flags & PyBUF_ND:
            buffer.shape = &self.len.ssize_t_len
        else:
            buffer.shape = NULL
        
        if flags & PyBUF_STRIDES:
            buffer.strides = &buffer.itemsize
        else:
            buffer.strides = NULL
        self.views += 1

    def __releasebuffer__(SBA self, Py_buffer* buffer):
        self.views -= 1
    
    cpdef np.ndarray to_np(SBA self, bint give_ownership = False):
        if not give_ownership:
            return np.frombuffer(memoryview(self), dtype=np.intc)
        self.raise_if_viewing()
        cdef np.npy_intp* dims = <np.npy_intp*>&self.len.ssize_t_len
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, dims, np.NPY_INT, self.indices)
        PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
        self.indices = NULL
        self.cap = 0
        self.len.len = 0
        return arr
    
    def __dealloc__(self):
        if self.views == 0: # owner?
            PyArray_free(self.indices)
    
    # =========================================================================================

    cpdef print_raw(SBA self):
        cdef int amount = 0
        cdef int val
        if self.len.len != 0:
            for i in range(self.len.len - 1):
                val = self.indices[i]
                amount += 2 + (0 if val >= 0 else 1) + (0 if val == 0 else <int>floorf(log10f(<float>val)))
            for i in range(amount):
                print(' ', end='')
            print("V")
        for i in range(self.cap):
            print(str(self.indices[i]) + " ", end='')
        print('')
    
    def __repr__(self):
        return "[" + " ".join([str(self.indices[i]) for i in range(self.len.len)]) + "]"
    
    def __iter__(self):
        return (self.indices[i] for i in range(len(self)))

    def __len__(self):
        return self.len.len
    
    cdef void turn_on(self, int index) nogil:
        self.raise_if_viewing()
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle = 0
        cdef int mid_val = INT_MAX
        while left <= right:
            middle = (right + left) >> 1
            mid_val = self.indices[middle]
            if mid_val < index:
                left = middle + 1
            elif mid_val > index:
                right = middle - 1
            else:
                return; # skip duplicate

        if index > mid_val:
            middle += 1

        self.lengthen_if_needed()
        memmove(self.indices + middle + 1, self.indices + middle, (self.len.len - middle) * sizeof(self.indices[0]))
        self.len.len += 1
        self.indices[middle] = index
    
    cdef void turn_off(SBA self, int index) nogil:
        self.raise_if_viewing()
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle = 0
        cdef int mid_val
        while left <= right:
            middle = (right + left) >> 1
            mid_val = self.indices[middle]
            if mid_val == index:
                self.len.len -= 1
                memmove(self.indices + middle, self.indices + middle + 1, (self.len.len - middle) * sizeof(self.indices[0]))
                self.shorten_if_needed(ignore_strict_shorten=True)
                return
            elif mid_val < index:
                left = middle + 1
            else:
                right = middle - 1
    
    def set(SBA self, int index, bint state):
        if state:
            self.turn_on(index)
        else:
            self.turn_off(index)
        return self

    cdef inline int check_index(SBA self, int index) nogil except -1:
        if index >= self.len.len:
            raise SBAException("Index out of bounds.")
        if index < 0:
            index = self.len.len + index
            if index < 0:
                raise SBAException("Index out of bounds.")
        return index
    
    def __delitem__(SBA self, int index):
        self.raise_if_viewing()
        cdef int i = self.check_index(index)
        self.len.len -= 1
        memmove(&self.indices[i], &self.indices[i + 1], sizeof(int) * (self.len.len - i))

    def __setitem__(SBA self, int index, int value):
        self.__delitem__(index)
        self.turn_on(value)

    cdef SBA get_section(SBA self, int start_inclusive, int stop_inclusive):
        # start <= stop
        cdef SBA ret
        if ALLOC_THEN_SHRINK:
            ret = SBA.capacity(stop_inclusive - start_inclusive + 1, False)
        else:
            ret = SBA.__new__(SBA)
        if self.len.len == 0:
            return ret
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle
        cdef int mid_val
        while left <= right:
            middle = (right + left) >> 1
            mid_val = self.indices[middle]
            if mid_val == start_inclusive:
                break
            elif mid_val < start_inclusive:
                left = middle + 1
            else:
                right = middle - 1
        if start_inclusive > mid_val:
            middle += 1
            mid_val = self.indices[middle]

        while mid_val <= stop_inclusive:
            if not ALLOC_THEN_SHRINK:
                ret.lengthen_if_needed()
            ret.indices[ret.len.len] = mid_val
            ret.len.len += 1
            middle += 1
            if middle >= self.len.len:
                break # ran off end
            mid_val = self.indices[middle]
        if ALLOC_THEN_SHRINK or STRICT_SHORTEN:
            ret.shorten()
        return ret
    
    def __getitem__(SBA self, index):
        if isinstance(index, slice):
            if self.len.len != 0:
                start = self.indices[0] if index.start is None else index.start
                stop = self.indices[self.len.len - 1] if index.stop is None else index.stop
                return self.get(start, stop)
            else:
                return SBA()
        else:
            return self.indices[self.check_index(index)]

    cpdef SBA cp(SBA self):
        cdef SBA ret = SBA.capacity(self.len.len, False)
        ret.len.len = self.len.len
        for i in range(self.len.len):
            ret.indices[i] = self.indices[i]
        return ret
    
    cdef inline void _get_one(SBA a, int* offset, int* value, bint* nempty) nogil:
        if offset[0] >= a.len.len:
            nempty[0] = 0
            return
        value[0] = a.indices[offset[0]]
        offset[0] += 1
    
    cdef inline void _get_both(SBA a, int* a_offset, int* a_val, bint* a_empty, SBA b, int* b_offset, int* b_val, bint* b_empty) nogil:
        SBA._get_one(a, a_offset, a_val, a_empty)
        SBA._get_one(b, b_offset, b_val, b_empty)
    
    cdef inline void _add_to_output(SBA r, int* r_len, int val, bint len_only) nogil:
        if not len_only:
            if not ALLOC_THEN_SHRINK:
                r.lengthen_if_needed(r_len[0])
            r.indices[r_len[0]] = val
        r_len[0] += 1

    @staticmethod
    cdef inline SBA alloc_orc(SBA a, SBA b):
        cdef SBA ret
        if ALLOC_THEN_SHRINK:
            ret = SBA.capacity(a.len.len + b.len.len, False)
        else:
            ret = SBA.__new__(SBA)
        return ret

    @staticmethod
    cdef void orc(void* r, SBA a, SBA b, bint exclusive, bint len_only) nogil:
        # if len_only, r is an int* to uninitialized int
        #        else, r is an SBA*, with 0ed members
        cdef bint a_nempty = 1
        cdef int a_offset = 0
        cdef int a_val
        cdef bint b_nempty = 1
        cdef int b_offset = 0
        cdef int b_val
        cdef int r_len = 0
        SBA._get_both(a, &a_offset, &a_val, &a_nempty, b, &b_offset, &b_val, &b_nempty)
        while a_nempty or b_nempty:
            if (a_nempty and not b_nempty) or (a_nempty and b_nempty and a_val < b_val):
                SBA._add_to_output(<SBA>r, &r_len, a_val, len_only)
                SBA._get_one(a, &a_offset, &a_val, &a_nempty)
            elif (not a_nempty and b_nempty) or (a_nempty and b_nempty and a_val > b_val):
                SBA._add_to_output(<SBA>r, &r_len, b_val, len_only)
                SBA._get_one(b, &b_offset, &b_val, &b_nempty)
            elif a_nempty and b_nempty and a_val == b_val:
                if not exclusive:
                    SBA._add_to_output(<SBA>r, &r_len, a_val, len_only)
                SBA._get_both(a, &a_offset, &a_val, &a_nempty, b, &b_offset, &b_val, &b_nempty)
        if len_only:
            (<int*>r)[0] = r_len
        else:
            (<SBA>r).len.len = r_len
            if ALLOC_THEN_SHRINK or STRICT_SHORTEN:
                (<SBA>r).shorten()
    
    def orb(SBA a not None, SBA b not None):
        cdef SBA ret = SBA.alloc_andc(a, b)
        SBA.orc(<void*>ret, a, b, False, False)
        return ret
    
    def orl(SBA a not None, SBA b not None):
        cdef int ret
        SBA.orc(<void*>&ret, a, b, False, True)
        return ret
    
    def orp(SBA query not None, np.ndarray[dtype='object', ndim=1] arr):
        cdef int ln = <int>len(arr)
        cdef np.ndarray out = np.zeros(ln, dtype=np.intc)
        cdef int* out_ptr = <int*>PyArray_DATA(<void*>out)
        cdef void** arr_ptr = <void**>PyArray_DATA(<void*>arr)
        cdef int i
        for i in prange(ln, nogil=True):
            if not Py_IS_TYPE(arr_ptr[i], <void*>SBA):
                with gil:
                    raise TypeError("Found non SBA object in arr.")
            SBA.orc(<void*>&out_ptr[i], query, <SBA>arr_ptr[i], False, True)
        return out
    
    def xorb(SBA a not None, SBA b not None):
        cdef SBA ret = SBA.alloc_andc(a, b)
        SBA.orc(<void*>ret, a, b, True, False)
        return ret
    
    def xorl(SBA a not None, SBA b not None):
        cdef int ret
        SBA.orc(<void*>&ret, a, b, True, True)
        return ret
    
    def xorp(SBA query not None, np.ndarray[dtype='object', ndim=1] arr):
        cdef int ln = <int>len(arr)
        cdef np.ndarray out = np.zeros(ln, dtype=np.intc)
        cdef int* out_ptr = <int*>PyArray_DATA(<void*>out)
        cdef void** arr_ptr = <void**>PyArray_DATA(<void*>arr)
        cdef int i
        for i in prange(ln, nogil=True):
            if not Py_IS_TYPE(arr_ptr[i], <void*>SBA):
                with gil:
                    raise TypeError("Found non SBA object in arr.")
            SBA.orc(<void*>&out_ptr[i], query, <SBA>arr_ptr[i], True, True)
        return out

    @staticmethod
    cdef void andc(void* r, SBA a, SBA b, bint len_only) nogil:
        # if len_only, r is an int* to uninitialized int
        #        else, r is an SBA*, with 0ed members
        cdef bint a_nempty = 1
        cdef int a_offset = 0
        cdef int a_val
        cdef bint b_nempty = 1
        cdef int b_offset = 0
        cdef int b_val
        cdef int r_len = 0
        SBA._get_both(a, &a_offset, &a_val, &a_nempty, b, &b_offset, &b_val, &b_nempty)
        while a_nempty and b_nempty:
            if a_val < b_val:
                SBA._get_one(a, &a_offset, &a_val, &a_nempty)
            elif a_val > b_val:
                SBA._get_one(b, &b_offset, &b_val, &b_nempty)
            else: # ==
                SBA._add_to_output(<SBA>r, &r_len, a_val, len_only)
                SBA._get_both(a, &a_offset, &a_val, &a_nempty, b, &b_offset, &b_val, &b_nempty)
        if len_only:
            (<int*>r)[0] = r_len
        else:
            (<SBA>r).len.len = r_len
            if ALLOC_THEN_SHRINK or STRICT_SHORTEN:
                (<SBA>r).shorten()
    
    @staticmethod
    cdef inline SBA alloc_andc(SBA a, SBA b):
        cdef SBA ret
        if ALLOC_THEN_SHRINK:
            ret = SBA.capacity(b.len.len if a.len.len > b.len.len else a.len.len, False)
        else:
            ret = SBA.__new__(SBA)
        return ret

    def andb(SBA a not None, SBA b not None):
        cdef SBA ret = SBA.alloc_andc(a, b)
        SBA.andc(<void*>ret, a, b, False)
        return ret
    
    def andl(SBA a not None, SBA b not None):
        cdef int ret
        SBA.andc(<void*>&ret, a, b, True)
        return ret
    
    def andi(SBA a not None, SBA b not None):
        a.raise_if_viewing()
        SBA.andc(<void*>a, a, b, False)
        return a
    
    def andp(SBA query not None, np.ndarray[dtype='object', ndim=1] arr):
        cdef int ln = <int>len(arr)
        cdef np.ndarray out = np.zeros(ln, dtype=np.intc)
        cdef int* out_ptr = <int*>PyArray_DATA(<void*>out)
        cdef void** arr_ptr = <void**>PyArray_DATA(<void*>arr)
        cdef int i
        for i in prange(ln, nogil=True):
            if not Py_IS_TYPE(arr_ptr[i], <void*>SBA):
                with gil:
                    raise TypeError("Found non SBA object in arr.")
            SBA.andc(<void*>&out_ptr[i], query, <SBA>arr_ptr[i], True)
        return out
   
    def __add__(SBA self, other):
        if not isinstance(self, SBA): # __radd__
            if isinstance(self, str):
                return self + str(other)
            else:
                raise TypeError(str(type(self)) + " not supported for reverse + op.")
        
        if isinstance(other, str):
            return str(self) + other
        elif isinstance(other, SBA):
            return SBA.orb(self, other)
        cdef SBA cp = self.cp()
        if isinstance(other, int):
            cp.turn_on(<int>other)
            return cp
        elif hasattr(other, "__getitem__"):
            cp = self.cp()
            for i in other:
                if isinstance(i, int):
                    cp.turn_on(<int>i)
                else:
                    raise TypeError("for + op, all elements in a list must be integers")
            return cp
        else:
            raise TypeError(str(type(other)) + " not supported for + op.")
    
    def __or__(self, other):
        return self.__add__(other)
    
    cdef bint get_bit(SBA self, int index) nogil:
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle
        cdef int mid_val
        while left <= right:
            middle = (right + left) >> 1
            mid_val = self.indices[middle]
            if mid_val == index:
                return True
            elif mid_val < index:
                left = middle + 1
            else:
                right = middle - 1
        return False
    
    def get(SBA self, index1, index2 = None):
        if index2 is None:
            return self.get_bit(index1)
        else:
            if index2 < index1:
                tmp = index2
                index2 = index1
                index1 = tmp
            return self.get_section(index1, index2)
    
    def __contains__(SBA self, index):
        if isinstance(index, int):
            return self.get_bit(index)
        else:
            raise TypeError("Needs type int, not: " + type(index))

    def __mul__(SBA self, other):
        if isinstance(other, SBA):
            return self.andb(other)
        elif isinstance(other, int):
            return self.get_bit(other)
        elif isinstance(other, float):
            return self.cp().subsample(other)
        else:
            raise TypeError(str(type(other)) + " not supported for * or & ops.")
    
    def __and__(SBA self, other):
        return self.__mul__(other)
    
    def __mod__(SBA self, other):
        cdef SBA c = self.cp()
        if isinstance(other, int):
            c.subsample_length(other)
            return c
        else:
            raise TypeError("other must be int, not: " + str(type(other)))
    
    def __xor__(SBA self, other):
        return self.xorb(other)
    
    cdef void c_rm(SBA self, SBA rm) nogil:
        self.raise_if_viewing()
        cdef int a_from = 0
        cdef int a_to = 0
        cdef int a_val
        cdef int rm_offset = 0
        cdef int rm_val
        while a_from < self.len.len:
            if rm_offset < rm.len.len:
                a_val = self.indices[a_from]
                rm_val = rm.indices[rm_offset]
                if rm_val < a_val:
                    rm_offset += 1
                    continue
                elif rm_val == a_val:
                    rm_offset += 1
                    a_from += 1
                    continue
            self.indices[a_to] = self.indices[a_from]
            a_to += 1
            a_from += 1
        self.len.len = a_to
        self.shorten_if_needed()
    
    def rm(SBA self, SBA rm):
        self.c_rm(rm)
        return self
    
    def __sub__(SBA self, other):
        cdef SBA cp = self.cp()
        if isinstance(other, int):
            cp.turn_off(<int>other)
            return cp
        elif isinstance(other, SBA):
            cp.rm(other)
            return cp
        elif hasattr(other, "__getitem__"):
            for i in other:
                if isinstance(i, int):
                    cp.turn_off(<int>i)
                else:
                    raise TypeError("for - op, all elements must be integers")
            return cp
        else:
            raise TypeError(str(type(other)) + " not supported for - op.")
    
    cdef void c_shift(SBA self, int n) nogil:
        # self.raise_if_viewing() # not needed since realloc isn't called
        cdef int i = 0
        while i < self.len.len:
            self.indices[i] += n
            i += 1

    def shift(SBA self, int n):
        self.c_shift(n)
        return self
    
    def __lshift__(SBA self, int n):
        cdef SBA cp = self.cp()
        cp.c_shift(n)
        return cp

    def __rshift__(SBA self, int n):
        cdef SBA cp = self.cp()
        cp.c_shift(-n)
        return cp
    
    @staticmethod
    cdef void c_seed_rand() nogil:
        srand(<unsigned int>time(NULL))

    @staticmethod
    def seed_rand():
        SBA.c_seed_rand()

    @staticmethod
    def rand_int() -> int:
        return rand()
    
    cdef bint compare(SBA self, SBA other, int op) nogil:
        cdef int i
        if op == Py_EQ or op == Py_NE:
            if self.len.len != other.len.len:
                return False if op == Py_EQ else True
            for i in range(self.len.len):
                if self.indices[i] != other.indices[i]:
                    return False if op == Py_EQ else True
            return True if op == Py_EQ else False
        else:
            for i in range(min(self.len.len, other.len.len)):
                if self.indices[self.len.len - i - 1] < other.indices[other.len.len - i - 1]:
                    return True if op == Py_LE or op == Py_LT else False
            if self.len.len == other.len.len:
                return op == Py_LE or op == Py_GE
            else:
                if self.len.len < other.len.len:
                    return op == Py_LE or op == Py_LT
                else:
                    return op == Py_GE or op == Py_GT

    def __richcmp__(SBA self, other, int op):
        if isinstance(other, SBA):
            return self.compare(other, op)
        elif op == Py_EQ or op == Py_NE:
            try:
                t = iter(other)
                count = 0
                for i in t:
                    count += 1
                    if not i in self:
                        return False if op == Py_EQ else True 
                ln_eq = len(self) == count
                if op == Py_EQ:
                    return ln_eq if op == Py_EQ else not ln_eq
            except:
                return False
        else:
            raise TypeError("This comparison op is not supported for non SBA types.")
    
    def subsample(SBA self, amount):
        if isinstance(amount, int):
            self.subsample_length(amount)
            return self
        elif isinstance(amount, float):
            self.subsample_portion(amount)
            return self
        else:
            raise TypeError("arg must be an int or a float")

    @staticmethod
    cdef int _qsort_compare(const void* a, const void* b) nogil:
        return (<int*>a)[0] - (<int*>b)[0]
    
    cdef void subsample_length(SBA self, int amount) nogil:
        self.raise_if_viewing()
        if self.len.len <= amount:
            return
        cdef int i = self.len.len
        cdef int j
        while i > 1: # Sattolo's
            i -= 1
            j = rand() % i
            self.indices[j], self.indices[i] = self.indices[i], self.indices[j]
        self.len.len = amount
        qsort(self.indices, self.len.len, sizeof(self.indices[0]), &SBA._qsort_compare)
        self.shorten_if_needed()

    cdef int subsample_portion(SBA self, float amount) nogil except -1:
        self.raise_if_viewing()
        if amount < 0 or amount > 1:
            raise SBAException("amount must be from 0 to 1, inclusively")
        cdef int check_val = <int>(amount * RAND_MAX)
        cdef int to_offset = 0
        cdef int from_offset = 0
        while from_offset < self.len.len:
            if rand() < check_val:
                self.indices[to_offset] = self.indices[from_offset]
                to_offset += 1
            from_offset += 1
        self.len.len = to_offset
