# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from __future__ import print_function
import numpy as np
cimport numpy as np
np.import_array()

from cpython cimport Py_buffer, PyObject_CheckBuffer, PyBUF_WRITABLE, PyBUF_FORMAT, PyBUF_ND, PyBUF_STRIDES
from cpython.object cimport Py_EQ, Py_NE, Py_LT, Py_LE, Py_GE, Py_GT
from cpython.mem cimport PyMem_Realloc, PyMem_Free
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.string cimport memset, memcpy, memmove
from libc.limits cimport INT_MIN
from typing import Iterable, Union, Callable

'''
There's two different ways of handling memory allocations when doing operations. It's not apparent which is better, so I'm leaving both as an option for now.
ALLOC_THEN_SHRINK = True:
    Allocate the maximum size required for the result, based off of the lengths of the operands. Then, shrink the result to the actual used size when complete.
    For example, if A is of length 2, and B is of length 3, then A AND B will have at max a length of 2.
    This strategy would allocate an array of length 2, then shrink it down when the AND op is complete to free up the excess.
ALLOC_THEN_SHRINK = False:
    Continually realloc up from a length of 0 until the operation is done, according to SBA._lengthen_if_needed
'''
cdef bint ALLOC_THEN_SHRINK = False

'''
Always shorten the array when an op is complete. This may not be optimal with repeated calls to turnOff.
'''
cdef bint STRICT_SHORTEN = False

cdef extern from "math.h":
    float floorf(float)
    float log10f(float)
    int roundf(float)

cdef extern from "time.h":
    void* time(void*)

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef fused const_numeric:
    const short[::1]
    const int[::1]
    const long[::1]
    const float[::1]
    const double[::1]

class SBAException(Exception):
    pass

cdef union SBALen: # Needs little-endian
    Py_ssize_t ssize_t_len # for use in buffer protocol
    int len # for normal use in SBA

cdef bint do_sba_checking = 1

cdef class SBA:
    cdef int views # number of references to buffer
    cdef int cap # capacity, mem allocated for the indices
    cdef SBALen len # length, the number of ON bits in the array
    cdef int* indices # contains indices of bits that are ON. From MSB to LSB, aka descending index value

    @staticmethod
    def enable_checking():
        do_sba_checking = 1
    
    @staticmethod
    def disable_checking():
        do_sba_checking = 0
    
    def __init__(self, *args, **kwargs):
        if len(args) == 0:
            args = [0]
        if isinstance(args[0], int):
            if len(args) > 1 and isinstance(args[1], int):
                self.setFromRange(args[0], args[1])
            else:
                self.setFromCapacity(args[0], True)
        # elif isinstance(arg[0], np.ndarray):
        #     self.setFromNp(arg, True)
        elif PyObject_CheckBuffer(arg[0]):
            # bint check_valid=True, bint reverse=False, dense_filter=None
            if len(arg) == 1:
                self.setFromBuffer(args[0], check_valid)
            else:
                self.setFromDense(args[0], reverse, dense_filter)
        else:
            self._set_from_iterable(args[0])
    
    cdef inline int raiseIfViewing(self) except -1:
        if self.views > 0:
            raise SBAException("Buffer is still being viewed, or is not owned!")
    
    cdef _init(self, int cap, bint set_len = True):
        '''
        Initialize some members. This is used in the factory methods.
        cap: The new capacity.
        set_len: set the length equal to the capacity, else leave it uninitialized because it's about to be set.
        '''
        self.raiseIfViewing()
        self.cap = cap
        if set_len:
            self.len.len = cap
        self.indices = <int*>PyMem_Realloc(self.indices, sizeof(self.indices[0]) * cap)
    
    def _set_from_iterable(self, obj: Iterable[int], bint check_valid = True):
        # Replaces this SBA's data with the iterable's data. Deep-copies. 
        self.raiseIfViewing()
        cdef int ln = <int>len(obj)
        if do_sba_checking and check_valid:
            for i in obj:
                if not isinstance(i, int):
                    raise SBAException("Integers only.")
            for i in range(ln - 1):
                if obj[i] <= obj[i + 1]:
                    raise SBAException("Indices must be in descending order, with no duplicates.")
        self._init(ln)
        for i in range(ln):
            self.indices[i] = obj[i]
    
    @staticmethod
    def from_iterable(obj: Iterable[int], bint check_valid = True) -> SBA:
        cdef SBA ret = SBA.__new__(SBA)
        ret._set_from_iterable(obj, check_valid)
        return ret
    
    cdef inline void _range(self, int stop_inclusive, int start_inclusive):
        '''
        stop >= start  
        Set [stop, stop - 1, ..., start + 1, start]
        '''
        cdef int i = 0
        cdef int val = stop_inclusive
        while val >= start_inclusive:
            self.indices[i] = val
            i += 1
            val -= 1
    
    cdef inline void _default(self):
        # Set [len - 1, ..., 2, 1, 0]
        self._range(self.len.len - 1, 0)
    
    cdef setFromRange(self, int stop_inclusive, int start_inclusive):
        if stop_inclusive < start_inclusive:
            raise SBAException("stop must be >= start")
        self.raiseIfViewing()
        cdef int cap = stop_inclusive - start_inclusive + 1
        self._init(cap)
        self._range(stop_inclusive, start_inclusive)
    
    @staticmethod
    cdef SBA fromRange(int stop_inclusive, int start_inclusive):
        cdef SBA ret = SBA.__new__(SBA)
        ret.setFromRange(stop_inclusive, start_inclusive)
        return ret
    
    @staticmethod
    def from_range(stop_inclusive: int, start_inclusive: int) -> SBA:
        return SBA.fromRange(stop_inclusive, start_inclusive)

    cdef setFromCapacity(self, int initial_capacity, bint set_default):
        # Replace's this SBA's data.  
        # set_default: set to default value, otherwise leaves indices uninitialized and len = 0.  
        if initial_capacity < 0:
            raise SBAException("cap must be non-negative!")
        self.raiseIfViewing()
        self._init(initial_capacity, set_default)
        if set_default:
            self._default()

    @staticmethod
    cdef SBA fromCapacity(int initial_capacity = 0, bint set_default = 1):
        cdef SBA ret = SBA.__new__(SBA)
        ret.setFromCapacity(initial_capacity, set_default)
        return ret

    @staticmethod
    def from_capacity(initial_capacity = 0) -> SBA:
        return SBA.fromCapacity(initial_capacity, True)
    
    cdef inline _lengthen_if_needed(self, int length_override = -1):
        if (self.len.len if length_override == -1 else length_override) >= self.cap: # ==
            self.cap = self.cap + (self.cap >> 1) + 1; # cap = cap * 1.5 + 1, estimate for golden ratio (idk python uses the same strat)
            self.indices = <int*>PyMem_Realloc(self.indices, self.cap * sizeof(self.indices[0]))
    
    cdef setFromDense(self, const_numeric buf, bint reverse = 0, filter: Callable[[Union[int, float]], bool] = None):
        self._init(0)
        cdef int ln = len(buf)
        cdef int i = ln - 1 if reverse else 0
        while i > -1 if reverse else i < ln:
            if buf[i] != 0 if filter is None else filter(buf[i]):
                self._lengthen_if_needed() # allowing ALLOC_THEN_SHRINK = True here would not be practical
                self.indices[self.len.len] = i if reverse else ln - i - 1
                self.len.len += 1
            if reverse:
                i -= 1
            else:
                i += 1
    
    @staticmethod
    def from_dense(const_numeric buf, bint reverse = 0, filter: Callable[[Union[int, float]], bool] = None):
        cdef SBA ret = SBA.__new__(SBA)
        ret.setFromDense(buf, reverse, filter)
        return ret

    @staticmethod
    cdef inline bint _is_valid(const int[::1] arr):
        cdef int i = 0
        cdef int len = <int>arr.shape[0]
        while i < len - 1:
            if arr[i] <= arr[i + 1]:
                return 0
            i += 1
        return 1

    cdef setFromBuffer(self, const int[::1] buf, bint check_valid = True):
        self.raiseIfViewing()
        if do_sba_checking and check_valid and not SBA._is_valid(buf):
            raise SBAException("The buffer doesn't have valid indices.")
        cdef int len = <int>buf.shape[0]
        self._init(len)
        self.indices = <int*>PyMem_Realloc(self.indices, len * sizeof(self.indices[0]))
        memcpy(self.indices, &buf[0], len * sizeof(self.indices[0]))

    cdef setFromNp(self, np.ndarray[int, ndim=1] arr, bint deep_copy, bint check_valid = True):
        self.raiseIfViewing()
        if do_sba_checking and check_valid and not SBA._is_valid(arr):
            raise SBAException("The numpy array doesn't have valid indices.")
        cdef int len = <int>np.PyArray_DIMS(arr)[0]
        cdef int* data = <int*>np.PyArray_DATA(arr)
        self._init(len)
        if deep_copy:
            self.indices = <int*>PyMem_Realloc(self.indices, len * sizeof(self.indices[0]))
            memcpy(self.indices, data, len * sizeof(data[0]))
        else:
            PyMem_Free(self.indices)
            self.indices = data
            self.views = 1 # lock-out changing mem
    
    @staticmethod
    cdef SBA fromNp(np.ndarray[int, ndim=1] arr, bint deep_copy, bint check_valid):
        cdef SBA ret = SBA.__new__(SBA)
        ret.setFromNp(arr, deep_copy, check_valid)
        return ret
    
    @staticmethod
    def from_np(np_arr, deep_copy = True, check_valid = True) -> SBA:
        return SBA.fromNp(np_arr, deep_copy, check_valid)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
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

    def __releasebuffer__(self, Py_buffer *buffer):
        self.views -= 1
    
    cdef np.ndarray toNp(self, bint give_ownership):
        if not give_ownership:
            return np.frombuffer(memoryview(self), dtype=np.intc)

        self.raiseIfViewing()
        cdef np.npy_intp* dims = <np.npy_intp*>&self.len.ssize_t_len
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, dims, np.NPY_INT, self.indices)
        PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
        self.indices = NULL
        self.cap = 0
        self.len.len = 0
        return arr
    
    def to_np(self, give_ownership = True):
        return self.toNp(give_ownership)
    
    def __dealloc__(self):
        if self.views == 0: # owner?
            PyMem_Free(self.indices)
    
    # =========================================================================================

    cdef inline _shorten(self):
        self.cap = self.len.len
        self.indices = <int*>PyMem_Realloc(self.indices, sizeof(self.indices[0]) * self.cap)

    cdef inline _shorten_if_needed(self):
        if STRICT_SHORTEN or self.len.len < self.cap >> 1:
            self._shorten()

    cdef printRaw(self):
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

    def print_raw(self):
        self.printRaw()
    
    def __repr__(self):
        return "[" + " ".join([str(self.indices[i]) for i in range(self.len.len)]) + "]"
    
    def __iter__(self):
        return (self.indices[i] for i in range(len(self)))

    def __len__(self):
        return self.len.len
    
    cdef turnOn(self, int index):
        self.raiseIfViewing()
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle = 0
        cdef int mid_val = INT_MIN
        while left <= right:
            middle = (right + left) >> 1
            mid_val = self.indices[middle]
            if mid_val > index:
                left = middle + 1
            elif mid_val < index:
                right = middle - 1
            else:
                return; # skip duplicate

        if index < mid_val:
            middle += 1

        self._lengthen_if_needed()
        memmove(self.indices + middle + 1, self.indices + middle, (self.len.len - middle) * sizeof(self.indices[0]))
        self.len.len += 1
        self.indices[middle] = index
    
    cdef turnOff(self, int index):
        self.raiseIfViewing()
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
                self._shorten_if_needed()
                return
            elif mid_val > index:
                left = middle + 1
            else:
                right = middle - 1
    
    def set(self, int index, bint state):
        if state:
            self.turnOn(index)
        else:
            self.turnOff(index)

    cdef int _checkIndex(self, int index) except -1:
        if index >= self.len.len:
            raise SBAException("Index out of bounds.")
        if index < 0:
            index = self.len.len + index
            if index < 0:
                raise SBAException("Index out of bounds.")
        return index
    
    def __delitem__(self, index):
        self.raiseIfViewing()
        cdef int i = self._checkIndex(index)
        self.len.len -= 1
        memmove(&self.indices[i], &self.indices[i + 1], sizeof(int) * (self.len.len - i))

    def __setitem__(self, index: int, value: int):
        self.__delitem__(index)
        self.turnOn(value)
    
    cdef SBA getSection(self, int stop_inclusive, int start_inclusive):
        # stop >= start
        cdef SBA ret
        if ALLOC_THEN_SHRINK:
            ret = SBA.fromCapacity(stop_inclusive - start_inclusive + 1, False)
        else:
            ret = SBA.__new__(SBA)
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle
        cdef int mid_val
        while left <= right:
            middle = (right + left) >> 1
            mid_val = self.indices[middle]
            if mid_val == stop_inclusive:
                break
            elif mid_val > stop_inclusive:
                left = middle + 1
            else:
                right = middle - 1
        if stop_inclusive < mid_val:
            middle += 1
            mid_val = self.indices[middle]

        while mid_val >= start_inclusive:
            if not ALLOC_THEN_SHRINK:
                ret._lengthen_if_needed()
            ret.indices[ret.len.len] = mid_val
            ret.len.len += 1
            middle += 1
            if middle >= self.len.len:
                break # ran off end
            mid_val = self.indices[middle]
        if ALLOC_THEN_SHRINK or STRICT_SHORTEN:
            ret._shorten()
        return ret
    
    def __getitem__(self, index: Union[int, slice]) -> Union[int, SBA]:
        if isinstance(index, slice):
            start = self.indices[0] if index.start is None else index.start
            stop = self.indices[len(self) - 1] if index.stop is None else index.stop
            return self.get(stop, start)
        else:
            return self.indices[self._checkIndex(index)]

    cpdef SBA cp(self):
        cdef SBA ret = SBA.fromCapacity(self.len.len, 0)
        ret.len.len = self.len.len
        for i in range(self.len.len):
            ret.indices[i] = self.indices[i]
        return ret
    
    cdef inline void _get_one(SBA a, int* offset, int* value, bint* nempty):
        if offset[0] >= a.len.len:
            nempty[0] = 0
            return
        value[0] = a.indices[offset[0]]
        offset[0] += 1
    
    cdef inline void _get_both(SBA a, int* a_offset, int* a_val, bint* a_empty, SBA b, int* b_offset, int* b_val, bint* b_empty):
        SBA._get_one(a, a_offset, a_val, a_empty)
        SBA._get_one(b, b_offset, b_val, b_empty)
    
    cdef inline void _add_to_output(SBA r, int* r_len, int val, bint len_only):
        if not len_only:
            if not ALLOC_THEN_SHRINK:
                r._lengthen_if_needed(r_len[0])
            r.indices[r_len[0]] = val
        r_len[0] += 1

    @staticmethod
    cdef orc(void* r, SBA a, SBA b, bint exclusive, bint len_only):
        # if len_only, r is an int* to uninitilized int
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
            if (a_nempty and not b_nempty) or (a_nempty and b_nempty and a_val > b_val):
                SBA._add_to_output(<SBA>r, &r_len, a_val, len_only)
                SBA._get_one(a, &a_offset, &a_val, &a_nempty)
            elif (not a_nempty and b_nempty) or (a_nempty and b_nempty and a_val < b_val):
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
                (<SBA>r)._shorten()
    
    @staticmethod
    def orb(SBA a not None, SBA b not None) -> SBA:
        cdef SBA ret
        if ALLOC_THEN_SHRINK:
            ret = SBA.fromCapacity(a.len.len + b.len.len, False)
        else:
            ret = SBA.__new__(SBA)
        SBA.orc(<void*>ret, a, b, 0, 0)
        return ret
    
    @staticmethod
    def orl(SBA a not None, SBA b not None) -> int:
        cdef int ret
        SBA.orc(<void*>&ret, a, b, 0, 1)
        return ret
    
    @staticmethod
    def xorb(SBA a not None, SBA b not None) -> SBA:
        cdef SBA ret
        if ALLOC_THEN_SHRINK:
            ret = SBA.fromCapacity(a.len.len + b.len.len, False)
        else:
            ret = SBA.__new__(SBA)
        SBA.orc(<void*>ret, a, b, 1, 0)
        return ret
    
    @staticmethod
    def xorl(SBA a not None, SBA b not None) -> int:
        cdef int ret
        SBA.orc(<void*>&ret, a, b, 1, 1)
        return ret
    
    @staticmethod
    cdef andc(void* r, SBA a, SBA b, bint len_only):
        # if len_only, r is an int* to uninitilized int
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
            if a_val > b_val:
                SBA._get_one(a, &a_offset, &a_val, &a_nempty)
            elif a_val < b_val:
                SBA._get_one(b, &b_offset, &b_val, &b_nempty)
            else: # ==
                SBA._add_to_output(<SBA>r, &r_len, a_val, len_only)
                SBA._get_both(a, &a_offset, &a_val, &a_nempty, b, &b_offset, &b_val, &b_nempty)
        if len_only:
            (<int*>r)[0] = r_len
        else:
            (<SBA>r).len.len = r_len
            if ALLOC_THEN_SHRINK or STRICT_SHORTEN:
                (<SBA>r)._shorten()
    
    @staticmethod
    def andb(SBA a not None, SBA b not None) -> SBA:
        cdef SBA ret
        if ALLOC_THEN_SHRINK:
            ret = SBA.fromCapacity(b.len.len if a.len.len > b.len.len else a.len.len, False)
        else:
            ret = SBA.__new__(SBA)
        SBA.andc(<void*>ret, a, b, 0)
        return ret
    
    @staticmethod
    def andl(SBA a not None, SBA b not None) -> int:
        cdef int ret
        SBA.andc(<void*>&ret, a, b, 1)
        return ret
    
    def andi(self, SBA b not None) -> SBA:
        self.raiseIfViewing()
        SBA.andc(<void*>self, self, b, 0)
        return self
   
    def __add__(self, other):
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
            cp.turnOn(<int>other)
            return cp
        elif hasattr(other, "__getitem__"):
            cp = self.cp()
            for i in other:
                if isinstance(i, int):
                    cp.turnOn(<int>i)
                else:
                    raise TypeError("for + op, all elements in a list must be integers")
            return cp
        else:
            raise TypeError(str(type(other)) + " not supported for + op.")
    
    def __and__(SBA self, other):
        return self.__mul__(other)
    
    cdef bint getBit(self, int index):
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle
        cdef int mid_val
        while left <= right:
            middle = (right + left) >> 1
            mid_val = self.indices[middle]
            if mid_val == index:
                return True
            elif mid_val > index:
                left = middle + 1
            else:
                right = middle - 1
        return False
    
    def get(self, index1, index2 = None) -> Union[bool, SBA]:
        if index2 is None:
            return self.getBit(index1)
        else:
            if index2 > index1:
                tmp = index2
                index2 = index1
                index1 = tmp
            return self.getSection(index1, index2)
    
    def __contains__(SBA self, index):
        if isinstance(index, int):
            return self.getBit(index)
        else:
            return False

    def __mul__(SBA self, other):
        if isinstance(other, SBA):
            return SBA.andb(self, other)
        elif isinstance(other, int):
            return self.getBit(other)
        elif isinstance(other, float):
            return self.cp().subsample(other)
        elif hasattr(other, "__getitem__"):
            r = []
            for i in other:
                r.append(self.__mul__(i))
            return r
        else:
            raise TypeError(str(type(other)) + " not supported for * or & ops.")
    
    def __or__(self, other):
        return self.__add__(other)
    
    def __xor__(SBA self, SBA other):
        return SBA.xorb(self, other)
    
    cpdef rm(self, SBA rm):
        self.raiseIfViewing()
        cdef int a_from = 0
        cdef int a_to = 0
        cdef int a_val
        cdef int rm_offset = 0
        cdef int rm_val
        while a_from < self.len.len:
            if rm_offset < rm.len.len:
                a_val = self.indices[a_from]
                rm_val = rm.indices[rm_offset]
                if rm_val > a_val:
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
        self._shorten_if_needed()
        return self
    
    def __sub__(self, other):
        cdef SBA cp = self.cp()
        if isinstance(other, int):
            cp.turnOff(<int>other)
            return cp
        elif isinstance(other, SBA):
            cp.rm(other)
            return cp
        elif hasattr(other, "__getitem__"):
            for i in other:
                if isinstance(i, int):
                    cp.turnOff(<int>i)
                else:
                    raise TypeError("for - op, all elements in a list must be integers")
            return cp
        else:
            raise TypeError(str(type(other)) + " not supported for - op.")
    
    cpdef shift(self, int n):
        self.raiseIfViewing()
        for i in range(self.len.len):
            self.indices[i] += n
        return self
    
    def __lshift__(self, int n):
        return self.cp().shift(n)

    def __rshift__(self, int n):
        return self.cp().shift(-n)
    
    @staticmethod
    cdef seedRand():
        srand(<unsigned int>time(NULL))

    @staticmethod
    def seed_rand():
        SBA.seedRand()

    @staticmethod
    def rand_int() -> int:
        return rand()
    
    cdef bint compare(self, SBA other, int op):
        if op == Py_EQ or op == Py_NE:
            if self.len.len != other.len.len:
                return False if op == Py_EQ else True
            for i in range(self.len.len):
                if self.indices[i] != other.indices[i]:
                    return False if op == Py_EQ else True
            return True if op == Py_EQ else False
        else:
            for i in range(min(self.len.len, other.len.len)):
                if self.indices[i] < other.indices[i]:
                    return True if op == Py_LE or op == Py_LT else False
            if self.len.len == other.len.len:
                return op == Py_LE or op == Py_GE
            else:
                if self.len.len < other.len.len:
                    return op == Py_LE or op == Py_LT
                else:
                    return op == Py_GE or op == Py_GT

    def __richcmp__(self, other, int op):
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
            return False
    
    cpdef SBA subsample(self, float amount):
        self.raiseIfViewing()
        if amount < 0 or amount > 1:
            raise SBAException("Subsample amount must be from 0 to 1, inclusively")
        cdef int check_val = <int>(amount * RAND_MAX)
        cdef int to_offset = 0
        cdef int from_offset = 0
        while from_offset < self.len.len:
            if rand() < check_val:
                self.indices[to_offset] = self.indices[from_offset]
                to_offset += 1
            from_offset += 1
        self.len.len = to_offset
        return self
    
    @staticmethod
    cdef SBA encodeLinear(float input, int num_on_bits, int length):
        if num_on_bits > length:
            raise SBAException("The number of ON bits can't exceed the length of the array.")
        cdef SBA ret = SBA.fromCapacity(num_on_bits, False)
        ret.len.len = num_on_bits
        cdef int start_offset = <int>roundf((length - num_on_bits) * input)
        for i in range(num_on_bits):
            ret.indices[i] = start_offset + num_on_bits - i - 1
        return ret
    
    @staticmethod
    cdef SBA encodePeriodic(float input, float period, int num_on_bits, int length):
        if input < 0:
            input *= -1
        if num_on_bits > length:
            raise SBAException("The number of ON bits can't exceed the length of the array.")
        cdef SBA ret = SBA.fromCapacity(num_on_bits, False)
        ret.len.len = num_on_bits
        cdef float progress = input / period
        progress = progress - <int>progress
        cdef int start_offset = <int>roundf(progress * length)
        cdef int num_wrapper
        cdef int num_leading
        if start_offset + num_on_bits > length:
            num_wrapper = start_offset + num_on_bits - length
            while num_wrapper >= 0:
                num_wrapper -= 1
                ret.indices[num_on_bits - num_wrapper - 1] = num_wrapper
            num_leading = length - num_wrapper
            while num_leading >= 0:
                num_leading -= 1
                ret.indices[num_leading] = length - num_leading - 1
        else:
            for i in range(num_on_bits):
                ret.indices[i] = num_on_bits - i - 1 + start_offset
        return ret

    @staticmethod
    def encode(float input, int num_on_bits, int size, float period = 0):
        if period == 0:
            return SBA.encodeLinear(input, num_on_bits, size)
        else:
            return SBA.encodePeriodic(input, period, num_on_bits, size)
