# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from __future__ import print_function
import numpy as np
cimport numpy as np
np.import_array()

from cpython cimport Py_buffer, PyObject_CheckBuffer, PyBUF_WRITABLE, PyBUF_FORMAT, PyBUF_ND, PyBUF_STRIDES
from cpython.object cimport Py_EQ, Py_NE, Py_LT, Py_LE, Py_GE, Py_GT
from cpython.mem cimport PyMem_Realloc, PyMem_Malloc, PyMem_Free
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.string cimport memset, memcpy, memmove
from libc.limits cimport INT_MAX
from typing import Iterable, Union, Callable

'''
There's two different ways of handling memory allocations when doing operations. It's not apparent which is better, so I'm leaving both as an option for now.
ALLOC_THEN_SHRINK = True:
    Allocate the maximum size required for the result, based off of the lengths of the operands. Then, shrink the result to the actb ual used size when complete.
    For example, if A is of length 2, and B is of length 3, then A AND B will have at max a length of 2.
    This strategy would allocate an array of length 2, t hen shrink it down when the AND op is complete to free up the excess.
ALLOC_THEN_SHRINK = False:
    Realloc up from a length of 0 until the operation is done, according to SBA.lengthenIfNeeded
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
    const short
    const int
    const long
    const float
    const double

class SBAException(Exception):
    pass

cdef union SBALen:
    Py_ssize_t ssize_t_len # for use in buffer protocol
    int len # for normal use in SBA

cdef bint do_sba_verify = True

cdef class SBA:
    cdef int views # number of references to buffer
    cdef int cap # capacity, mem allocated for the indices
    cdef SBALen len # length, the number of ON bits in the array
    cdef int* indices # contains indices of bits that are ON.

    # snake case indicates the python function, lower camel case is for the c function.
    # cython doesn't support cpdef with @staticmethod, which neccessitates this ^

    def __init__(self):
        pass

    @staticmethod
    cdef verifyInput(bint enable):
        do_sba_verify = enable
    
    def verify_input(bint enable):
        SBA.verifyInput(enable)

    cdef inline int raiseIfViewing(self) except -1:
        if self.views > 0:
            raise SBAException("Buffer is still being viewed, or is not owned!")
    
    cdef inline void range(self, int start_inclusive, int stop_inclusive):
        '''
        sets the indicies' values  
        ensure start <= stop  
        [start, start + 1, ..., stop - 1, stop]
        '''
        cdef int i = 0
        cdef int val = start_inclusive
        while val <= stop_inclusive:
            self.indices[i] = val
            i += 1
            val += 1
    
    cdef inline lengthenIfNeeded(self, int length_override = -1):
        '''
        lengthen the indices if the capacity has been reached.  
        if length_override is specified, then use it as the current length instead of the SBA's length.
        '''
        if (self.len.len if length_override == -1 else length_override) >= self.cap: # ==
            self.cap = self.cap + (self.cap >> 1) + 1; # cap = cap * 1.5 + 1, estimate for golden ratio (idk python uses the same strat)
            self.indices = <int*>PyMem_Realloc(self.indices, self.cap * sizeof(self.indices[0]))
    
    cdef inline shorten(self):
        self.cap = self.len.len
        self.indices = <int*>PyMem_Realloc(self.indices, sizeof(self.indices[0]) * self.cap)

    cdef inline shortenIfNeeded(self):
        if STRICT_SHORTEN or self.len.len < self.cap >> 1:
            self.shorten()
    
    @staticmethod
    cdef SBA fromRange(int start_inclusive, int stop_inclusive):
        if start_inclusive > stop_inclusive:
            raise SBAException("start must be <= stop")
        cdef SBA ret = SBA.__new__(SBA)
        # ret.views = 0 # implicit since initialization guarantees 0s in memory. Keep this in mind for all factory methods.
        ret.cap = stop_inclusive - start_inclusive + 1
        ret.indices = <int*>PyMem_Malloc(sizeof(ret.indices[0]) * ret.cap)
        ret.range(start_inclusive, stop_inclusive)
        ret.len.len = ret.cap
        return ret

    @staticmethod
    def from_range(start_inclusive, stop_inclusive):
        return SBA.fromRange(start_inclusive, stop_inclusive)

    @staticmethod
    cdef SBA fromCapacity(int cap, bint default = True):
        '''
        the `default` parameter is not documented in the python stub since is should only be used in this file.
        if default is True:  
            initalizes the indices such that they are descending from cap-1 to 0.  
        else:  
            allocate the capacity but leave it uninitialized, and sets this SBA's length to 0.
        '''
        if cap < 0:
            raise SBAException("cap must be non-negative!")
        cdef SBA ret = SBA.__new__(SBA)
        ret.cap = cap
        ret.indices = <int*>PyMem_Malloc(sizeof(ret.indices[0]) * ret.cap)
        if default:
            ret.len.len = ret.cap
            ret.range(0, ret.len.len - 1)
        return ret
    
    @staticmethod
    def from_capacity(cap=0):
        return SBA.fromCapacity(cap)
    
    @staticmethod
    def from_iterable(obj: Iterable[int], filter: Union[None, Callable[[Union[int, float]], bool]] = None, *, bint reverse = False, verify = None):
        cdef SBA ret = SBA.__new__(SBA)
        cdef int ln = <int>len(obj)
        if filter is None:
            # iterable is a sparse array
            if do_sba_verify if verify is None else verify:
                for i in range(ln - 1):
                    if type(obj[i]) is not int:
                        raise SBAException("Indices must be ints.")
                    if obj[i] >= obj[i + 1]:
                        raise SBAException("Indices must be in ascending order, with no duplicates.")
            ret.cap = ln
            ret.indices = <int*>PyMem_Malloc(sizeof(ret.indices[0]) * ret.cap)
            ret.len.len = ret.cap
            for i in range(ln):
                ret.indices[i] = obj[i] # implicit check within c int range for each element
        else:
            # iterable is a dense array
            i = 0
            while i < ln:
                if filter(obj[ln - i - 1 if reverse else i]):
                    ret.lengthenIfNeeded()
                    ret.indices[ret.len.len] = i
                    ret.len.len += 1
                i += 1
        return ret

    @staticmethod
    def from_buffer(const_numeric[:] buf, filter: Union[None, Callable[[Union[int, float]], bool]] = None, *, bint copy = True, bint reverse = False, verify = None):
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
                ret.indices = <int*>PyMem_Malloc(sizeof(ret.indices[0]) * ret.cap)
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
                    ret.lengthenIfNeeded()
                    ret.indices[ret.len.len] = i
                    ret.len.len += 1
                i += 1
        return ret 

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
    
    cdef np.ndarray toBuffer(self, bint give_ownership):
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
    
    def to_buffer(self, give_ownership = True):
        return self.toBuffer(give_ownership)
    
    def __dealloc__(self):
        if self.views == 0: # owner?
            PyMem_Free(self.indices)
    
    # =========================================================================================

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

        self.lengthenIfNeeded()
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
                self.shortenIfNeeded()
                return
            elif mid_val < index:
                left = middle + 1
            else:
                right = middle - 1
    
    cpdef set(self, int index, bint state):
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
    
    cdef SBA getSection(self, int start_inclusive, int stop_inclusive):
        # start <= stop
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
                ret.lengthenIfNeeded()
            ret.indices[ret.len.len] = mid_val
            ret.len.len += 1
            middle += 1
            if middle >= self.len.len:
                break # ran off end
            mid_val = self.indices[middle]
        if ALLOC_THEN_SHRINK or STRICT_SHORTEN:
            ret.shorten()
        return ret
    
    def __getitem__(self, index: Union[int, slice]) -> Union[int, SBA]:
        if isinstance(index, slice):
            start = self.indices[0] if index.start is None else index.start
            stop = self.indices[len(self) - 1] if index.stop is None else index.stop
            return self.get(start, stop)
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
                r.lengthenIfNeeded(r_len[0])
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
            elif mid_val < index:
                left = middle + 1
            else:
                right = middle - 1
        return False
    
    cpdef get(self, int index1, index2 = None):
        if index2 is None:
            return self.getBit(index1)
        else:
            if index2 < index1:
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
        self.shortenIfNeeded()
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
    
    cpdef bint compare(self, SBA other, int op):
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
            ret.indices[i] = start_offset + i
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
            return SBA.encodeLinear(input, num_on_bits, size)
        else:
            return SBA.encodePeriodic(input, period, num_on_bits, size)
