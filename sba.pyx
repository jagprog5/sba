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
from libc.limits cimport INT_MAX, INT_MIN
from typing import Iterable, Union, ByteString

cdef extern from "math.h":
    float floorf(float)
    float log10f(float)
    int roundf(float)

cdef extern from "time.h":
    void* time(void*)

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

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
        ''' This is enabled by default. On creation of an SBA, ensures that indices are valid. '''
        do_sba_checking = 1
    
    @staticmethod
    def disable_checking():
        '''
        Disables check that ensure that indices are valid on SBA creation.  
        Overrides argument param 'check_valid', throughout.  
        '''
        do_sba_checking = 0
    
    def __init__(self, arg: Union[int, Iterable[int], ByteString, np.ndarray, None] = None):
        '''
        Initializations through __init__ will deep_copy the arg (if the arg is of appropriate type).  
        For zero-copy, see from_np().
        '''
        if arg is None:
            arg = 0
        if isinstance(arg, int):
            if arg < 0:
                raise SBAException("Requires non-negative integer.")
            self.set_from_capacity(arg)
            return
        if isinstance(arg, np.ndarray):
            self.set_from_np(arg)
            return
        if PyObject_CheckBuffer(arg):
            self.set_from_buf(arg)
            return
        self.set_from_iterable(arg)
    
    cdef _init(self, int i):
        ''' Init some members. This is used in factory methods. '''
        self.views = 0
        self.cap = i
        self.len.len = i
        memset(&self.len.len + 1, 0, sizeof(Py_ssize_t) - sizeof(int)) # clear rest of union
    
    cdef int _raise_if_viewing(self) except -1:
        if self.views > 0:
            raise SBAException("Buffer is still being viewed, or is not owned!")
    
    def view_count(self) -> int:
        return self.views
    
    def set_from_iterable(self, obj: Iterable[int], bint check_valid = True):
        ''' Replaces this SBA's data with the iterable's data. Deep-copies. '''
        self._raise_if_viewing()
        cdef int ln = <int>len(obj)
        if do_sba_checking and check_valid:
            for i in obj:
                if not isinstance(i, int):
                    raise SBAException("Integers only.")
            for i in range(ln - 1):
                if obj[i] <= obj[i + 1]:
                    raise SBAException("Indices must be in descending order, with no duplicates.")
            if obj[-1] < INT_MIN:
                raise SBAException("Element exceeds INT_MIN")
            if obj[0] > INT_MAX:
                raise SBAException("Element exceeds INT_MAX")
        self._init(ln)
        self.indices = <int*>PyMem_Realloc(self.indices, sizeof(self.indices[0]) * ln)
        for i in range(ln):
            self.indices[i] = obj[i]
    
    @staticmethod
    def from_iterable(obj: Iterable[int], bint check_valid = True) -> SBA:
        '''
        Deep-copy iterable to create and init SBA instance.  
        check_valid: check that all elements are valid (integers, in int range, descending order, no duplicates).  
        '''
        cdef SBA ret = SBA.__new__(SBA)
        ret._set_from_iterable(obj, check_valid)
        return ret
    
    cdef inline void _default(self):
        ''' Set [len - 1, ..., 2, 1, 0] '''
        cdef int i = 0
        cdef biggest_val = self.len.len - 1
        while i < self.len.len:
            self.indices[i] = biggest_val - i
            i += 1

    cdef set_from_capacity(self, int initial_capacity = 0, bint set_default = 1):
        '''
        Replace's this SBA's data.  
        set_default: set to default value otherwise leaves uninitialized.  
        '''
        self._raise_if_viewing()
        self._init(initial_capacity)
        self.indices = <int*>PyMem_Realloc(self.indices, initial_capacity * sizeof(self.indices[0]))
        if set_default:
            self._default()

    @staticmethod
    cdef SBA c_from_capacity(int initial_capacity = 0, bint set_default = 1):
        ''' cython doen't allow @staticmethod with cpdef. This is wrapped by from_capacity. '''
        cdef SBA ret = SBA.__new__(SBA)
        ret.set_from_capacity(initial_capacity, set_default)
        return ret

    @staticmethod
    def from_capacity(initial_capacity = 0) -> SBA:
        '''
        Create and init an empty SBA with specified initial capacity.  
        sets the indices to a default value [len - 1, ..., 3, 2, 1, 0].  
        '''
        if initial_capacity < 0:
            raise SBAException("Capacity must be positive.")
        return SBA.c_from_capacity(initial_capacity, True)

    @staticmethod
    cdef inline bint _is_valid(const int[:] arr):
        cdef int i = 0
        cdef int len = <int>arr.shape[0]
        while i < len - 1:
            if arr[i] <= arr[i + 1]:
                return 0
            i += 1
        return 1

    cdef set_from_buf(self, const int[:] buf):
        self._raise_if_viewing()
        if do_sba_checking and not SBA._is_valid(buf):
            raise SBAException("The buffer doesn't have valid indices.")
        cdef int len = <int>buf.shape[0]
        self._init(len)
        self.indices = <int*>PyMem_Realloc(self.indices, len * sizeof(self.indices[0]))
        memcpy(self.indices, &buf[0], len * sizeof(self.indices[0]))

    cdef set_from_np(self, np.ndarray[np.int32_t, ndim=1] arr, bint deep_copy = 1, bint check_valid = 1):
        self._raise_if_viewing()
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
    cdef SBA c_from_np(np.ndarray[np.int32_t, ndim=1] arr, bint deep_copy = 1, bint check_valid = 1):
        ''' cython doen't allow @staticmethod with cpdef. This is wrapped by from_np. '''
        cdef SBA ret = SBA.__new__(SBA)
        ret.set_from_np(arr, deep_copy, check_valid)
        return ret
    
    @staticmethod
    def from_np(np_arr, deep_copy = True, check_valid = True) -> SBA:
        '''
        Creates and initalizes an SBA from a numpy array.
        deep_copy:
            true: The sba gets a separate copy of the data.
            false: The sba gets a read-only reference to the data.
        '''
        return SBA.c_from_np(np_arr, deep_copy, check_valid)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        ''' Buffer protocol. Read-only '''
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
    
    cpdef np.ndarray to_np(self, bint give_ownership = 1):
        '''
        Create a numpy array.
        give_ownership:  
            true: Makes the returned numpy array the owner of the data, and clears this SBA's reference to the data.  
            false: The returned numpy array gets a read-only buffer to the data.
        '''
        if not give_ownership:
            return np.frombuffer(memoryview(self), dtype=int)

        self._raise_if_viewing()
        cdef np.npy_intp* dims = <np.npy_intp*>&self.len.ssize_t_len
        cdef np.ndarray arr = np.PyArray_SimpleNewFromData(1, dims, np.NPY_INT, self.indices)
        PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
        self.indices = NULL
        self.cap = 0
        self.len.len = 0
        return arr
    
    def __dealloc__(self):
        if self.views == 0: # owner?
            PyMem_Free(self.indices)
    
    # =========================================================================================

    cdef shorten(self):
        self.cap = self.len.len
        self.indices = <int*>PyMem_Realloc(self.indices, sizeof(self.indices[0]) * self.cap)

    cdef shorten_if_needed(self):
        if self.len.len < self.cap // 2:
            self.shorten()
    
    cdef lengthen_if_needed(self):
        if self.len.len >= self.cap: # ==
            self.cap = self.cap + (self.cap >> 1) + 1; # cap *= 1.5 + 1, estimate for golden ratio
            self.indices = <int*>PyMem_Realloc(self.indices, self.cap * sizeof(self.indices[0]))

    cpdef print_raw(self):
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
    
    cdef turnOn(self, int index):
        self._raise_if_viewing()
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle = 0
        cdef int mid_val = INT_MIN
        while left <= right:
            middle = (right + left) // 2
            mid_val = self.indices[middle]
            if mid_val > index:
                left = middle + 1
            elif mid_val < index:
                right = middle - 1
            else:
                return; # skip duplicate

        if index < mid_val:
            middle += 1

        self.lengthen_if_needed()
        memmove(self.indices + middle + 1, self.indices + middle, (self.len.len - middle) * sizeof(self.indices[0]))
        self.len.len += 1
        self.indices[middle] = index
    
    cdef turnOff(self, int index):
        self._raise_if_viewing()
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle = 0
        cdef int mid_val
        while left <= right:
            middle = (right + left) // 2
            mid_val = self.indices[middle]
            if mid_val == index:
                self.len.len -= 1
                memmove(self.indices + middle, self.indices + middle + 1, (self.len.len - middle) * sizeof(self.indices[0]))
                self.shorten_if_needed()
                return
            elif mid_val > index:
                left = middle + 1
            else:
                right = middle - 1
    
    cpdef set_bit(self, int index, bint state=1):
        if state:
            self.turnOn(index)
        else:
            self.turnOff(index)
    
    cdef int _check_index(self, int index) except -1:
        if index >= self.len.len:
            raise SBAException("Index out of bounds.")
        if index < 0:
            index = self.len.len + index
            if index < 0:
                raise SBAException("Index out of bounds.")
        return index
    
    def __delitem__(self, index):
        ''' Turns the i-th ON bit to OFF '''
        self._raise_if_viewing()
        cdef int i = self._check_index(index)
        self.len.len -= 1
        memmove(&self.indices[i], &self.indices[i + 1], sizeof(int) * (self.len.len - i))

    def __setitem__(self, index: int, value: int):
        '''
        Turns off the index-th ON bit, and turns on the value-th bit.  
        Not to be confused with set_bit
        ```python
        >>> a = SBA([15, 10, 5, 0])
        >>> a[2] = 6
        >>> a
        [15 10 6 0]
        ```
        '''
        self.__delitem__(index)
        self.turnOn(value)
    
    cdef SBA getSection(self, int stop_inclusive, int start_inclusive):
        cdef SBA ret = SBA.c_from_capacity(stop_inclusive - start_inclusive + 1, False)
        ret.len.len = 0
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle
        cdef int mid_val
        while left <= right:
            middle = (right + left) // 2
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
            ret.indices[ret.len.len] = mid_val
            ret.len.len += 1
            middle += 1
            if middle >= self.len.len:
                break # ran off end
            mid_val = self.indices[middle]
        return ret
    
    def __getitem__(self, index: Union[int, slice]) -> int:
        '''
        Note that __getitem__ completes two totally different operation with a slice versus with an int.  

        =-======For an int=========  
        Returns the index of the i-th ON bit.  
        Not to be confused with get_bit.
        ```python
        >>> SBA([4, 3, 2, 1])[-2]
        2
        ```
        =-======For a slice========  
        Returns the ON bits within the specified range (inclusive stop, inclusive start).  
        The specified step is ignored (uses 1).  
        ```python
        >>> SBA(]15, 10, 2, 1])[15:2]
        [15 10 2]
        >>> SBA(range(0, 10000, 2))[100:110]
        [110 108 106 104 102 100]
        ```
        '''
        if isinstance(index, slice):
            start = self.indices[0] if index.start is None else index.start
            stop = self.indices[len(self) - 1] if index.stop is None else index.stop
            if start > stop:
                tmp = start
                start = stop
                stop = tmp
            return self.getSection(stop, start)
        else:
            return self.indices[self._check_index(index)]

    cpdef SBA cp(self):
        ''' Creates deep-copy. '''
        cdef SBA ret = SBA.c_from_capacity(self.len.len, 0)
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

    @staticmethod
    cdef orBits(void* r, SBA a, SBA b, bint exclusive, bint size_only):
        cdef bint a_nempty = 1
        cdef int a_offset = 0
        cdef int a_val
        cdef bint b_nempty = 1
        cdef int b_offset = 0
        cdef int b_val
        cdef int r_size = 0

        SBA._get_both(a, &a_offset, &a_val, &a_nempty, b, &b_offset, &b_val, &b_nempty)
        while a_nempty or b_nempty:
            if (a_nempty and not b_nempty) or (a_nempty and b_nempty and a_val > b_val):
                if not size_only:
                    (<SBA>r).indices[r_size] = a_val
                r_size += 1
                SBA._get_one(a, &a_offset, &a_val, &a_nempty)
            elif (not a_nempty and b_nempty) or (a_nempty and b_nempty and a_val < b_val):
                if not size_only:
                    (<SBA>r).indices[r_size] = b_val
                r_size += 1
                SBA._get_one(b, &b_offset, &b_val, &b_nempty)
            elif a_nempty and b_nempty and a_val == b_val:
                if not exclusive:
                    if not size_only:  
                        (<SBA>r).indices[r_size] = a_val
                    r_size += 1
                SBA._get_both(a, &a_offset, &a_val, &a_nempty, b, &b_offset, &b_val, &b_nempty)
        if size_only:
            (<int*>r)[0] = r_size
        else:
            (<SBA>r).len.len = r_size
    
    @staticmethod
    def or_bits(SBA a not None, SBA b not None) -> SBA:
        cdef SBA ret = SBA.c_from_capacity(a.len.len + b.len.len, False)
        SBA.orBits(<void*>ret, a, b, 0, 0)
        return ret
    
    @staticmethod
    def or_size(SBA a not None, SBA b not None) -> int:
        cdef int ret
        SBA.orBits(<void*>&ret, a, b, 0, 1)
        return ret
    
    @staticmethod
    def xor_bits(SBA a not None, SBA b not None) -> SBA:
        cdef SBA ret = SBA.c_from_capacity(a.len.len + b.len.len, False)
        SBA.orBits(<void*>ret, a, b, 1, 0)
        return ret
    
    @staticmethod
    def xor_size(SBA a not None, SBA b not None) -> int:
        cdef int ret
        SBA.orBits(<void*>&ret, a, b, 1, 1)
        return ret
    
    @staticmethod
    cdef andBits(void* r, SBA a, SBA b, bint size_only):
        cdef bint a_nempty = 1
        cdef int a_offset = 0
        cdef int a_val
        cdef int a_size = a.len.len # store in case r = a
        cdef bint b_nempty = 1
        cdef int b_offset = 0
        cdef int b_val
        cdef int b_size = b.len.len # store in case r = b
        cdef int r_size = 0

        if a_size == 0 or b_size == 0:
            if size_only:
                (<int*>r)[0] = r_size
            return

        SBA._get_both(a, &a_offset, &a_val, &a_nempty, b, &b_offset, &b_val, &b_nempty)
        while a_nempty and b_nempty:
            if a_val > b_val:
                SBA._get_one(a, &a_offset, &a_val, &a_nempty)
            elif a_val < b_val:
                SBA._get_one(b, &b_offset, &b_val, &b_nempty)
            else: # ==
                if not size_only:
                    (<SBA>r).indices[r_size] = a_val
                r_size += 1
                SBA._get_both(a, &a_offset, &a_val, &a_nempty, b, &b_offset, &b_val, &b_nempty)
        if size_only:
            (<int*>r)[0] = r_size
        else:
            (<SBA>r).len.len = r_size
    
    @staticmethod
    def and_bits(SBA a not None, SBA b not None) -> SBA:
        cdef SBA ret = SBA.c_from_capacity(min(a.len.len, b.len.len), False)
        SBA.andBits(<void*>ret, a, b, 0)
        return ret
    
    @staticmethod
    def and_size(SBA a not None, SBA b not None) -> int:
        cdef int ret
        SBA.andBits(<void*>&ret, a, b, 1)
        return ret
    
    def __add__(self, other):
        '''
        For a string: concatenates string of self.
        ```python
        >>> SBA([2, 1]) + " hi"
        '[2 1] hi'
        ```
        For an int: Turns on the bit.
        ```python
        >>> SBA([2, 1]) + 0
        [2 1 0]
        ```
        For an SBA: return a copy of self OR other
        ```python
        >>> SBA([3, 2]) + SBA([2, 1])
        [3 2 1]
        ```
        For a iterable of ints: adds each element, separately
        >>> SBA([2, 1]) + [5, 0, 2]
        [5 2 1 0]
        '''
        if not isinstance(self, SBA): # __radd__
            if isinstance(self, str):
                return self + str(other)
            else:
                raise TypeError(str(type(self)) + " not supported for reverse + op.")

        if isinstance(other, str):
            return str(self) + other
        elif isinstance(other, int):
            cp = self.cp()
            cp.set_bit(other, True)
            return cp
        elif isinstance(other, SBA):
            return SBA.or_bits(self, other)
        elif hasattr(other, "__getitem__"):
            cp = self.cp()
            for i in other:
                if isinstance(i, int):
                    cp.set_bit(i, True)
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
            middle = (right + left) // 2
            mid_val = self.indices[middle]
            if mid_val == index:
                return True
            elif mid_val > index:
                left = middle + 1
            else:
                right = middle - 1
        return False

    def __mul__(SBA self, other):
        '''
        For an int: Returns the state of the bit.
        ```python
        >>> SBA([2, 1]) * 50
        False
        >>> SBA([2, 1]) * 2
        True
        ```
        For an SBA: ANDs the bits
        ```python
        >>> SBA([3, 2]) * SBA([2, 1])
        [2]
        ```
        For a float: returns a random subsample
        ```python
        >>> SBA([5, 4, 3, 2, 1, 0]) * (1 / 3)
        [5, 2]
        ```
        '''
        if isinstance(other, SBA):
            return SBA.and_bits(self, other)
        elif isinstance(other, int):
            return self.getBit(other)
        elif isinstance(other, float):
            return self.cp().subsample(other)
        else:
            raise TypeError(str(type(other)) + " not supported for * or & ops.")
    
    def __or__(self, other):
        return self.__add__(other)
    
    def __xor__(SBA self, SBA other):
        return SBA.xor_bits(self, other)
    
    cdef turnOffAll(self, SBA rm):
        self._raise_if_viewing()
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
        self.shorten_if_needed()

    def turn_off_all(self, SBA rm not None):
        self.turnOffAll(rm)
    
    def __sub__(self, other):
        '''
        For an int: Turns off the bit.
        ```python
        >>> SBA((2, 1)) - 2
        [1]
        ```
        For an SBA: removes all elements
        ```python
        >>> SBA((3, 2, 1)) - SBA((3, 2))
        [1]
        ```
        ```python
        For a iterable of ints: removes each element
        >>> SBA((2, 1)) - [2, 0, 5]
        [1]
        ```
        '''
        if isinstance(other, int):
            cp = self.cp()
            cp.set_bit(other, False)
            return cp
        elif isinstance(other, SBA):
            cp = self.cp()
            cp.turn_off_all(other)
            return cp
        elif hasattr(other, "__getitem__"):
            cp = self.cp()
            for i in other:
                if isinstance(i, int):
                    cp.set_bit(i, False)
                else:
                    raise TypeError("for - op, all elements in a list must be integers")
            return cp
        else:
            raise TypeError(str(type(other)) + " not supported for - op.")
    
    cpdef shift(self, int n):
        self._raise_if_viewing()
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
        ''' Call SBA.seed_rand() before using this function. '''
        self._raise_if_viewing()
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
        if input < 0:
            raise SBAException("Can't encode a negative value in this function.")
        cdef SBA ret = SBA.c_from_capacity(num_on_bits, False)
        ret.len.len = num_on_bits
        cdef int start_offset = <int>roundf((length - num_on_bits) * input)
        for i in range(num_on_bits):
            ret.indices[i] = start_offset + num_on_bits - i - 1
        return ret
    
    @staticmethod
    def encode_linear(float input, int num_on_bits, int size):
        return SBA.encodeLinear(input, num_on_bits, size)
    
    @staticmethod
    cdef SBA encodePeriodic(float input, float period, int num_on_bits, int length):
        cdef SBA ret = SBA.c_from_capacity(num_on_bits, False)
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
    def encode_periodic(float input, float period, int num_on_bits, int size):
        return SBA.encodePeriodic(input, period, num_on_bits, size)
                

