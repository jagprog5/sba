# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from __future__ import print_function
import numpy as np
cimport numpy as np
np.import_array()

import math

from cpython cimport Py_buffer, PyObject_CheckBuffer, PyBUF_WRITABLE, PyBUF_FORMAT, PyBUF_ND, PyBUF_STRIDES
from cpython.mem cimport PyMem_Realloc, PyMem_Free
from libc.string cimport memset, memcpy, memmove
from typing import Iterable, Union, ByteString

ctypedef np.uint32_t uint

cdef extern from "limits.h":
    uint UINT32_MAX

cdef extern from "numpy/arrayobject.h":
    void PyArray_CLEARFLAGS(np.ndarray arr, int flags)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    int PyArray_FLAGS(np.ndarray arr)
    object PyArray_SimpleNewFromData(int nd, np.npy_intp* dims, int typenum, void* data)
    object PyArray_FromBuffer(object buf, np.dtype dtype, np.npy_intp count, np.npy_intp offset)
    np.npy_intp* PyArray_DIMS(np.ndarray arr)
    void* PyArray_DATA(np.ndarray arr)

class SBAException(Exception):
    pass

cdef union SBALen: # Needs little-endian
    Py_ssize_t ssize_t_len # for use in buffer protocol
    uint len # for normal use in SBA

cdef class SBA:
    cdef uint views # number of references to buffer
    cdef uint cap # capacity, mem allocated for the indices
    cdef SBALen len # length, the number of ON bits in the array
    cdef uint* indices # contains indices of bits that are ON. From MSB to LSB, aka descending index value
    
    def __init__(self, arg: Union[int, Iterable[int], ByteString, np.ndarray, None]):
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
    
    cdef _init(self, uint i):
        ''' Init some members. This is used in factory methods. '''
        self.views = 0
        self.cap = i
        self.len.len = i
        memset(&self.len.len + 1, 0, sizeof(Py_ssize_t) - sizeof(uint)) # clear rest of union
    
    cdef _raise_if_viewing(self):
        if self.views != 0:
            raise SBAException("Buffer is still being viewed, or is not owned!")
    
    def set_from_iterable(self, obj: Iterable[int], bint check_valid = True):
        ''' Replaces this SBA's data with the iterable's data. Deep-copies. '''
        self._raise_if_viewing()
        if check_valid:
            for i in obj:
                if not isinstance(i, int):
                    raise SBAException("Integers only.")
            for i in range(len(obj) - 1):
                if obj[i] <= obj[i + 1]:
                    raise SBAException("Indices must be in descending order, with no duplicates.")
            if obj[-1] < 0:
                raise SBAException("Must be non-negative.")
            if obj[0] > UINT32_MAX:
                raise SBAException("Element exceeds UINT32_MAX")
        self._init(<uint>len(obj))
        self.indices = <uint*>PyMem_Realloc(self.indices, sizeof(self.indices[0]) * len(obj))
        for i in range(len(obj)):
            self.indices[i] = obj[i]
    
    @staticmethod
    def from_iterable(obj: Iterable[int], bint check_valid = True) -> SBA:
        '''
        Deep-copy iterable to create and init SBA instance.  
        check_valid: check that all elements are valid (integers, in uint32 range, descending order, no duplicates).  
        '''
        cdef SBA ret = SBA.__new__(SBA)
        ret._set_from_iterable(obj, check_valid)
        return ret
    
    cdef inline void _default(self):
        ''' Set [len - 1, ..., 2, 1, 0] '''
        cdef uint i = 0
        cdef biggest_val = self.len.len - 1
        while i < self.len.len:
            self.indices[i] = biggest_val - i
            i += 1

    cdef set_from_capacity(self, uint initial_capacity = 0, bint set_default = 1):
        '''
        Replace's this SBA's data.  
        set_default: set to default value otherwise leaves uninitialized.  
        '''
        self._raise_if_viewing()
        self._init(initial_capacity)
        self.indices = <uint*>PyMem_Realloc(self.indices, initial_capacity * sizeof(self.indices[0]))
        if set_default:
            self._default()

    @staticmethod
    cdef SBA c_from_capacity(uint initial_capacity = 0, bint set_default = 1):
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
    cdef inline bint _is_valid(const uint[:] arr):
        cdef uint i = 0
        cdef uint len = <uint>arr.shape[0]
        while i < len - 1:
            if arr[i] <= arr[i + 1]:
                return 0
            i += 1
        return 1

    cdef set_from_buf(self, const uint[:] buf):
        self._raise_if_viewing()
        if not SBA._is_valid(buf):
            raise SBAException("The buffer doesn't have valid indices.")
        cdef uint len = <uint>buf.shape[0]
        self._init(len)
        self.indices = <uint*>PyMem_Realloc(self.indices, len * sizeof(self.indices[0]))
        memcpy(self.indices, &buf[0], len * sizeof(self.indices[0]))

    cdef set_from_np(self, np.ndarray[np.uint32_t, ndim=1] arr, bint deep_copy = 1, bint check_valid = 1):
        ''' Replace this SBA's data with the array's data. '''
        self._raise_if_viewing()
        if check_valid and not SBA._is_valid(arr):
            raise SBAException("The numpy array doesn't have valid indices.")
        cdef uint len = <uint>PyArray_DIMS(arr)[0]
        cdef uint* data = <uint*>PyArray_DATA(arr)
        cdef int flags

        self._init_len(len)
        if deep_copy:
            self.indices = <uint*>PyMem_Realloc(self.indices, len * sizeof(self.indices[0]))
            memcpy(self.indices, data, len * sizeof(data[0]))
        else:
            flags = PyArray_FLAGS(arr)
            if not flags & np.NPY_ARRAY_OWNDATA:
                raise BufferError("Array did not have ownership to give!")
            if not flags & np.NPY_ARRAY_WRITEABLE:
                raise BufferError("Array is not writable!")
            PyMem_Free(self.indices)
            self.indices = data
            PyArray_CLEARFLAGS(arr, np.NPY_ARRAY_WRITEABLE)
            PyArray_CLEARFLAGS(arr, np.NPY_ARRAY_OWNDATA)
            ''' Suppose array a is the owner of the data and is writable, and array b is made with reference to array a's buffer.
                Array a is then set to read-only. This will NOT update the writability of b.
                If a is then passed as the array arg to this function, and then the data is modified via b,
                then the indices inside the SBA are probably not valid (descending order, etc.).
                Weird stuff might happen (not segfaults, but meaningless op return values).
                ===== In general, don't modify the data while SBA is working with it. ===== '''
    
    @staticmethod
    cdef SBA c_from_np(np.ndarray[np.uint32_t, ndim=1] arr, bint deep_copy = 1, bint check_valid = 1):
        ''' cython doen't allow @staticmethod with cpdef. This is wrapped by from_np. '''
        cdef SBA ret = SBA.__new__(SBA)
        ret.set_from_np(arr, deep_copy, check_valid)
        return ret
    
    @staticmethod
    def from_np(np_arr, deep_copy = True, check_valid = True):
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
            buffer.format = 'I'
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
            return PyArray_FromBuffer(self, np.uint32, -1, 0)

        self._raise_if_viewing()
        cdef np.npy_intp* dims = <np.npy_intp*>&self.len.ssize_t_len
        cdef np.ndarray arr = PyArray_SimpleNewFromData(1, dims, np.NPY_UINT32, self.indices)
        PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
        self.indices = NULL
        self.cap = 0
        self.len.len = 0
        return arr
    
    def __dealloc__(self):
        PyMem_Free(self.indices)
    
    # =========================================================================================

    cdef shorten(self):
        self.cap = self.len.len
        self.indices = <uint*>PyMem_Realloc(self.indices, sizeof(self.indices[0]) * self.cap)

    cpdef printSBA(SBA a):
        cdef uint amount = 0
        cdef uint val
        if a.len.len != 0:
            for i in range(a.len.len - 1):
                val = a.indices[i]
                amount += 2 + (0 if val == 0 else math.floor(math.log10f(val)))
            for i in range(amount):
                print(' ', end='')
            print("V")
        for i in range(a.cap):
            print(a.indices[i] + " ", end='')
        print('')
    
    cpdef turnOn(self, int index):
        cdef int left = 0
        cdef int right = self.len.len - 1
        cdef int middle = 0
        cdef uint mid_val = 0
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

        if self.len.len >= self.cap:
            self.cap = self.cap + (self.cap >> 1) + 1; # cap *= 1.5 + 1, estimate for golden ratio
            self.indices = <uint*>PyMem_Realloc(self.indices, self.cap * sizeof(self.indices[0]))
        # cdef uint* ptr = &self.indices[0] # double check ptr arith
        memmove(self.indices + middle + 1, self.indices + middle, (self.len.len - middle) * sizeof(self.indices[0]))
        self.len.len += 1
        self.indices[middle] = index

    # def encode_periodic(input: float, period: float, num_on_bits: int, size: int) -> SBA:
    #     '''
    #     input is the the value to encode. it is encoded linearly, except its encoding wraps back to 0 as it approaches period  
    #     num_on_bits is the number of bits that will be flipped to ON  
    #     size is the total size of the array
    #     '''
    #     if period <= 0:
    #         raise SBAException("Period must be positive.")
    #     elif num_on_bits > size:
    #         raise SBAException("The number of on bits can't exceed the size of the array.")
    #     ret = SBA.from_capacity(num_on_bits, False)

        # uint_fast32_t cap = r->capacity;
        # float progress = input / period;
        # progress = progress - (int)progress;
        # uint_fast32_t start_offset = roundf(progress * n);
        # if (start_offset + ret.cap > n) {
        #     uint_fast32_t num_wrapped = start_offset + ret.cap - n;
        #     uint_fast32_t num_leading = n - start_offset;
        #     do {
        #         --num_leading;
        #         r->indices[num_leading] = n - num_leading - 1;
        #     } while (num_leading > 0);
        #     do {
        #         --num_wrapped;
        #         r->indices[ret.cap - num_wrapped - 1] = num_wrapped;
        #     } while (num_wrapped > 0);
        # } else {
        #     for (uint_fast32_t i = 0; i < ret.cap; ++i) {
        #         r->indices[i] = ret.cap - i - 1 + start_offset;
        #     }
        # }

        # SBA.encodePeriodic(c.c_float(input), c.c_float(period), c.c_uint32(size), c.byref(r))
        # return r
