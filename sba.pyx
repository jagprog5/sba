# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
import numpy as np
cimport numpy as np
np.import_array()

from cpython.ref cimport PyObject
from cpython cimport (Py_buffer, PyObject_GetBuffer, PyObject_CheckBuffer,
    PyBUF_WRITABLE, PyBUF_FORMAT, PyBUF_ANY_CONTIGUOUS, PyBUF_ND, PyBUF_STRIDES)
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.string cimport memset, memcpy

from typing import List, Iterable, Union

ctypedef unsigned int uint

cdef extern from "numpy/arrayobject.h":
    void PyArray_CLEARFLAGS(np.ndarray arr, int flags)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void* PyArray_BASE(np.ndarray arr) # not actually void*, just checking if null

class SBAException(Exception):
    pass

cdef union SBALen: # Needs little-endian
    Py_ssize_t buffer_length # for use in buffer protocol
    uint len # for normal use in SBA

cdef class SBA:
    cdef char owner # bool
    cdef uint views # number of references to buffer
    cdef uint cap # capacity, mem currently allocated for the list, number of elements
    cdef SBALen len # length, the number of ON bits in the array
    cdef uint* indices # contains indices of bits that are ON. From MSB to LSB, aka descending index value
    
    # def __cinit__(self, obj):
    #     '''
    #     Initialize an SBA from a python object.  
    #     '''
    #     print("From buffer stuff")
    #     if not PyObject_CheckBuffer(obj):
    #         raise BufferError("Object does not support the buffer interface.")
    #     cdef Py_buffer buf
    #     cdef int result = PyObject_GetBuffer(obj, &buf, PyBUF_ANY_CONTIGUOUS & PyBUF_FORMAT)
    #     print(result)
    
    def __init__(self, obj):
        pass
    #     ''' Initialize an empty SBA with specified capacity. '''
        # print("Init from cap: " + str(initial_capacity))
        # self.views = 0
        # self.cap = initial_capacity
        # self.len.len = initial_capacity
        # memset(&self.len.len + 1, 0, sizeof(Py_ssize_t) - sizeof(uint)) # clear rest of union
        # self.indices = <uint*> PyMem_Malloc(initial_capacity * sizeof(uint))
        # pass
    
    cdef _init_len(self, uint i):
        '''Init some members. This is used in factory methods. '''
        self.views = 0
        self.cap = i
        self.len.len = i
        memset(&self.len.len + 1, 0, sizeof(Py_ssize_t) - sizeof(uint)) # clear rest of union
    
    @staticmethod
    def from_capacity(initial_capacity: int = 0):
        return SBA.c_from_capacity(initial_capacity)

    @staticmethod
    cdef SBA c_from_capacity(uint initial_capacity):
        ''' Initialize an empty SBA with specified initial capacity. '''
        cdef SBA ret = SBA.__new__(SBA)
        ret._init_len(initial_capacity)
        ret.owner = 1
        ret.indices = <uint*> PyMem_Malloc(initial_capacity * sizeof(uint))
        return ret
    
    @staticmethod
    def from_np(np_arr, deep_copy = True, take_ownership = False, check_valid = True):
        return SBA.c_from_np(np_arr, deep_copy, take_ownership, check_valid)
    
    @staticmethod
    cdef SBA c_from_np(np.ndarray[np.uint32_t, ndim=1] arr, bint deep_copy = True, bint take_ownership = False, bint check_valid = True):
        if PyArray_BASE(arr) != NULL:
            print("arr did not have ownership to give!")
        print
        pass

    # def from_buffer(const uint[::1] view, bint deep_copy = True, bint take_ownership = False, bint check_valid = True) -> SBA:
    #     '''
    #     Creates an SBA from a contiguous 1D buffer of uints.  
    #     If deep_copy is True then take_ownership is overridden to True.  
    #     check_valid ensures that the bit order is descending,  
    #             and that the size of items is correct (but not neccessarily the type. Please ensure the buffer's type is uint).  
    #     If deep_copy is False, and take_ownership is True, then ensure that the buffer is writable, and not modified by source.  
    #     '''
    #     if check_valid:
    #         if view.itemsize != sizeof(uint):
    #             raise SBAException("Itemsize is incorrect. Ensure mem view has uint element with length " + str(sizeof(uint)) + " bytes.")
    #         for i in range(len(view) - 1):
    #             if view[i] <= view[i + 1]:
    #                 raise SBAException("Indices must be in descending order")
    #     cdef uint* view_p = &view[0]
    #     cdef SBA ret = SBA.__new__(SBA)
    #     ret._init_len(len(view))
    #     if deep_copy:
    #         ret.owner = 1
    #         memcpy(&ret.indices[0], view_p, len(view) * sizeof(uint))
    #     else:
    #         ret.owner = take_ownership
    #         ret.indices = view_p
    #     return ret

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        ''' Buffer protocol. Read-only '''
        if buffer == NULL:
            raise BufferError("Buffer is NULL")
        if flags & PyBUF_WRITABLE:
            # not sure if cython does this automatically or not... set buffer to null (next line)
            # Required by protocol: https://docs.python.org/3/c-api/buffer.html#c.PyObject_GetBuffer
            # *buffer = NULL
            raise BufferError("Buffer is read-only. Try to_np()")
        
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
            buffer.shape = &self.len.buffer_length
        else:
            buffer.shape = NULL
        
        if flags & PyBUF_STRIDES:
            buffer.strides = &buffer.itemsize
        else:
            buffer.strides = NULL
        
        self.views += 1

    def __releasebuffer__(self, Py_buffer *buffer):
        self.views -= 1
    
    cpdef np.ndarray to_np(self):
        ''' Give ownership of data to returned numpy array, and removes reference to data in self. '''
        if self.views > 0:
            raise BufferError("Buffer is still being viewed.")
        if not self.owner:
            raise BufferError("Lacks ownership.")
        cdef np.npy_intp val = self.len.len
        cdef np.ndarray[dtype=uint, ndim=1] arr = np.PyArray_SimpleNewFromData(1, &val, np.NPY_UINT, self.indices)
        PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
        self.indices = NULL
        self.cap = 0
        self.len.len = 0
        return arr
    
    def __dealloc__(self):
        if self.owner:
            PyMem_Free(self.indices)

# narr = np.arange(10, dtype=np.dtype("I"))
# cdef int [:] narr_view = narr
# test_buffer(narr)