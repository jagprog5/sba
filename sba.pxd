cimport numpy as np

cdef fused const_numeric:
    const short
    const unsigned short
    const int
    const unsigned int
    const long
    const unsigned long
    const float
    const double

cdef union SBALen:
    Py_ssize_t ssize_t_len # for use in buffer protocol
    int len # for normal use in SBA

cdef class SBA:
    cdef int views # number of references to buffer
    cdef int cap # capacity, mem allocated for the indices
    cdef SBALen len # length, the number of ON bits in the array
    cdef int* indices # contains indices of bits that are ON.
    
    @staticmethod
    cdef verifyInput(bint enable)
    cdef inline int raiseIfViewing(self) except -1
    cdef inline void range(self, int start_inclusive, int stop_inclusive)
    cdef inline lengthenIfNeeded(self, int length_override = -1)
    cdef inline shorten(self)
    cdef inline shortenIfNeeded(self)
    @staticmethod
    cdef SBA fromRange(int start_inclusive, int stop_inclusive)
    @staticmethod
    cdef SBA fromCapacity(int cap, bint default)
    cdef np.ndarray toBuffer(self, bint give_ownership)
    cdef printRaw(self)
    cdef turnOn(self, int index)
    cdef turnOff(self, int index)
    cpdef set(self, int index, bint state)
    cdef inline int checkIndex(self, int index) except -1
    cdef SBA getSection(self, int start_inclusive, int stop_inclusive)
    cpdef SBA cp(self)
    cdef inline void _get_one(SBA a, int* offset, int* value, bint* nempty)
    cdef inline void _get_both(SBA a, int* a_offset, int* a_val, bint* a_empty, SBA b, int* b_offset, int* b_val, bint* b_empty)
    cdef inline void _add_to_output(SBA r, int* r_len, int val, bint len_only)
    @staticmethod
    cdef orc(void* r, SBA a, SBA b, bint exclusive, bint len_only)
    @staticmethod
    cdef andc(void* r, SBA a, SBA b, bint len_only)
    cdef bint getBit(self, int index)
    cpdef rm(self, SBA rm)
    cpdef shift(self, int n)
    @staticmethod
    cdef seedRand()
    cpdef bint compare(self, SBA other, int op)
    @staticmethod
    cdef int _qsort_compare(const void* a, const void* b) nogil
    cdef subsampleLength(self, int amount)
    cdef int subsamplePortion(self, float amount) except -1
    @staticmethod
    cdef SBA encodeLinear(float input, int num_on_bits, int length)
    @staticmethod
    cdef SBA encodePeriodic(float input, float period, int num_on_bits, int length)