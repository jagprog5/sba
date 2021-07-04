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
    cdef inline void c_verify_input(bint enable) 
    cdef inline int raise_if_viewing(self) except -1
    cdef inline void range(self, int start_inclusive, int stop_inclusive) 
    cdef inline void lengthen_if_needed(self, int length_override = -1) nogil
    cdef inline void shorten(self)
    cdef inline void shorten_if_needed(self)
    @staticmethod
    cdef SBA c_from_range(int start_inclusive, int stop_inclusive)
    @staticmethod
    cdef SBA c_from_capacity(int cap, bint default)
    cpdef np.ndarray to_buffer(self, bint give_ownership = *)
    cpdef print_raw(self)
    cdef void turn_on(self, int index)
    cdef void turn_off(self, int index)
    cdef inline int check_index(self, int index) except -1
    cdef SBA get_section(self, int start_inclusive, int stop_inclusive)
    cpdef SBA cp(self)
    cdef inline void _get_one(SBA a, int* offset, int* value, bint* nempty) nogil
    cdef inline void _get_both(SBA a, int* a_offset, int* a_val, bint* a_empty, SBA b, int* b_offset, int* b_val, bint* b_empty) nogil
    cdef inline void _add_to_output(SBA r, int* r_len, int val, bint len_only) nogil
    @staticmethod
    cdef void orc(void* r, SBA a, SBA b, bint exclusive, bint len_only) nogil
    @staticmethod
    cdef void andc(void* r, SBA a, SBA b, bint len_only) nogil
    cdef bint get_bit(self, int index)
    cpdef void rm(self, SBA rm)
    cpdef void shift(self, int n)
    @staticmethod
    cdef seedRand()
    cpdef bint compare(self, SBA other, int op)
    @staticmethod
    cdef int _qsort_compare(const void* a, const void* b) nogil
    cdef void subsample_length(self, int amount)
    cdef int subsample_portion(self, float amount) except -1
    @staticmethod
    cdef SBA encode_linear(float input, int num_on_bits, int length)
    @staticmethod
    cdef SBA encode_periodic(float input, float period, int num_on_bits, int length)