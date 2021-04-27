''' This stub is included for autocomplete and docstring stuff. '''

import numpy
from typing import Iterable, Union, overload

class SBAException(Exception):
    pass

class SBA:
    '''
    Instantiation through __init__ does a deep copy.  
    For zero-copy, see the factory methods.
    ```python
    >>> SBA() # empty
    []
    >>> SBA(5) # initial capacity of 5
    [4 3 2 1 0]
    >>> SBA(5, 2) # 5 downto 2
    [5 4 3 2]
    >>> SBA(numpy.array([3, 2, 0])) # np
    [3 2 0]
    >>> SBA(SBA([5, 2, 0])) # buffer
    [5, 2, 0]
    >>> SBA([5, 2, 0]) # iterable
    [5, 2, 0]
    ```
    '''

    def enable_checking() -> None:
        '''
        This is enabled by default.  
        On the creation of an SBA, ensure that indices are valid.  
        The checks include:
          - Each indice is of type int.
          - Each indice is in c int range.
          - Indices are in descending order, with no duplicates.
        Note that the 'check_valid' arg in factory methods can disable checking for individual calls,
        even if checking is enabled globally.  
        '''
    
    def disable_checking() -> None:
        '''
        Disables the check that ensures indices are valid on SBA creation.  
        This overrides the 'check_valid' param used in factory methods.  
        '''
    
    def from_iterable(obj: Iterable[int], check_valid: bool = True) -> SBA:
        '''
        Deep-copy iterable to init an SBA.  
        check_valid: check that all elements are valid (integers, in int range, descending order, no duplicates). 
        ```python
        >>> SBA.from_iterable([5, 2, 0])
        [5 2 0]
        ``` 
        '''
    
    def from_range(stop_inclusive: int, start_inclusive: int) -> SBA:
        '''
        Initializes an SBA with specified range of bits set to ON.
        ```python
        >>> SBA.from_range(3, -2)
        [3 2 1 0 -1 -2]
        ```
        '''
    
    def from_capacity(initial_capacity: int = 0) -> SBA:
        '''
        Initializes an SBA with specified initial capacity.  
        ```python
        >>> SBA.from_capacity(5)
        [4 3 2 1 0]
        ```
        '''
    
    def from_dense(buffer, reverse = False, filter = lambda x: x != 0):
        '''
        Converts an array to an SBA.  
        buffer is a memoryview to a contiguous list of type short, int, long, float, or double.  
        filter should give True for elements that should should be placed in the SBA.  
        ```python
        >>> from array import array
        >>> a = memoryview(array('h', [0, 0, 0, 2, 0, -1]))
        >>> SBA.from_dense(a)
        [2 0]
        >>> SBA.from_dense(a, reverse=True)
        [5 3]
        ```
        '''
    
    def from_np(np_arr, deep_copy = True, check_valid = True) -> SBA:
        '''
        Creates and initializes an SBA from a numpy array.  
        deep_copy:  
            true: The sba gets a separate copy of the data.  
            false: The sba gets a read-only reference to the data.  
        check_valid: check that all elements are valid (descending order, no duplicates).
        ```python
        >>> SBA.from_np(np.array([5, 2, 0], np.intc)) # intc needed to work cross-platform
        [5 2 0]
        >>> a = SBA.from_np(np.array([5, 2, 0]), deep_copy=False) 
        >>> a[0] = 3 # raises exception since it's read-only.
        ``` 
        '''
    
    def to_np(self, give_ownership = True) -> numpy.ndarray:
        '''
        Create a numpy array.  
        give_ownership:  
            true: Makes the returned numpy array the owner of the data, and clears this SBA's reference to the data.  
            false: The returned numpy array gets a read-only buffer to the data.
        '''
    
    def print_raw(self):
        '''
        Prints the underlying allocated memory for the indices, and indicates where the used memory ends.
        ```python
        >>> a = SBA([5, 2, 0])
        >>> a.print_raw()
            V
        5 2 0
        >>> a.set_bit(5, False)
        >>> a.print_raw()       
          V
        2 0 0
        ```
        '''
    
    def set(self, index: int, state: bool = True):
        ''' Sets the state of a bit, as indicated by the position in the array. '''
    
    def __delitem__(self, index):
        ''' Turns off a bit, as indicated by its position in the SBA. '''

    def __setitem__(self, index: int, value: int):
        '''
        Turns OFF a bit, as indicated by index: a position in the SBA,  
        then turns ON a bit indicated by value: a bit in the array.  
        Not to be confused with set_bit.
        ```python
        >>> a = SBA([15, 10, 5, 0])
        >>> a[2] = 6
        >>> a
        [15 10 6 0]
        ```
        '''

    @overload
    def __getitem__(self, index: int) -> int:
        '''
        Returns a bit as indicated by a position in the SBA.  
        Not to be confused with get_bit.
        ```python
        >>> SBA([4, 3, 2, 1])[-2]
        2
        ```
        '''
    
    @overload
    def __getitem__(self, range: slice) -> SBA:
        '''
        Returns the bits in the array within the specified range (inclusive stop, inclusive start).  
        The specified step is ignored (uses 1).  
        ```python
        >>> SBA([15, 10, 2, 1])[15:2]
        [15 10 2]
        >>> SBA(range(0, 10000, 2))[100:110]
        [110 108 106 104 102 100]
        '''
    
    def cp(self) -> SBA:
        ''' Creates a deep-copy. '''
    
    def orb(a: SBA, b: SBA) -> SBA:
        ''' OR bits. Returns bits in a OR in b. '''

    def orl(a: SBA, b: SBA) -> int:
        ''' OR length. Returns the number of bits in a OR in b. '''
    
    def xorb(a: SBA, b: SBA) -> SBA:
        ''' XOR bits. Returns bits in a XOR in b. '''

    def xorl(a: SBA, b: SBA) -> int:
        ''' XOR length. Returns the number of bits in a XOR in b. '''
    
    def andb(a: SBA, b: SBA) -> SBA:
        ''' AND bits. Returns bits in a AND in b. '''

    def andl(a: SBA, b: SBA) -> int:
        ''' AND length. Returns the number of bits in a AND in b. '''
    
    def andi(self, a: SBA) -> SBA:
        ''' AND bits inplace, placed the result in this SBA. '''
    
    @overload
    def __add__(self, other: str) -> str:
        '''
        Returns string concatenated with self.  
        ```python
        >>> SBA([2, 1]) + " hi"
        '[2 1] hi'
        ```
        '''
    
    @overload
    def __add__(self, other: int) -> SBA:
        '''
        Returns SBA with specified bit set to ON.
        ```python
        >>> SBA([2, 1]) + 0
        [2 1 0]
        ```
        '''
    
    @overload
    def __add__(self, other: Union[Iterable[int], SBA]) -> SBA:
        '''
        Returns a copy of self OR other.
        ```python
        >>> SBA([3, 2]) + SBA([2, 1])
        [3 2 1]
        >>> SBA([2, 1]) + [5, 0, 2]
        [5 2 1 0]
        ```
        '''
    
    def __or__(self, other):
        ''' See __add__ '''
    
    def get(self, index1: int, index2: int = None) -> Union[bool, SBA]:
        '''
        if index2 is left blank:  
            Returns the state of a bit as indicated by the position in the array.  
        ```python
            >>> SBA([3, 2]).get(3)
            True
        ```
        if index2 is specified:
            returns an SBA of all elements in a section from index1 downto index2, inclusively.
        ```python
            >>> SBA([11, 9, 6, 2]).get(10, 1)
            [9 6 2]
        ```
        '''
    
    @overload
    def __mul__(self, other: int) -> bool:
        '''
        Returns the state of the bit.  
        ```python
        >>> SBA([2, 1]) * 50
        False
        >>> SBA([2, 1]) * 2
        True
        ```
        '''
    
    @overload
    def __mul__(self, other: SBA) -> SBA:
        '''
        Returns a copy of self AND other.  
        ```python
        >>> SBA([3, 2]) * SBA([2, 1])
        [2]
        ```
        '''
    
    @overload
    def __mul__(self, other: float) -> SBA:
        '''
        Returns a random subsample.  
        ```python
        >>> SBA([5, 4, 3, 2, 1, 0]) * (1 / 3)
        [5, 2]
        ```
        '''
    
    def __and__(self, other):
        ''' See __mul__ '''
    
    @overload
    def __sub__(self, other: int) -> SBA:
        '''
        Returns SBA with specified bit set to OFF.
        ```python
        >>> SBA((2, 1)) - 2
        [1]
        ```
        '''
    
    @overload
    def __sub__(self, other: Union[SBA, Iterable[int]]) -> SBA:
        '''
        Returns a copy of self with all elements in other removed.
        ```python
        >>> SBA((3, 2, 1)) - SBA((3, 2))
        [1]
        ```
        '''
    
    def rm(self, r: SBA) -> SBA:
        ''' Turns off all bits that are in r. '''
    
    def shift(self, n: int) -> SBA:
        ''' Shifts self by n places. A positive n is a left shift. '''
    
    def seed_rand():
        ''' Seeds c srand, for use in rand_int and subsample. '''
    
    def rand_int() -> int:
        ''' Returns random positive c int. See seed_rand. '''
    
    def subsample(self, amount: float) -> SBA:
        ''' Returns a random subsample. See seed_rand. ''' 
    
    def encode_linear(input: float, num_on_bits: int, size: int) -> SBA:
        '''
        Encodes input, a float from 0 to 1, inclusively, with  
        num_on_bits being the length of the SBA, and  
        size being the length of the underlying array being represented.  
        '''

    def encode_periodic(input: float, period: float, num_on_bits: int, size: int) -> SBA:
        '''
        Encodes input linearly such that the same difference from a multiple
        of period will give the same encoding, with
        num_on_bits being the length of the SBA, and  
        size being the length of the underlying array being represented.  
        '''
