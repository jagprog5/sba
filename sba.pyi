from __future__ import annotations
import numpy
from typing import Iterable, Union, Optional, Callable, final, overload

class SBAException(Exception):
    pass

@final
class SBA:
    '''
    SBAs can be instantiated through factory methods:

    `length()`

    `range()`

    `iterable()`

    `buffer()`

    `encode()`
    '''

    @staticmethod
    def verify_input(enable: bool):
        '''
        This is enabled by default.

        When an SBA is created via `iterable` or `buffer`,
        verify ensures that indices are in ascending order with no duplicates.

        If the `verify` arg in specified in `iterable` or `buffer`,
        then the arg has precedence over the setting specified by this function.
        '''
    
    @staticmethod
    def range(start_inclusive: int, stop_inclusive: int) -> SBA:
        '''
        Initializes an SBA with specified range of bits set to ON.
        ```python
        >>> SBA.range(2, 5)
        [2 3 4 5]
        ```
        '''
    
    @staticmethod
    def length(len: int = 0) -> SBA:
        '''
        ```python
        >>> SBA.length(5)
        [0 1 2 3 4]
        ```
        '''
    
    @staticmethod
    def iterable(obj: Iterable[int],
                filter: Union[None, Callable[[Union[int, float]], bool]] = None, *,
                reverse: bool = False,
                verify: Optional[bool] = None) -> SBA:
        '''
        Initializes an SBA from an iterable. 

        If `filter` is not specified:
            The input iterable is a sparse array, and contains the indices to be used in this SBA.
        ```python
        >>> SBA.iterable([0, 2, 5])
        [0 2 5]
        ```
            `verify` ensures the indices are ascending and without duplicates. If specified, overrides setting in `verify_input()`

        ---

        If `filter` is specified:
            The input iterable is a dense array. This SBA will contain the indices that satisfy the filter.

            `reverse` flips the order of the indices.

        ```python
        >>> SBA.iterable([1, 0, 0, 1, 1], filter = lambda x : x != 0)
        [0 3 4]
        >>> SBA.iterable([1, 0, 0, 1, 1], filter = lambda x : x != 0, reverse = True)
        [0 1 4]
        ``` 
        '''
    
    @staticmethod
    def buffer(readable_buffer: Union[any, bytes, memoryview, bytearray],
            filter: Union[None, Callable[[Union[int, float]], bool]] = None, *,
            copy: bool = True,
            reverse: bool = False,
            verify: Optional[bool] = None) -> SBA:
        '''
        Initializes an SBA from a buffer.

        If `filter` is not specified:
            The input buffer is a sparse array, and contains the indices to be used in this SBA.
        ```python
        >>> SBA.buffer(SBA.from_range(0, 9))
        [0 1 2 3 4 5 6 7 8 9]
        ```
            `copy` gives a separate copy of the data to the SBA. If `copy` is `False`, the SBA keeps a read-only reference to the buffer.

            `verify` ensures the indices are ascending and without duplicates. If specified, overrides setting in `verify_input()`
        
        ---
        If `filter` is specified:
            The input buffer is a dense array. This SBA will contain the indices that satisfy the filter.

            `reverse` flips the order of the indices.

        ```python
        >>> a = memoryview(array('h', [0, 0, 0, 2, 0, -1]))
        >>> SBA.buffer(a, lambda x : x != 0)
        [3 5]
        >>> SBA.buffer(a, lambda x : x != 0, reverse=True)
        [0 2]
        ``` 
        '''
    
    def to_np(self, give_ownership = False) -> numpy.ndarray:
        '''
        Create a numpy array.

        `give_ownership`:  
            True: Makes the returned numpy array the owner of the data, and clears this SBA's reference to the data.  
            False: The returned numpy array gets a read-only buffer to the data.
                While the data is being viewed, this SBA is locked from making changes that may shorten or lengthen the data.
        '''
    
    def print_raw(self) -> None:
        '''
        Prints the underlying allocated memory for the indices, and indicates where the used memory ends.
        ```python
        >>> a = SBA.iterable([0, 2, 5])
        >>> a.print_raw()
            V
        0 2 5
        >>> a.set_bit(0, False)
        >>> a.print_raw()       
          V
        2 5 5
        ```
        '''
    
    def set(self, index: int, state: bool = True) -> SBA:
        ''' Sets the state of a bit, as indicated by the position in the array. Returns self. '''
    
    def __delitem__(self, index):
        ''' Turns off a bit, as indicated by its position in the SBA. '''

    def __setitem__(self, index: int, value: int):
        '''
        Turns OFF a bit, as indicated by `index`: a position in the SBA,  
        then turns ON a bit indicated by `value`: a bit in the underlying array.  
        ```python
        >>> a = SBA.iterable([0, 5, 10, 15])
        >>> a[2] = 1
        >>> a
        [0 1 5 15]
        ```
        '''

    @overload
    def __getitem__(self, index: int) -> int:
        ...

    @overload
    def __getitem__(self, index: slice) -> SBA:
        '''
        if `index` is an `int`:  
            Returns a bit as indicated by a position in the SBA.  
        ```python
        >>> SBA.iterable([1, 2, 3, 4])[-2]
        3
        ```
        ---
        if `index` is a `slice`:  
            Returns the bits in the array within the specified range (inclusive stop, inclusive start).  
            The specified step is ignored (uses 1).  
        ```python
        >>> SBA.iterable([1, 2, 10, 15])[15:2]
        [2 10 15]
        >>> SBA.iterable(range(0, 1000, 2))[100:110]
        [100 102 104 106 108 110]
        '''
    
    def cp(self) -> SBA:
        ''' Creates a deep-copy. '''
    
    def orb(a: SBA, b: SBA) -> SBA:
        ''' OR bits. Returns bits in a OR in b. '''

    def orl(a: SBA, b: SBA) -> int:
        ''' OR length. Returns the number of bits in a OR in b. '''
    
    def orp(query: SBA, arr: numpy.ndarray) -> numpy.ndarray:
        ''' OR parallel. See andp() '''
    
    def xorb(a: SBA, b: SBA) -> SBA:
        ''' XOR bits. Returns bits in a XOR in b. '''

    def xorl(a: SBA, b: SBA) -> int:
        ''' XOR length. Returns the number of bits in a XOR in b. '''
    
    def xorp(query: SBA, arr: numpy.ndarray) -> numpy.ndarray:
        ''' XOR parallel. See andp() '''
    
    def andb(a: SBA, b: SBA) -> SBA:
        ''' AND bits. Returns bits in a AND in b. '''

    def andl(a: SBA, b: SBA) -> int:
        ''' AND length. Returns the number of bits in a AND in b. '''
    
    def andi(a: SBA, b: SBA) -> SBA:
        ''' AND bits inplace, placed the result in this SBA. Returns self. '''
    
    def andp(query: SBA, arr: numpy.ndarray) -> numpy.ndarray:
        '''
        AND parallel. Computes andl on query with each element of arr, and places each result
        in it's respective index of the returned 1D numpy array of c ints.
        ```python
        >>> from sba import *
        >>> import numpy as np
        >>> arr = np.array([SBA.length(i) for i in range(3)], dtype='object')           
        >>> arr
        array([[], [0], [0 1]], dtype=object)
        >>> a = SBA.length(3)
        >>> a
        [0 1 2]
        >>> SBA.andp(a, arr)
        array([0, 1, 2], dtype=int32)
        ```
        '''
    
    def __radd__(self, other: str) -> str:
        ...
    
    @overload
    def __add__(self, other: str) -> str:
        ...
    
    @overload
    def __add__(self, other: SBA) -> SBA:
        ...
    
    @overload
    def __add__(self, other: Iterable[int]) -> SBA:
        ...
    
    @overload
    def __add__(self, other: int) -> SBA:
        '''
        Returns SBA with specified bits set to ON.
        ```python
        >>> SBA.iterable([1, 2]) + 0
        [0, 1, 2]
        ```
        '''
    
    def __or__(self, other):
        ''' Same as __add__ '''
    
    @overload
    def get(self, index: int) -> bool:
        ...
    
    @overload
    def get(self, index1: int, index2: int) -> SBA:
        '''
        if index2 is left blank:  
            Returns the state of a bit as indicated by the position in the array.  
        ```python
            >>> SBA.iterable([2, 3]).get(3)
            True
        ```
        ---
        if index2 is specified:
            returns an SBA of all elements in a section from index1 to index2, inclusively.
        ```python
            >>> SBA.iterable([2, 6, 9, 11]).get(10, 1)
            [2 6 9]
        ```
        '''
    
    def __contains__(self, index: int) -> bool:
        ''' Returns True if the specified index is contained in the SBA. '''

    @overload
    def __mul__(self, other: int) -> bool:
        ...
    
    @overload
    def __mul__(self, other: float) -> SBA:
        ...
    
    @overload
    def __mul__(self, other: SBA) -> SBA:
         '''  
        if `other` is an `int`:
            Returns the state of the bit.  
        ```python
        >>> SBA.iterable([1, 2]) * 50
        False
        >>> SBA.iterable([1, 2]) * 2
        True
        ```
        ---
        if `other` is an `float`:
            Returns a random subsample where each bit has `other` chance of being in the output.  
        ```python
        >>> SBA.lengthgth(6) * (1 / 3)
        [2 5]
        True
        ```
        ---
        if `other` is an `SBA`:
            Returns self AND other.  
        ```python
        >>> SBA.iterable([2, 3]) * SBA.iterable([1, 2])
        [2]
        ```
        '''
    
    def __and__(self, other):
        ''' Same as __mul__ '''
    
    def __mod__(self, other: int) -> SBA:
        ''' Returns a random subsample that contains at most `other` bits. '''
    
    def __sub__(self, other: int) -> SBA:
        '''
        Returns SBA with specified bit set to OFF.
        ```python
        >>> SBA.iterable((1, 2)) - 2
        [1]
        ```
        '''
    
    def rm(self, r: SBA) -> SBA:
        ''' Turns off all bits that are in r. Returns self. '''
    
    def shift(self, n: int) -> SBA:
        ''' Shifts self by n places. A positive n is a increases each index. Returns self. '''
    
    @staticmethod
    def seed_rand():
        ''' Calls c stdlib srand. '''
    
    @staticmethod
    def rand_int() -> int:
        ''' Returns random positive c int. Should have a prior call to seed_rand() '''
    
    @overload
    def subsample(self, amount: int) -> SBA:
        ...
    
    @overload
    def subsample(self, amount: float) -> SBA:
        '''
        Should have a prior call to seed_rand().

        Returns self after op.

        If `amount` is an `int`:
            Randomly turns off bits until the length matches the amount.
            If the length is already less than the amount then this does nothing.
        ---
        If `amount` is a `float`:  
            Each bit has a chance of being turned off, where an amount of
                0.0 returns an empty sba,
                1.0 almost always returns a copy of this sba.
        
        ''' 
    
    @staticmethod
    def encode(input: float, num_on_bits: int, size: int, period: Optional[float]) -> SBA:
        '''
        Encodes input as a Sparse Distributed Representation.

        `num_on_bits` is the length of the SBA.

        `size` is the length of the underlying array being represented.
        
        If `period` is not specified:  
            `input` should be from 0 to 1, inclusively.
            An `input` of 0 turns on the least significant bits.  

            An `input` of 1 turns on the most significant bits.
        ```python
        >>> SBA.encode(0.5, 3, 100)
        [49 50 51]
        ```
        ---
        If `period` is specified:
            Encodes input such that it wraps back to 0 as it approaches a multiple of the period.
        ```python
        >>> SBA.encode(1, 3, 100, period=10) == SBA.encode(11, 3, 100, period=10)
        True
        ```
        '''
