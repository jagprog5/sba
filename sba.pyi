from __future__ import annotations
import numpy
from typing import Iterable, Union, Optional, Callable, overload

class SBAException(Exception):
    pass

class SBA:
    '''
    SBAs can be instantiated through factory methods:

    `from_capacity()`

    `from_range()`

    `from_iterable()`

    `from_buffer()`
    '''

    def verify_input(enable: bool):
        '''
        This is enabled by default.

        When an SBA is created via `from_iterable` or `from_buffer`,
        verify ensures that indices are in ascending order with no duplicates.

        If the `verify` arg in specified in `from_iterable` or `from_buffer`,
        then the arg has precedence over the setting specified by this function.
        '''
    
    def from_range(start_inclusive: int, stop_inclusive: int) -> SBA:
        '''
        Initializes an SBA with specified range of bits set to ON.
        ```python
        >>> SBA.from_range(2, 5)
        [2 3 4 5]
        ```
        '''
    
    def from_capacity(capacity: int = 0) -> SBA:
        '''
        ```python
        >>> SBA.from_capacity(5)
        [0 1 2 3 4]
        ```
        '''
    
    def from_iterable(obj: Iterable[int],
                filter: Union[None, Callable[[Union[int, float]], bool]] = None, *,
                verify: Optional[bool] = None,
                reverse: bool = False) -> SBA:
        '''
        Initializes an SBA from an iterable. 

        If `filter` is not specified:
            The input iterable is a sparse array, and contains the indices to be used in this SBA.
        ```python
        >>> SBA.from_iterable([0, 2, 5])
        [0 2 5]
        ```
            `verify` ensures the indices are ascending and without duplicates. If specified, overrides setting in `verify_input()`

        ==========================================

        If `filter` is specified:
            The input iterable is a dense array. This SBA will contain the indices that satisfy the filter.

            `reverse` flips the order of the indices.

        ```python
        >>> SBA.from_iterable([1, 0, 0, 1, 1], filter = lambda x : x != 0)
        [0 3 4]
        >>> SBA.from_iterable([1, 0, 0, 1, 1], filter = lambda x : x != 0, reverse = True)
        [0 1 4]
        ``` 
        '''
    
    def from_buffer(buffer,
            filter: Union[None, Callable[[Union[int, float]], bool]] = None, *,
            copy: bool = True,
            verify: Optional[bool] = None,
            reverse: bool = False) -> SBA:
        '''
        Initializes an SBA from a buffer.

        If `filter` is not specified:
            The input buffer is a sparse array, and contains the indices to be used in this SBA.
        ```python
        >>> SBA.from_buffer(SBA.from_range(0, 9))
        [0 1 2 3 4 5 6 7 8 9]
        ```
            `copy` gives a separate copy of the data to the SBA. If `copy` is `False`, the SBA keeps a read-only reference to the buffer.

            `verify` ensures the indices are ascending and without duplicates. If specified, overrides setting in `verify_input()`

        ==========================================

        If `filter` is specified:
            The input buffer is a dense array. This SBA will contain the indices that satisfy the filter.

            `reverse` flips the order of the indices.

        ```python
        >>> a = memoryview(array('h', [0, 0, 0, 2, 0, -1]))
        >>> SBA.from_buffer(a, lambda x : x != 0)
        [3 5]
        >>> SBA.from_buffer(a, lambda x : x != 0, reverse=True)
        [0 2]
        ``` 
        '''
    
    def to_buffer(self, give_ownership = True) -> numpy.ndarray:
        '''
        Create a numpy array.  
        `give_ownership`:  
            True: Makes the returned numpy array the owner of the data, and clears this SBA's reference to the data.  
            False: The returned numpy array gets a read-only buffer to the data.
                While the data is being used, it is locked out from being changed by this SBA.
        '''
    
    def print_raw(self):
        '''
        Prints the underlying allocated memory for the indices, and indicates where the used memory ends.
        ```python
        >>> a = SBA.from_iterable([0, 2, 5])
        >>> a.print_raw()
            V
        0 2 5
        >>> a.set_bit(0, False)
        >>> a.print_raw()       
          V
        2 5 5
        ```
        '''
    
    def set(self, index: int, state: bool = True):
        ''' Sets the state of a bit, as indicated by the position in the array. '''
    
    def __delitem__(self, index):
        ''' Turns off a bit, as indicated by its position in the SBA. '''

    def __setitem__(self, index: int, value: int):
        '''
        Turns OFF a bit, as indicated by `index`: a position in the SBA,  
        then turns ON a bit indicated by `value`: a bit in the array.  
        ```python
        >>> a = SBA.from_iterable([0, 5, 10, 15])
        >>> a[2] = 1
        >>> a
        [0 1 5 15]
        ```
        '''

    @overload
    def __getitem__(self, index: int) -> int:
        '''
        if `index` is an `int`:  
            Returns a bit as indicated by a position in the SBA.  
        ```python
        >>> SBA.from_iterable([1, 2, 3, 4])[-2]
        3
        ```

        if `index` is a `slice`:  
            Returns the bits in the array within the specified range (inclusive stop, inclusive start).  
            The specified step is ignored (uses 1).  
        ```python
        >>> SBA.from_iterable([1, 2, 10, 15])[15:2]
        [2 10 15]
        >>> SBA.from_iterable(range(0, 1000, 2))[100:110]
        [100 102 104 106 108 110]
        '''
    
    @overload
    def __getitem__(self, range: slice) -> SBA: ...
    
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
    def __add__(self, other: int) -> SBA:
        '''
        if `other` is an `int`:
            Returns SBA with specified bit set to ON.
        ```python
        >>> SBA.from_iterable([1, 2]) + 0
        [0, 1, 2]
        ```

        if `other` is an `Iterable`:
            Returns self OR other.
        ```python
        >>> SBA.from_iterable([2, 3]) + SBA.from_iterable([1, 2])
        [1 2 3]
        >>> SBA([1, 2]) + [5, 0, 2]
        [0 1 2 5]
        ```
        '''
    
    @overload
    def __add__(self, other: str) -> str: ...
    
    @overload
    def __add__(self, other: Union[Iterable[int], SBA]) -> SBA: ...
    
    def __or__(self, other):
        ''' See __add__ '''
    
    def get(self, index1: int, index2: int = None) -> Union[bool, SBA]:
        '''
        if index2 is left blank:  
            Returns the state of a bit as indicated by the position in the array.  
        ```python
            >>> SBA.from_iterable([2, 3]).get(3)
            True
        ```
        if index2 is specified:
            returns an SBA of all elements in a section from index1 to index2, inclusively.
        ```python
            >>> SBA.from_iterable([2, 6, 9, 11]).get(10, 1)
            [2 6 9]
        ```
        '''
    
    def __contains__(self, index: int) -> bool:
        ''' Returns True if the specified index is contained in the SBA. '''

    recursive_numeric = Iterable[Union[int, float, recursive_numeric]]

    @overload
    def __mul__(self, other: recursive_numeric) -> recursive_numeric:
        '''
        if `other` is an `int`:
            Returns the state of the bit.  
        ```python
        >>> SBA.from_iterable([1, 2]) * 50
        False
        >>> SBA.from_iterable([1, 2]) * 2
        True
        ```

        if `other` is an `SBA`:
            Returns self AND other.  
        ```python
        >>> SBA.from_iterable([2, 3]) * SBA.from_iterable([1, 2])
        [2]
        ```

        if `other` is a `float`:
            Returns a random subsample, each bit has a chance of turning off.  
        ```python
        >>> SBA.from_iterable([0, 1, 2, 3, 4, 5]) * (1 / 3)
        [2 5]
        ```

        if `other` is an `Iterable`:
            Returns the result of mul on each element.  
        ```python
        >>> SBA.from_iterable([2, 3]) * [0, 3, [3, 2]]
        [False, True, [True, True]]
        ```
        '''
    
    def __and__(self, other):
        ''' See __mul__ '''
    
    @overload
    def __sub__(self, other: int) -> SBA:
        '''
        if `other` is an `int`:
            Returns SBA with specified bit set to OFF.
        ```python
        >>> SBA.from_iterable((1, 2)) - 2
        [1]
        ```

        if `other` is an `Iterable`:
            Returns cp with all elements in other removed.
        ```python
        >>> SBA.from_iterable((1, 2, 3)) - SBA.from_iterable((2, 3))
        [1]
        ```
        '''
    
    @overload
    def __sub__(self, other: Union[SBA, Iterable[int]]) -> SBA: ...
    
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
    
    def encode(input: float, num_on_bits: int, size: int, period: Optional[float]) -> SBA:
        '''
        Encodes input as an SBA.

        `num_on_bits` is the length of the SBA.  
        `size` is the length of the underlying array being represented.  
        if `period` is not specified:  
            input should be from 0 to 1, inclusively.  
            An input of 0 turns on the least significant bits,  
            an input of 1 turns on the most significant bits, and  
            the output is scaled in-between.  
        ```python
        >>> SBA.encode(0.5, 3, 100)
        [49 50 51]
        ```
        if `period` is specified:
            Encodes input such that it wraps back to 0 as it approaches a multiple of the period.
        ```python
        >>> SBA.encode(1, 3, 100, period=10) == SBA.encode(11, 3, 100, period=10)
        True
        ```
        '''
