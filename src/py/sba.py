from __future__ import annotations
import ctypes as c
from typing import List, Iterable, Union
import pathlib

try:
    import numpy
except:
    # I don't want to have it as a dependency
    pass # No need to do conversions if numpy isn't installed

class SBAException(Exception):
    pass

class SBAIterator:
    def __init__(self, sba: SBA):
        self._sba = sba
        self._index = 0
    
    def __next__(self):
        if self._index >= self._sba.size:
            raise StopIteration
        ret = self._sba.indices[self._index]
        self._index += 1
        return ret
        
class SBA(c.Structure):
    '''
    Sparse bit array. This is a wrapper for sba.h
    '''

    do_checking = True
    sba_lib = None

    _fields_ = [
        ('size', c.c_uint32),
        ('capacity', c.c_uint32),
        ('indices', c.POINTER(c.c_uint32))]
    
    def _init_lib_if_needed():
        if SBA.sba_lib is None:
            lib_folder = pathlib.Path(__file__).absolute().parents[0] / "c-build"
            g = lib_folder.glob("sba_lib.*")
            try:
                sba_lib_file = g.__next__()
            except StopIteration as e:
                print(e)
                raise SBAException("Couldn't load sba_lib in: " + str(lib_folder))
                    
            SBA.sba_lib = c.CDLL(str(sba_lib_file)) # str needed for windows path

            # printSBA
            SBA.printSBA = SBA.sba_lib.printSBA
            SBA.printSBA.restype = None
            SBA.printSBA.argtype = [c.POINTER(SBA)]

            # turnOn, turnOff
            # both require realloc, manually implemented

            # getBit
            SBA.getBit = SBA.sba_lib.getBit
            SBA.getBit.restype = c.c_uint8
            SBA.getBit.argtype = [c.POINTER(SBA), c.c_uint32]

            # getSection
            SBA.getSection = SBA.sba_lib.getSection
            SBA.getSection.restype = None
            SBA.getSection.argtype = [c.POINTER(SBA)] * 2 + [c.c_uint32] * 3

            # turnOffAll
            SBA.turnOffAll = SBA.sba_lib.turnOffAll
            SBA.turnOffAll.restype = None
            SBA.turnOffAll.argtype = [c.POINTER(SBA)] * 2

            # andBits
            SBA.andBits = SBA.sba_lib.andBits
            SBA.andBits.restype = None
            SBA.andBits.argtype = [c.c_void_p] + [c.POINTER(SBA)] * 2 + [c.c_uint8]

            # orBits
            SBA.orBits = SBA.sba_lib.orBits
            SBA.orBits.restype = None
            SBA.orBits.argtype = [c.c_void_p] + [c.POINTER(SBA)] * 2 + [c.c_uint8] * 2

            # rshift
            SBA.rshift = SBA.sba_lib.rshift
            SBA.rshift.restype = None
            SBA.rshift.argtype = [c.POINTER(SBA), c.c_uint32]

            # lshift
            SBA.lshift = SBA.sba_lib.lshift
            SBA.lshift.restype = None
            SBA.lshift.argtype = [c.POINTER(SBA), c.c_uint32]

            # equality
            SBA.equal = SBA.sba_lib.equal
            SBA.equal.restype = c.c_uint8
            SBA.equal.argtype = [c.POINTER(SBA)] * 2

            # cp
            SBA.f_cp = SBA.sba_lib.cp
            SBA.f_cp.restype = None
            SBA.f_cp.argtype = [c.POINTER(SBA)] * 2

            # subsample
            SBA.sba_lib.seed_rand()
            SBA.f_subsample = SBA.sba_lib.subsample
            SBA.f_subsample.restype = None
            SBA.f_subsample.argtype = [c.POINTER(SBA), c.c_float]

            # encodeLinear
            SBA.encodeLinear = SBA.sba_lib.encodeLinear
            SBA.encodeLinear.restype = None
            SBA.encodeLinear.argtype = [c.c_float, c.c_uint32, c.POINTER(SBA)]

            # encodePeriodic
            SBA.encodePeriodic = SBA.sba_lib.encodePeriodic
            SBA.encodePeriodic.restype = None
            SBA.encodePeriodic.argtype = [c.c_float, c.c_float, c.c_uint32, c.POINTER(SBA)]
    
    def _throw_if_indices_not_valid(l, ln):
        if any(not isinstance(l[i], int) for i in range(ln)):
            raise SBAException("indices must only contain ints.")
        if any(l[i] >= l[i+1] for i in range(ln-1)):
            raise SBAException("indices must be in ascending order, with no duplicates.")
        if any(l[i] < 0 for i in range(ln)):
            raise SBAException("indices must only contain non-negative integers.")
        if any(l[i] > 0xFFFFFFFF for i in range(ln)):
            raise SBAException("indices must not exceed uint32 max.")
    
    def __init__(self, *on_bits: Union[int, Iterable[int], SBA], **_special):
        """
        Initalize an SBA. 
        ```python
        a = SBA(0, 2, 500)   # turn bits 0, 2, and 500 to ON.  
        b = SBA(a)  
        a = SBA([0, 2, 500])  

        ```
        On initialization of an SBA, the indices are checked for validity.  
        Valid indices are:  
          - In ascending order, with no duplicates
          - Unsigned 32 bit integers
        This can be changed with:  

        ```python
        SBA.disable_checking()  
        SBA.enable_checking()   # default behaviour

        ```
        """
        SBA._init_lib_if_needed()
        if len(_special) == 0:
            if len(on_bits) > 0 and hasattr(on_bits[0], "__getitem__"): # support list, tuple, etc as first arg
                on_bits = on_bits[0]
            ln = len(on_bits)
            if SBA.do_checking:
                SBA._throw_if_indices_not_valid(on_bits, ln)
            self.size = (c.c_uint32)(ln)
            self.capacity = self.size
            self.indices = (c.c_uint32 * ln)(*on_bits)
        elif 'uninit' in _special:
            return
        elif 'blank_cap' in _special:
            ln = _special['blank_cap']
            self.size = (c.c_uint32)(ln)
            self.capacity = self.size
            self.indices = (c.c_uint32 * ln)()
        else:
            raise SBAException("Unknown kws: " + str(_special))
            
    
    def enable_checking():
        '''This is enabled by default. On creation of an SBA, ensures that indices are valid.'''
        SBA.do_checking = True
    
    def disable_checking():
        ''' Disables check that ensure that indices are valid on SBA creation. Invalid indices may lead to segfaults.'''
        SBA.do_checking = False
    
    def to_list(self) -> List[int]:
        return [self.indices[i] for i in range(self.size)]
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return str(self.to_list())
    
    def __bool__(self):
        ''' Returns False if empty. '''
        return self.size > 0

    def __iter__(self):
        '''
        Gives indices of ON bits.  
        ```python
        >>> a = SBA(1, 2, 10)
        >>> for i in a:
        >>>     print(i)
        1
        2
        10
        ```
        '''
        return SBAIterator(self)
    
    def __len__(self):
        ''' Returns the number of bits that are ON. '''
        return self.size
    
    def __add__(self, other):
        '''
        For a string: concatenates string of self.
        ```python
        >>> SBA(1, 2) + " hi"
        '[1, 2] hi'
        ```
        For an int: Turns on the bit.
        ```python
        >>> SBA(1, 2) + 0
        [0, 1, 2]
        ```
        For an SBA: return a copy of self OR other
        ```python
        >>> SBA(1, 2) + SBA(2, 3)
        [1, 2, 3]
        ```
        For a iterable of ints: adds each element, separately
        >>> SBA(1, 2) + [0, 2, 5]
        [0, 1, 2, 5]
        '''
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
    
    def __radd__(self, other):
        if isinstance(other, str):
            return other + str(self)
        else:
            raise TypeError(str(type(other)) + " not supported for reverse + op.")
    
    def __and__(self, other):
        return self.__mul__(other)
    
    def __mul__(self, other):
        '''
        For an int: Returns the state of the bit.
        ```python
        >>> SBA(1, 2) * 50
        False
        >>> SBA(1, 2) * 2
        True
        ```
        For an SBA: ANDs the bits
        ```python
        >>> SBA(1, 2) * SBA(2, 3)
        [2]
        ```
        For a float: returns a random subsample
        ```python
        >>> SBA(1, 2, 3, 4, 5, 6) * (1 / 3)
        [2, 5]
        ```
        '''
        if isinstance(other, SBA):
            return SBA.and_bits(self, other)
        elif isinstance(other, int):
            return self.get_bit(other)
        elif isinstance(other, float):
            cp = self.cp()
            cp.subsample(other)
            return cp
        else:
            raise TypeError(str(type(other)) + " not supported for * or & ops.")
    
    def __rmul__(self, other):
        if isinstance(other, float):
            cp = self.cp()
            cp.subsample(other)
            return cp
        else:
            raise TypeError(str(type(other)) + " not supported for reverse * op.")
    
    def __or__(self, other):
        return self.__add__(other)
    
    def __xor__(self, other):
        if isinstance(other, SBA):
            return SBA.xor_bits(self, other)
        else:
            raise TypeError(str(type(other)) + " not supported for ^ op.")
    
    def __sub__(self, other):
        '''
        For an int: Turns off the bit.
        ```python
        >>> SBA(1, 2) - 2
        [1]
        ```
        For an SBA: removes all elements
        ```python
        >>> SBA(1, 2, 3) - SBA(2, 3)
        [1]
        ```
        ```python
        For a iterable of ints: removes each element, separately
        >>> SBA(1, 2) - [2, 0, 5]
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
    
    def __lshift__(self, n):
        return self.__rshift__(-n)

    def __rshift__(self, n):
        '''
        >>> SBA(0, 2, 3) >> 2
        [0, 1]
        '''
        if isinstance(n, int):
            cp = self.cp()
            cp.shift(n)
            return cp
        else:
            raise TypeError(str(type(other)) + " not supported for >> or << ops.")
    
    def _check_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise SBAException("Index must be an int.")
        if index >= self.size:
            raise SBAException("Index out of bounds.")
        if index < 0:
            index = self.size + index
            if index < 0:
                raise SBAException("Index out of bounds.")
        return index

    def __getitem__(self, index: Union[int, slice]) -> int:
        '''
        Note that __getitem__ complete two totally different operation with a slice versus with an int.  

        =-======For an int=========  
        Returns the index of the i-th ON bit.  
        Not to be confused with get_bit.
        ```python
        >>> SBA(1, 2, 3, 4)[-2]
        3
        ```
        =-======For a slice========  
        Returns the ON bits within the specified range (inclusive start, exclusive stop)  
        The specified step is ignored (uses 1)
        ```python
        >>> SBA(1, 2, 10, 15)[2:15]
        [2, 10]
        >>> SBA(range(0, 10000, 2))[100:110]
        [100, 102, 104, 106, 108]
        ```
        '''
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = self.indices[self.size - 1] + 1 if index.stop is None else index.stop
            if start > stop:
                tmp = start
                start = stop
                stop = tmp
            if start < 0:
                start = 0
            r = SBA(blank_cap=stop-start)
            SBA.getSection(c.byref(r), c.byref(self), c.c_uint32(start), c.c_uint32(stop))
            return r
        else:
            return self.indices[self._check_index(index)]
    
    def get_bit(self, index: int) -> bool:
        '''
        Returns the value of the i-th bit.
        ```python
        >>> a = SBA(0, 5)
        >>> a.get_bit(0)
        True
        >>> a.get_bit(1)
        False
        ```
        '''
        if index < 0:
            raise SBAException("Requires non-negative index.")
        return bool(SBA.getBit(c.byref(self), c.c_uint32(index)))
    
    def __setitem__(self, index: int, value: int):
        '''
        Turns off the index-th ON bit, and turns on the value-th bit.  
        Not to be confused with set_bit
        ```python
        >>> a = SBA(0, 5, 10, 15)
        >>> a[1] = 6
        >>> a
        [0, 6, 10, 15]
        ```
        '''
        self.__delitem__(index)
        self.set_bit(value, True)
    
    def set_bit(self, index: int, value: bool):
        '''
        Sets the value of the i-th bit
        ```python
        >>> a = SBA(0, 5, 10, 15)
        >>> a.set_bit(5, False)
        >>> a
        [0, 10, 15]
        >>> a.set_bit(15, True)
        [0, 10, 15]
        ```
        '''
        if index < 0:
            raise SBAException("Requires non-negative index.")
        left = 0
        right = self.size - 1
        middle = 0
        mid_val = 0xFFFFFFFF # u32 max
        while left <= right:
            middle = (right + left) // 2
            mid_val = self.indices[middle]
            if mid_val < index:
                left = middle + 1
            elif mid_val > index:
                right = middle - 1
            else:
                if not value:
                    self.__delitem__(middle)
                return # skip duplicate
        if not value:
            return
        if index > mid_val:
            middle += 1
        self._lengthen_if_needed()
        ptr = c.cast(self.indices, c.c_void_p)
        addr = ptr.value
        type_size = c.sizeof(c.c_uint32)
        from_addr = addr + middle * type_size
        to_addr = from_addr + type_size
        c.memmove(to_addr, from_addr, type_size * (self.size - middle))
        self.size += 1
        self.indices[middle] = index
        return
        
    def __delitem__(self, index: int):
        ''' Turns the i-th ON bit to OFF '''
        index = self._check_index(index)
        self.size -= 1
        ptr = c.cast(self.indices, c.c_void_p)
        addr = ptr.value
        type_size = c.sizeof(c.c_uint32)
        to_addr = addr + index * type_size
        from_addr = to_addr + type_size
        c.memmove(to_addr, from_addr, type_size * (self.size - index))
        self._shorten_if_needed()
    
    def _lengthen_if_needed(self):
        if self.size >= self.capacity:
            self.capacity = 1 + self.capacity + self.capacity // 2
            new_indices = (c.c_uint32 * self.capacity)()
            c.memmove(new_indices, self.indices, c.sizeof(c.c_uint32) * self.size)
            self.indices = new_indices
    
    def _shorten_if_needed(self):
        if self.size < self.capacity // 2:
            self.shorten()

    def shorten(self):
        ''' Reduces the allocated memory to match the size. '''
        self.capacity = self.size
        new_indices = (c.c_uint32 * self.capacity)()
        c.memmove(new_indices, self.indices, c.sizeof(c.c_uint32) * self.size)
        self.indices = new_indices
    
    def print_SBA(self):
        ''' 
        Not to be confused with a normal print. 
        This prints out the raw contiguous ints allocated to the SBA, and indicates where the used mem ends.
        ```python
        >>> a = SBA(0, 1, 2, 3)
        >>> del a[0]
        >>> a.print_SBA()
            V
        1 2 3 3
        ```
        '''
        SBA.printSBA(c.byref(self))
    
    def turn_off_all(self, rm: SBA):
        '''
        Turns off all bits also contained in rm.
        ```python
        >>> a = SBA(1, 2, 3)
        >>> rm = SBA(2, 4)
        >>> a.turn_off_all(rm)
        >>> a
        [1, 3]
        ```
        '''
        SBA.turnOffAll(c.byref(self), c.byref(rm))
    
    def and_bits(a: SBA, b: SBA) -> SBA:
        '''
        ```python
        >>> SBA.and_bits(SBA(1, 2, 3), SBA(2, 3, 4))
        [2, 3]
        ```
        '''
        r = SBA(blank_cap = min(a.size, b.size))
        SBA.andBits(c.byref(r), c.byref(a), c.byref(b), 0)
        return r

    def and_size(a: SBA, b: SBA) -> int:
        '''
        Returns the number of bits in a AND in b.
        ```python
        >>> SBA.and_size(SBA(1, 2, 3), SBA(2, 3, 4))
        2
        '''
        r = (c.c_uint32)()
        SBA.andBits(c.byref(r), c.byref(a), c.byref(b), 1)
        return r.value

    def or_bits(a: SBA, b: SBA) -> SBA:
        '''
        ```python
        >>> SBA.or_bits(SBA(1, 2, 3), SBA(2, 3, 4))
        [1, 2, 3, 4]
        ```
        '''
        r = SBA(blank_cap = a.size + b.size)
        SBA.orBits(c.byref(r), c.byref(a), c.byref(b), 0, 0)
        return r

    def or_size(a: SBA, b: SBA) -> int:
        '''
        Returns the number of bits in a OR b.
        ```python
        >>> SBA.or_size(SBA(1, 2, 3), SBA(2, 3, 4))
        4
        ```
        '''
        r = (c.c_uint32)()
        SBA.orBits(c.byref(r), c.byref(a), c.byref(b), 0, 1)
        return r.value

    def xor_bits(a: SBA, b: SBA) -> SBA:
        '''
        ```python
        >>> SBA.xor_bits(SBA(1, 2, 3), SBA(2, 3, 4))
        [1, 4]
        ```
        '''
        r = SBA(blank_cap = a.size + b.size)
        SBA.orBits(c.byref(r), c.byref(a), c.byref(b), 1, 0)
        return r
    
    def xor_size(a: SBA, b: SBA) -> int:
        '''
        Returns the number of bits in a XOR b.
        ```python
        >>> SBA.xor_bits(SBA(1, 2, 3), SBA(2, 3, 4))
        2
        ```
        '''
        r = (c.c_uint32)()
        SBA.orBits(c.byref(r), c.byref(a), c.byref(b), 1, 1)
        return r.value
    
    def shift(self, n: int):
        ''' Bitshift '''
        if n > 0:
            SBA.lshift(c.byref(self), c.c_uint32(n))
        else:
            SBA.rshift(c.byref(self), c.c_uint32(-n))

    def __eq__(self, other: Union[Iterable[int], SBA]) -> bool:
        if isinstance(other, SBA):
            return bool(SBA.equal(c.byref(self), c.byref(other)))
        elif hasattr(other, "__getitem__"):
            ln = self.size
            return len(other) == ln and all(other[i] == self[i] for i in range(ln))
        else:
            return False
    
    def cp(self) -> SBA:
        ''' Returns deep copy of self. '''
        r = SBA(blank_cap = self.size)
        SBA.f_cp(c.byref(r), c.byref(self))
        return r

    def subsample(self, retain_amount: float) -> SBA:
        ''' Randomly flip bits off. 0 clears the list, and 1 leaves the list unchanged. '''
        if retain_amount < 0 or retain_amount > 1:
            raise SBAException("retain_amount must be in range [0,1]")
        SBA.f_subsample(c.byref(self), c.c_float(retain_amount))
    
    def encode_linear(input: float, num_on_bits: int, size: int) -> SBA:
        '''
        Encodes a float as an SBA.  
        input is in the value to encode, from 0 (inclusive) to 1 (also inclusive)  
        num_on_bits is the number of bits that will be flipped to ON  
        size is the total size of the array
        '''
        if input < 0: # nothing breaks if input is > 1
            raise SBAException("Can't encode a negative value in this function.")
        r = SBA(blank_cap = num_on_bits)
        r.size = r.capacity
        SBA.encodeLinear(c.c_float(input), c.c_uint32(size), c.byref(r))
        return r
    
    def encode_periodic(input: float, period: float, num_on_bits: int, size: int) -> SBA:
        '''
        input is the the value to encode. it is encoded linearly, except its encoding wraps back to 0 as it approaches period  
        num_on_bits is the number of bits that will be flipped to ON  
        size is the total size of the array
        '''
        if period <= 0:
            raise SBAException("Period must be positive.")
        elif num_on_bits > size:
            raise SBAException("The number of on bits can't exceed the size of the array.")
        r = SBA(blank_cap = num_on_bits)
        r.size = r.capacity
        SBA.encodePeriodic(c.c_float(input), c.c_float(period), c.c_uint32(size), c.byref(r))
        return r
    
    def from_np(arr: numpy.ndarray, deep_copy=True) -> SBA:
        '''
        Converts from a numpy array.
        ```python3
        >>> arr = numpy.array([0, 2, 50], numpy.uint32)
        >>> a = SBA.from_np(arr, False)
        >>> arr = None # Important!
        >>> a
        [0, 2, 50]
        ```
        If deep_copy is True, then the returned SBA has a separate copy of the data.

        If deep_copy is False, then the returned SBA will be a shallow copy of the numpy array's buffer.  
        Read: Two objects sharing the same section of memory.  
        Set the input array to None after shallow copying to prevent aliasing, as this will lead to segfaults if the SBAs indices are modified to no longer be valid.  
        '''
        if arr.dtype != numpy.uint32:
            raise SBAException("The numpy array must be of type uint32, try `arr.astype(np.uint32)`")
        size = arr.size
        if SBA.do_checking and not all(arr[i] <= arr[i+1] for i in range(size-1)):
            raise SBAException("the indices must be in ascending order")
        a = SBA(uninit=True)
        a.size = (c.c_uint32)(size)
        a.capacity = a.size
        arr_ptr = arr.ctypes.data_as(c.POINTER(c.c_uint32))
        if deep_copy:
            a.indices = (c.c_uint32 * size)()
            c.memmove(a.indices, arr_ptr, c.sizeof(c.c_uint32) * size)
        else:
            a.indices = arr_ptr
        return a

    def to_np(self, deep_copy=True) -> numpy.ndarray:
        '''
        Converts to a numpy array.
        ```python3
        >>> a = SBA(0, 2, 50)
        >>> arr = a.to_np(False)
        >>> a = None # Important!
        >>> arr
        array([ 0,  2, 50], dtype=uint32)
        ```
        If deep_copy is False, then the returned array will be a shallow copy of this SBA.  
        Read: Two objects sharing the same section of memory.  
        Set this SBA to None after shallow copying to prevent aliasing.

        If deep_copy is True, then the returned array has a separate copy of the data.
        '''
        arr = numpy.frombuffer(c.cast(self.indices, c.POINTER(c.c_uint32 * self.size)).contents, dtype=numpy.uint32)
        if deep_copy:
            new_indices = (c.c_uint32 * self.size)()
            c.memmove(new_indices, self.indices, c.sizeof(c.c_uint32) * self.size)
            self.indices = c.cast(new_indices, c.POINTER(c.c_uint32))
        return arr
