from __future__ import annotations
import ctypes as c
import numpy
from typing import List, Iterable, Union
import pathlib

class SBAException(Exception):
    pass

def _sba_struct_factory(capacity):
    # flexible array member, must be done as a local class
    class SBAStruct(c.Structure):
        _fields_ = [
            ('size', c.c_uint32),
            ('capacity', c.c_uint32),
            ('indices', c.c_uint32 * capacity)]
    ret = SBAStruct()
    ret.capacity = (c.c_uint32)(capacity)
    return ret

class SBAStructIterator:
    def __init__(self, sba: SBAStruct):
        self._sba = sba
        self._index = 0
    
    def __next__(self):
        if self._index >= self._sba.size:
            raise StopIteration
        ret = self._sba.indices[self._index]
        self._index += 1
        return ret
        
class SBA():
    '''
    Sparse bit array. This is a wrapper for sba.h
    '''

    do_checking = True
    sba_lib = None

    _fields_ = [
        ('size',c.c_uint32),
        ('capacity',c.c_uint32),
        ('indices',c.POINTER(c.c_uint32))]
    
    def _init_lib_if_needed():
        if SBA.sba_lib is None:
            lib_folder = pathlib.Path(__file__).absolute().parents[1] / "build"
            sba_lib_file = lib_folder / "sba_lib.so"
            try:
                SBA.sba_lib = c.CDLL(sba_lib_file)
            except OSError as e:
                print(e)
                raise SBAException("Couldn't load sba_lib.so in build dir. Run `make shared` at repo root")

            # printSBA
            SBA.printSBA = SBA.sba_lib.printSBA
            SBA.printSBA.restype = None
            SBA.printSBA.argtype = [c.c_void_p]

            # turnOn, turnOff
            # both require realloc, manually implemented

            # getBit
            SBA.getBit = SBA.sba_lib.getBit
            SBA.getBit.restype = c.c_uint8
            SBA.getBit.argtype = [c.c_void_p, c.c_uint32]

            # turnOffAll
            SBA.turnOffAll = SBA.sba_lib.turnOffAll
            SBA.turnOffAll.restype = None
            SBA.turnOffAll.argtype = [c.c_void_p] * 2

            # andBits
            SBA.andBits = SBA.sba_lib.andBits
            SBA.andBits.restype = None
            SBA.andBits.argtype = [c.c_void_p] * 3

            # andSize
            SBA.andSize = SBA.sba_lib.andSize
            SBA.andSize.restype = c.c_uint32
            SBA.andSize.argtype = [c.c_void_p] * 2

            # orBits
            SBA.orBits = SBA.sba_lib.orBits
            SBA.orBits.restype = None
            SBA.orBits.argtype = [c.c_void_p] * 3 + [c.c_uint8]

            # orSize
            SBA.orSize = SBA.sba_lib.orSize
            SBA.orSize.restype = c.c_uint32
            SBA.orSize.argtype = [c.c_void_p] * 2 + [c.c_uint8]

            # rshift
            SBA.rshift = SBA.sba_lib.rshift
            SBA.rshift.restype = None
            SBA.rshift.argtype = [c.c_void_p, c.c_uint32]

            # lshift
            SBA.lshift = SBA.sba_lib.lshift
            SBA.lshift.restype = None
            SBA.lshift.argtype = [c.c_void_p, c.c_uint32]

            # equality
            SBA.equal = SBA.sba_lib.equal
            SBA.equal.restype = c.c_uint8
            SBA.equal.argtype = [c.c_void_p] * 2

            # cp
            SBA.f_cp = SBA.sba_lib.cp
            SBA.f_cp.restype = None
            SBA.f_cp.argtype = [c.c_void_p] * 2

            # subsample
            SBA.sba_lib.seed_rand()
            SBA.f_subsample = SBA.sba_lib.subsample
            SBA.f_subsample.restype = None
            SBA.f_subsample.argtype = [c.c_void_p, c.c_float]

            # encodeLinear
            SBA.encodeLinear = SBA.sba_lib.encodeLinear
            SBA.encodeLinear.restype = None
            SBA.encodeLinear.argtype = [c.c_float, c.c_uint32, c.c_void_p]

            # encodePeriodic
            SBA.encodePeriodic = SBA.sba_lib.encodePeriodic
            SBA.encodePeriodic.restype = None
            SBA.encodePeriodic.argtype = [c.c_float, c.c_float, c.c_uint32, c.c_void_p]
    
    def __init__(self, *on_bits: Union[int, Iterable[int], SBA], **_special):
        SBA._init_lib_if_needed()
        if not 'blank_size' in _special:
            if len(on_bits) > 0 and hasattr(on_bits[0], "__getitem__"): # support list, tuple, etc as first arg
                on_bits = on_bits[0]
            ln = len(on_bits)
            if SBA.do_checking:
                if not all(isinstance(on_bits[i], int) for i in range(ln)):
                    raise SBAException("on_bits must only contain ints.")
                if not all(on_bits[i] <= on_bits[i+1] for i in range(ln-1)):
                    raise SBAException("on_bits must be in ascending order.")
                if not all(on_bits[i] >= 0 for i in range(ln)):
                    raise SBAException("on_bits must only contain non-negative integers.")
            self.struct = _sba_struct_factory(ln)
            self.struct.indices = (c.c_uint32 * ln)(*on_bits)
            self.struct.size = self.struct.capacity
        else:
            self.struct = _sba_struct_factory(_special['blank_size'])
    
    def enable_checking():
        '''This is enabled by default. On creation of an SBA, ensures that on_bits are valid. '''
        SBA.do_checking = True
    
    def disable_checking():
        ''' Disables check that ensure that on_bits are valid on SBA creation. '''
        SBA.do_checking = False
    
    def to_list(self) -> List[int]:
        return [self.struct.indices[i] for i in range(self.struct.size)]
    
    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        return str(self.to_list())
    
    def __bool__(self):
        return self.struct.size > 0

    def __iter__(self):
        ''' Gives indices of ON bits. '''
        return SBAStructIterator(self.struct)
    
    def __len__(self):
        ''' Returns the number of bits that are ON. '''
        return self.struct.size
    
    def __add__(self, other):
        if isinstance(other, str):
            return str(self) + other
        elif isinstance(other, int):
            return self.cp().set_bit(other, True) # Return modified copy. Do not modify self
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
        return self.__mul__()
    
    def __mul__(self, other):
        if isinstance(other, SBA):
            return SBA.and_bits(self, other)
        elif isinstance(other, int):
            return self.get_bit(other)
        elif isinstance(other, float):
            return self.cp().subsample(other)
        else:
            raise TypeError(str(type(other)) + " not supported for * or & ops.")
    
    def __rmul__(self, other):
        if isinstance(other, float):
            return self.cp().subsample(other)
        else:
            raise TypeError(str(type(other)) + " not supported for reverse * op.")
    
    def __or__(self, other):
        return self.__add__()
    
    def __xor__(self, other):
        if isinstance(other, SBA):
            return SBA.xor_bits(self, other)
        else:
            raise TypeError(str(type(other)) + " not supported for ^ op.")
    
    def __sub__(self, other):
        if isinstance(other, int):
            return self.cp().set_bit(other, False)
        elif isinstance(other, SBA):
            return self.cp().turn_off_all(other)
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
        if isinstance(n, int):
            return self.cp().shift(n)
        else:
            raise TypeError(str(type(other)) + " not supported for >> or << ops.")
    
    def _check_index(self, index: int) -> int:
        if index >= self.struct.size:
            raise SBAException("Index out of bounds.")
        if index < 0:
            index = self.struct.size + index
            if index < 0:
                raise SBAException("Index out of bounds.")
        return index

    def __getitem__(self, index: int) -> int:
        '''
        Returns the index of the i-th ON bit.
        Not to be confused with get_bit
        '''
        return self.struct.indices[self._check_index(index)]
    
    
    def get_bit(self, index: int) -> bool:
        '''
        Returns the value of the i-th bit.
        '''
        if index < 0:
            raise SBAException("Requires non-negative index.")
        return bool(SBA.getBit(c.byref(self.struct), c.c_uint32(index)))
    
    def __setitem__(self, index: int, value: int):
        '''
        Turns off the index-th ON bit, and turns on the value-th bit.
        Not to be confused with set_bit
        '''
        self.__delitem__(index)
        self.set_bit(value, True)
    
    def set_bit(self, index: int, value: bool):
        '''
        Sets the value of the i-th bit
        '''
        if index < 0:
            raise SBAException("Requires non-negative index.")
        left = 0
        right = self.struct.size - 1
        middle = 0
        mid_val = 0xFFFFFFFF # u32 max
        while left <= right:
            middle = (right + left) // 2
            mid_val = self.struct.indices[middle]
            if mid_val < index:
                left = middle + 1
            elif mid_val > index:
                right = middle - 1
            else:
                if not value:
                    self.__delitem__(middle)
                return self # skip duplicate
        if not value:
            return self
        if index > mid_val:
            middle += 1
        self._lengthen_if_needed()
        ptr = c.cast(c.pointer(self.struct.indices), c.c_void_p)
        addr = ptr.value
        type_size = c.sizeof(c.c_uint32)
        from_addr = addr + middle * type_size
        to_addr = from_addr + type_size
        c.memmove(to_addr, from_addr, type_size * (self.struct.size - middle))
        self.struct.size += 1
        self.struct.indices[middle] = index
        return self
        
    def __delitem__(self, index: int):
        ''' Turns the i-th ON bit to OFF '''
        index = self._check_index(index)
        self.struct.size -= 1
        ptr = c.cast(c.pointer(self.struct.indices), c.c_void_p)
        addr = ptr.value
        type_size = c.sizeof(c.c_uint32)
        to_addr = addr + index * type_size
        from_addr = to_addr + type_size
        c.memmove(to_addr, from_addr, type_size * (self.struct.size - index))
        self._shorten_if_needed()
        return self
    
    def _lengthen_if_needed(self):
        if self.struct.size >= self.struct.capacity:
            new_cap = 1 + self.struct.capacity + self.struct.capacity // 2
            new_struct = _sba_struct_factory(new_cap)
            new_struct.size = self.struct.size
            c.memmove(new_struct.indices, self.struct.indices, c.sizeof(c.c_uint32) * new_struct.capacity)
            self.struct = new_struct

    
    def _shorten_if_needed(self):
        if self.struct.size < self.struct.capacity // 2:
            self.shorten()

    def shorten(self):
        ''' Reduces the allocated memory to match the size. '''
        new_struct = _sba_struct_factory(self.struct.size)
        new_struct.size = new_struct.capacity
        c.memmove(new_struct.indices, self.struct.indices, c.sizeof(c.c_uint32) * new_struct.capacity)
        self.struct = new_struct
        return self
    
    def print_SBA(self):
        ''' 
        Not to be confused with a normal print. 
        This prints out the raw contiguous ints allocated to the SBA, and indicates where the used mem ends.
        '''
        SBA.printSBA(c.byref(self.struct))
    
    def turn_off_all(self, rm: SBA):
        ''' Turns off all bits also contained in rm. '''
        SBA.turnOffAll(c.byref(self.struct), c.byref(rm.struct))
        return self
    
    def and_bits(a: SBA, b: SBA) -> SBA:
        r = SBA(blank_size = min(a.struct.size, b.struct.size))
        SBA.andBits(c.byref(r.struct), c.byref(a.struct), c.byref(b.struct))
        return r

    def and_size(a: SBA, b: SBA) -> int:
        ''' Returns the number of bits in a AND in b. '''
        return SBA.andSize(c.byref(a.struct), c.byref(b.struct))

    def or_bits(a: SBA, b: SBA) -> SBA:
        r = SBA(blank_size = a.struct.size + b.struct.size)
        SBA.orBits(c.byref(r.struct), c.byref(a.struct), c.byref(b.struct), c.c_uint8(0))
        return r

    def or_size(a: SBA, b: SBA) -> int:
        ''' Returns the number of bits in a OR b. '''
        return SBA.orSize(c.byref(a.struct), c.byref(b.struct), c.c_uint8(0))

    def xor_bits(a: SBA, b: SBA) -> SBA:
        r = SBA(blank_size = a.struct.size + b.struct.size)
        SBA.orBits(c.byref(r.struct), c.byref(a.struct), c.byref(b.struct), c.c_uint8(1))
        return r
    
    def xor_size(a: SBA, b: SBA) -> iny:
        ''' Returns the number of bits in a XOR b. '''
        return SBA.orSize(c.byref(a.struct), c.byref(b.struct), c.c_uint8(1))
    
    def shift(self, n: int):
        ''' Bitshift '''
        if n > 0:
            SBA.lshift(c.byref(self.struct), c.c_uint32(n))
        else:
            SBA.rshift(c.byref(self.struct), c.c_uint32(-n))
        return self

    def __eq__(self, other) -> bool:
        if isinstance(other, SBA):
            return bool(SBA.equal(c.byref(self.struct), c.byref(other.struct)))
        elif hasattr(other, "__getitem__"):
            ln = self.struct.size
            return len(other) == ln and all(other[i] == self[i] for i in range(ln))
        else:
            return False
    
    def cp(self) -> SBA:
        ''' Returns deep copy of self. '''
        r = SBA(blank_size = self.struct.size)
        SBA.f_cp(c.byref(r.struct), c.byref(self.struct))
        return r

    def subsample(self, retain_amount: float) -> SBA:
        ''' Randomly flips bits off. 0 clears the list, and 1 leaves the list unchanged. '''
        if retain_amount < 0 or retain_amount > 1:
            raise SBAException("retain_amount must be in range [0,1]")
        SBA.f_subsample(c.byref(self.struct), c.c_float(retain_amount))
        return self
    
    def encode_linear(input: float, num_on_bits: int, size: int) -> SBA:
        '''
        Encodes a float as an SBA
        input is in range [0,1]
        num_on_bits is the number of bits that will be flipped to ON
        size is the total size of the array
        '''
        if input < 0: # nothing breaks if input is > 1
            raise SBAException("Can't encode a negative value in this function.")
        r = SBA(blank_size = num_on_bits)
        r.struct.size = r.struct.capacity
        SBA.encodeLinear(c.c_float(input), c.c_uint32(size), c.byref(r.struct))
        return r
    
    def encode_periodic(input: float, period: float, num_on_bits: int, size: int) -> SBA:
        '''
        input is the the value to encode. it is encoded linearly,
            except its encoding wraps back to 0 as it approaches period
        num_on_bits is the number of bits that will be flipped to ON
        size is the total size of the array
        '''
        if period <= 0:
            raise SBAException("Period must be positive.")
        elif num_on_bits > size:
            raise SBAException("The number of on bits can't exceed the size of the array.")
        r = SBA(blank_size = num_on_bits)
        r.struct.size = r.struct.capacity
        SBA.encodePeriodic(c.c_float(input), c.c_float(period), c.c_uint32(size), c.byref(r.struct))
        return r
    
    def from_numpy_array(arr: numpy.ndarray) -> SBA:
        '''
        copys the memory from the numpy array into an SBA
        '''
        if arr.dtype != numpy.uint32:
            raise SBAException("The numpy array must be of type uint32, try `arr.astype(np.uint32)`")
        size = arr.size
        if SBA.do_checking and not all(arr[i] <= arr[i+1] for i in range(size-1)):
            raise SBAException("the indices must be in ascending order")

        r = SBA(blank_size = size)
        r.struct.size = r.struct.capacity
        src_ptr = arr.ctypes.data_as(c.POINTER(c.c_uint32))
        c.memmove(r.struct.indices, src_ptr, c.sizeof(c.c_uint32) * c.sizeof(c.c_uint32))

        return r

    def to_numpy_array(self, deep_copy=True) -> numpy.ndarray:
        '''
        Warning:
        If deep_copy is False, then the returned array will be a shallow copy of this SBA.
        Read: Two objects modifying the same section of memory.
        Which is fine if you know what you're doing, but aliasing is scary.
        Perhaps set this SBA to None after shallow copying

        If deep_copy is True, then the returned array has a separate copy of the data.
        '''
        arr = numpy.frombuffer(self.struct.indices, dtype=numpy.uint32)
        if deep_copy:
            self.struct = self.cp().struct
        return arr

