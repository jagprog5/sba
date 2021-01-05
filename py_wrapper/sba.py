from __future__ import annotations
import ctypes
from typing import List, Iterable, Optional
import pathlib

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

class SBA(ctypes.Structure):
    '''
    Sparse bit array. This is a wrapper for sba.h
    '''

    do_checking = True
    ml_lib = None

    _fields_ = [
        ('indices',ctypes.POINTER(ctypes.c_uint64)),
        ('size',ctypes.c_uint64),
        ('capacity',ctypes.c_uint64)]
    
    def _init_lib_if_needed():
        if SBA.ml_lib is None:
            lib_folder = pathlib.Path(__file__).absolute().parents[1] / "build"
            ml_lib_file = lib_folder / "ml_lib.so"
            try:
                SBA.ml_lib = ctypes.CDLL(ml_lib_file)
            except OSError as e:
                print(e)
                raise SBAException("Couldn't load ml_lib.so in build dir. Run `make shared` at repo root")

            # printSBA
            SBA.printSBA = SBA.ml_lib.printSBA
            SBA.printSBA.restype = None
            SBA.printSBA.argtype = [ctypes.POINTER(SBA)]

            # turnOn
            # requires realloc, manual implementation on Python side in set_bit

            # turnOff
            SBA.turnOff = SBA.ml_lib.turnOff
            SBA.turnOff.restype = None
            SBA.turnOff.argtype = [ctypes.POINTER(SBA), ctypes.c_uint64]

            # getBit
            SBA.getBit = SBA.ml_lib.getBit
            SBA.getBit.restype = ctypes.c_uint8
            SBA.getBit.argtype = [ctypes.POINTER(SBA), ctypes.c_uint64]

            # turnOffAll
            SBA.turnOffAll = SBA.ml_lib.turnOffAll
            SBA.turnOffAll.restype = None
            SBA.turnOffAll.argtype = [ctypes.POINTER(SBA)] * 2

            # andBits
            SBA.andBits = SBA.ml_lib.andBits
            SBA.andBits.restype = None
            SBA.andBits.argtype = [ctypes.POINTER(SBA)] * 3

            # andSize
            SBA.andSize = SBA.ml_lib.andSize
            SBA.andSize.restype = ctypes.c_uint64
            SBA.andSize.argtype = [ctypes.POINTER(SBA)] * 2

            # orBits
            SBA.orBits = SBA.ml_lib.orBits
            SBA.orBits.restype = None
            SBA.orBits.argtype = [ctypes.POINTER(SBA)] * 3 + [ctypes.c_uint8]

            # orSize
            SBA.orSize = SBA.ml_lib.orSize
            SBA.orSize.restype = ctypes.c_uint64
            SBA.orSize.argtype = [ctypes.POINTER(SBA)] * 2 + [ctypes.c_uint8]

            # rshift
            SBA.rshift = SBA.ml_lib.rshift
            SBA.rshift.restype = None
            SBA.rshift.argtype = [ctypes.POINTER(SBA), ctypes.c_uint64]

            # lshift
            SBA.lshift = SBA.ml_lib.lshift
            SBA.lshift.restype = None
            SBA.lshift.argtype = [ctypes.POINTER(SBA), ctypes.c_uint64]

            # equality
            SBA.equal = SBA.ml_lib.equal
            SBA.equal.restype = ctypes.c_uint8
            SBA.equal.argtype = [ctypes.POINTER(SBA)] * 2

            # cp
            SBA.f_cp = SBA.ml_lib.cp
            SBA.f_cp.restype = None
            SBA.f_cp.argtype = [ctypes.POINTER(SBA)] * 2

            # subsample
            seed_rand = SBA.ml_lib.seed_rand
            seed_rand()
            SBA.f_subsample = SBA.ml_lib.subsample3
            SBA.f_subsample.restype = None
            SBA.f_subsample.argtype = [ctypes.POINTER(SBA), ctypes.c_uint64]

            # encodeLinear
            SBA.encodeLinear = SBA.ml_lib.encodeLinear
            SBA.encodeLinear.restype = None
            SBA.encodeLinear.argtype = [ctypes.c_float, ctypes.c_uint64, ctypes.POINTER(SBA)]

            # encodePeriodic
            SBA.encodePeriodic = SBA.ml_lib.encodePeriodic
            SBA.encodePeriodic.restype = None
            SBA.encodePeriodic.argtype = [ctypes.c_float, ctypes.c_float, ctypes.c_uint64, ctypes.POINTER(SBA)]
    
    def __init__(self, *on_bits: Iterable[int], **_special):
        SBA._init_lib_if_needed()
        if not 'blank_size' in _special:
            if hasattr(on_bits[0], "__getitem__"): # support list, tuple, etc as first arg
                on_bits = on_bits[0]
            ln = len(on_bits)
            if SBA.do_checking:
                if not all(isinstance(on_bits[i], int) for i in range(ln)):
                    raise SBAException("on_bits must only contain ints.")
                if not all(on_bits[i] <= on_bits[i+1] for i in range(ln-1)):
                    raise SBAException("on_bits must be in ascending order.")
                if not all(on_bits[i] >= 0 for i in range(ln)):
                    raise SBAException("on_bits must only contain non-negative integers.")
            self.indices = (ctypes.c_uint64 * ln)(*on_bits)
            self.size = (ctypes.c_uint64)(ln)
            self.capacity = (ctypes.c_uint64)(ln)
        else:
            ln = _special['blank_size']
            self.indices = (ctypes.c_uint64 * ln)()
            self.size = (ctypes.c_uint64)(ln)
            self.capacity = (ctypes.c_uint64)(ln)
    
    def enable_checking():
        '''This is enabled by default. On creation of a SBA, ensures that on_bits are valid. '''
        SBA.do_checking = True
    
    def disable_checking():
        ''' Disables check that ensure that on_bits are valid on SBA creation. '''
        SBA.do_checking = False
    
    def to_list(self) -> List[int]:
        return [self.indices[i] for i in range(self.size)]
    
    def __str__(self):
        return str(self.to_list())
    
    def __bool__(self):
        return self.size > 0

    def __iter__(self):
        ''' Gives indices of ON bits. '''
        return SBAIterator(self)
    
    def __len__(self):
        ''' Returns the number of bits that are ON. '''
        return self.size
    
    def __add__(self, other):
        if isinstance(other, str):
            return str(self) + other
        elif isinstance(other, int):
            return self.cp().set_bit(other, True) # Return modified copy. Do not modify self
        elif isinstance(other, SBA):
            return SBA.or_bits(self, other)
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
        else:
            raise TypeError(str(type(other)) + " not supported for * or & ops.")
    
    def __or__(self, other):
        if isinstance(other, SBA):
            return SBA.or_bits(self, other)
        else:
            raise TypeError(str(type(other)) + " not supported for | op.")
    
    def __xor__(self, other):
        if isinstance(other, SBA):
            return SBA.xor_bits(self, other)
        else:
            raise TypeError(str(type(other)) + " not supported for ^ op.")
    
    def __sub__(self, other):
        if isinstance(other, int):
            return self.cp().set_bit(other, False) # Return modified copy. Do not modify self
        elif isinstance(other, SBA):
            return self.cp().turn_off_all(other)
        else:
            raise TypeError(str(type(other)) + " not supported for - op.")
    
    def __lshift__(self, n):
        return self.__rshift__(-n)

    def __rshift__(self, n):
        if isinstance(n, int):
            return self.shift(n)
        else:
            raise TypeError(str(type(other)) + " not supported for >> or << ops.")
    
    def _check_index(self, index: int) -> int:
        if index >= self.size:
            raise SBAException("Index out of bounds.")
        if index < 0:
            index = self.size + index
            if index < 0:
                raise SBAException("Index out of bounds.")
        return index

    def __getitem__(self, index: int) -> int:
        '''
        Returns the index of the i-th ON bit.
        Not to be confused with get_bit
        '''
        return self.indices[self._check_index(index)]
    
    
    def get_bit(self, index: int) -> bool:
        '''
        Returns the value of the i-th bit.
        '''
        if index < 0:
            raise SBAException("Requires positive index.")
        return bool(SBA.getBit(ctypes.pointer(self), ctypes.c_uint64(index)))
    
    def __setitem__(self, index: int, value: int):
        '''
        Turns off the index-th ON bit, and turns on the value-th bit.
        Not to be confused with set_bit
        '''
        self.set_bit(self.indices[self._check_index(index)], False)
        self.set_bit(value, True)
    
    def set_bit(self, index: int, value: bool):
        '''
        Sets the value of the i-th bit
        '''
        if index < 0:
            raise SBAException("Requires positive index.")
        if value:
            # can't call turnOn in c code since realloc must be done here
            left = 0
            right = self.size - 1
            middle = 0
            mid_val = 0
            while left <= right:
                middle = (right + left) // 2
                mid_val = self.indices[middle]
                if mid_val < index:
                    left = middle + 1
                elif mid_val > index:
                    right = middle - 1
                else:
                    return # skip duplicate
            if index > mid_val:
                middle += 1
            type_size = ctypes.sizeof(ctypes.c_uint64)
            if self.size >= self.capacity:
                self.capacity <<= 1
                new_indices = (ctypes.c_uint64 * self.capacity)()
                ctypes.memmove(new_indices, self.indices, type_size * self.size)
                self.indices = new_indices
            addr = ctypes.cast(self.indices, ctypes.c_void_p).value
            from_addr = addr + middle * type_size
            to_addr = from_addr + type_size
            ctypes.memmove(to_addr, from_addr, type_size * (self.size - middle))
            self.size += 1
            self.indices[middle] = index
        else:
            SBA.turnOff(ctypes.pointer(self), ctypes.c_uint64(index))
        return self
        
    def __delitem__(self, index: int):
        ''' Turns the i-th ON bit to OFF '''
        index = self._check_index(index)
        self.size -= 1
        addr = ctypes.cast(self.indices, ctypes.c_void_p).value # disgusting
        type_size = ctypes.sizeof(ctypes.c_uint64)
        to_addr = addr + index * type_size
        from_addr = to_addr + type_size
        ctypes.memmove(to_addr, from_addr, type_size * (self.size - index))
    
    def shorten(self):
        ''' Reduces the allocated memory to match the size. '''
        self.capacity = self.size
        new_indices = (ctypes.c_uint64 * self.capacity)()
        ctypes.memmove(new_indices, self.indices, ctypes.sizeof(ctypes.c_uint64) * self.capacity)
        self.indices = new_indices
        return self
    
    def print_SBA(self):
        ''' 
        Not to be confused with a normal print. 
        This prints out the raw contiguous uints allocated to the SBA, and indicates where the used mem ends.
        '''
        SBA.printSBA(ctypes.pointer(self))
    
    def turn_off_all(self, rm: SBA):
        ''' Turns off all bits also contained in rm. '''
        SBA.turnOffAll(ctypes.pointer(self), ctypes.pointer(rm))
        return self
    
    def and_bits(a: SBA, b: SBA) -> SBA:
        r = SBA(blank_size = min(a.size, b.size))
        SBA.andBits(ctypes.pointer(r), ctypes.pointer(a), ctypes.pointer(b))
        return r

    def and_size(a: SBA, b: SBA) -> int:
        ''' Returns the number of bits in a AND in b. '''
        return SBA.andSize(ctypes.pointer(a), ctypes.pointer(b))

    def or_bits(a: SBA, b: SBA) -> SBA:
        r = SBA(blank_size = a.size + b.size)
        SBA.orBits(ctypes.pointer(r), ctypes.pointer(a), ctypes.pointer(b), ctypes.c_uint8(0))
        return r

    def or_size(a: SBA, b: SBA) -> int:
        ''' Returns the number of bits in a OR b. '''
        return SBA.orSize(ctypes.pointer(a), ctypes.pointer(b), ctypes.c_uint8(0))

    def xor_bits(a: SBA, b: SBA) -> SBA:
        r = SBA(blank_size = a.size + b.size)
        SBA.orBits(ctypes.pointer(r), ctypes.pointer(a), ctypes.pointer(b), ctypes.c_uint8(1))
        return r
    
    def xor_sizes(a: SBA, b: SBA) -> iny:
        ''' Returns the number of bits in a XOR b. '''
        return SBA.orSize(ctypes.pointer(a), ctypes.pointer(b), ctypes.c_uint8(1))
    
    def shift(self, n: int):
        ''' Bitshift '''
        if n > 0:
            SBA.rshift(ctypes.pointer(self), ctypes.c_uint64(n))
        else:
            SBA.lshift(ctypes.pointer(self), ctypes.c_uint64(-n))
        return self

    def __eq__(self, other):
        return bool(SBA.equal(ctypes.pointer(self), ctypes.pointer(other)))
    
    def cp(self) -> SBA:
        ''' Returns deep copy of self. '''
        r = SBA(blank_size = self.size)
        SBA.f_cp(ctypes.pointer(r), ctypes.pointer(self))
        return r

    def subsample(self, n: int) -> SBA:
        ''' Randomly flips bits off. There is a 1 / n chance of each bit remaining on. '''
        SBA.f_subsample(ctypes.pointer(self), ctypes.c_uint64(n))
        return self
    
    def encode_linear(input: float, num_on_bits: int, size: int) -> SBA:
        '''
        Encodes a float as a SBA
        input is in range [0,1]
        num_on_bits is the number of bits that will be flipped to ON
        size is the total size of the array
        '''
        if input < 0: # nothing breaks if input is > 1
            raise SBAException("Can't encode a negative value in this function.")
        r = SBA(blank_size = num_on_bits)
        SBA.encodeLinear(ctypes.c_float(input), ctypes.c_uint64(size), ctypes.pointer(r))
        return r
    
    def encode_periodic(input: float, period: float, num_on_bits: int, size: int) -> SBA:
        '''
        input is the the value to encode. it is encoded linearly,
            except its encoding wraps back to 0 as it approaches period
        num_on_bits is the number of bits that will be flipped to ON
        size is the total size of the array
        '''
        if period <= 0: # nothing breaks if input is > 1
            raise SBAException("Period must be positive.")
        r = SBA(blank_size = num_on_bits)
        SBA.encodePeriodic(ctypes.c_float(input), ctypes.c_float(period), ctypes.c_uint64(size), ctypes.pointer(r))
        return r

if __name__ == "__main__":
    a = SBA(1, 2, 8, 9)
    print(a)
    a[-2] = 10
    print(a)
    print(a - 2)
    b = SBA(a)
    a.set_bit(2, False)
    a.set_bit(1000, True)
    print(a + " & " + b + " = " + a * b)
