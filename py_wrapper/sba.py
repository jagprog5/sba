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
    sba_lib = None

    _fields_ = [
        ('indices',ctypes.POINTER(ctypes.c_uint64)),
        ('size',ctypes.c_uint64),
        ('capacity',ctypes.c_uint64)]
    
    def __init__(self, *on_bits: Iterable[int], **_special):
        if SBA.sba_lib is None:
            lib_folder = pathlib.Path(__file__).absolute().parents[1] / "build"
            sba_lib_file = lib_folder / "sba.so"
            try:
                SBA.sba_lib = ctypes.CDLL(sba_lib_file)
            except OSError as e:
                print(e)
                raise SBAException("Couldn't load sba.so in build dir. Run `make shared` at repo root")

            # printSBA
            SBA.printSBA = SBA.sba_lib.printSBA
            SBA.printSBA.restype = None
            SBA.printSBA.argtype = [ctypes.POINTER(SBA)]

            # turnOffAll
            SBA.turnOffAll = SBA.sba_lib.turnOffAll
            SBA.turnOffAll.restype = None
            SBA.turnOffAll.argtype = [ctypes.POINTER(SBA)] * 2

            # andBits
            SBA.andBits = SBA.sba_lib.andBits
            SBA.andBits.restype = None
            SBA.andBits.argtype = [ctypes.POINTER(SBA)] * 3

            # andSize
            SBA.andSize = SBA.sba_lib.andSize
            SBA.andSize.restype = ctypes.c_uint64
            SBA.andSize.argtype = [ctypes.POINTER(SBA)] * 2

            # orBits
            SBA.orBits = SBA.sba_lib.orBits
            SBA.orBits.restype = None
            SBA.orBits.argtype = [ctypes.POINTER(SBA)] * 3

            # orSize
            SBA.orSize = SBA.sba_lib.orSize
            SBA.orSize.restype = ctypes.c_uint64
            SBA.orSize.argtype = [ctypes.POINTER(SBA)] * 2

            # shift
            SBA.f_shift = SBA.sba_lib.shift
            SBA.f_shift.restype = None
            SBA.f_shift.argtype = [ctypes.POINTER(SBA), ctypes.c_uint64]

            # equality
            SBA.equal = SBA.sba_lib.equal
            SBA.equal.restype = ctypes.c_uint8
            SBA.equal.argtype = [ctypes.POINTER(SBA)] * 2

            # cp
            SBA.f_cp = SBA.sba_lib.cp
            SBA.f_cp.restype = None
            SBA.f_cp.argtype = [ctypes.POINTER(SBA)] * 2

            # subsample
            seed_rand = SBA.sba_lib.seed_rand
            seed_rand()
            SBA.f_subsample = SBA.sba_lib.subsample3
            SBA.f_subsample.restype = None
            SBA.f_subsample.argtype = [ctypes.POINTER(SBA), ctypes.c_uint64]

            # encodeLinear
            SBA.encodeLinear = SBA.sba_lib.encodeLinear
            SBA.encodeLinear.restype = None
            SBA.encodeLinear.argtype = [ctypes.c_float, ctypes.c_uint64, ctypes.POINTER(SBA)]

            # encodePeriodic
            SBA.encodePeriodic = SBA.sba_lib.encodePeriodic
            SBA.encodePeriodic.restype = None
            SBA.encodePeriodic.argtype = [ctypes.c_float, ctypes.c_float, ctypes.c_uint64, ctypes.POINTER(SBA)]

        # ========================================================================================
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
            self.indices = (ctypes.c_uint64 * ln)(0) # <-- TODO leave mem uninitalized rather than set to 0
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

    def __repr__(self):
        return str(self.to_list())

    def __iter__(self):
        return SBAIterator(self)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, key):
        return self.indices[key]
    
    def print_SBA(self):
        ''' 
        Not to be confused with a normal print. 
        This prints out the raw contiguous uints allocated to the SBA, and indicates where the used mem ends.
        '''
        SBA.printSBA(ctypes.pointer(self))
    
    def turn_off_all(self, rm: SBA):
        ''' Turns off all bits also contained in rm. '''
        SBA.turnOffAll(ctypes.pointer(self), ctypes.pointer(rm))
    
    def and_bits(a: SBA, b: SBA) -> SBA:
        r = SBA(blank_size = min(a.size, b.size))
        SBA.andBits(ctypes.pointer(r), ctypes.pointer(a), ctypes.pointer(b))
        return r

    def and_size(a: SBA, b: SBA) -> int:
        ''' Returns the number of bits in a AND in b. '''
        return SBA.andSize(ctypes.pointer(a), ctypes.pointer(b))

    def or_bits(a: SBA, b: SBA) -> SBA:
        r = SBA(blank_size = a.size + b.size)
        SBA.orBits(ctypes.pointer(r), ctypes.pointer(a), ctypes.pointer(b))
        return r

    def or_size(a: SBA, b: SBA) -> int:
        ''' Returns the number of bits in a OR in b. '''
        return SBA.orSize(ctypes.pointer(a), ctypes.pointer(b))
    
    def shift(self, n: int):
        ''' Increases by bitshifting n places '''
        SBA.f_shift(ctypes.pointer(self), ctypes.c_uint64(n))
        return self

    def __eq__(self, other):
        return bool(SBA.equal(ctypes.pointer(self), ctypes.pointer(other)))
    
    def cp(self) -> SBA:
        ''' Returns deep copy of self. '''
        r = SBA(blank_size = self.size)
        SBA.f_cp(ctypes.pointer(r), ctypes.pointer(self))
        return r

    def subsample(self, n: int) -> SBA:
        ''' Randomly flips bits off. There is a  1 / n chance of each bit remaining on. '''
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
    a = SBA.encode_periodic(0.5, 1, 10, 20)
    print(a.subsample(2))
    # a = SBA(1, 2, 3, 4)
    # b = SBA(3, 4, 5, 6)
    # r = SBA.or_bits(a, b)
    # r.shift(2)
    # q = r.cp()
    # print(q)
    # print(r.shift(2))
    # r = SBA.and_bits(a, b)
    # a.print_SBA()
    # b = SBA(a)
    # print(b)
    # res = SBA.and_size(a, b)
    # print(res)
