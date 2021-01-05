import ctypes
from sba import SBA
import random
# from typing import List, Iterable, Optional


class SBAMapperOutput(ctypes.Structure):
    _fields_ = [
        ('numConnections',ctypes.c_uint64),
        ('inputBits',ctypes.POINTER(ctypes.c_uint64)),
        ('strengths',ctypes.POINTER(ctypes.c_uint8))]

class SBAMapper(ctypes.Structure):
    _fields_ = [
        ('numOutputs',ctypes.c_uint64),
        ('outputs',ctypes.POINTER(SBAMapperOutput)),
        ('numActiveOutputs',ctypes.c_uint64),
        ('connectionStrengthThreshold',ctypes.c_uint8),
        ('connectionStrengthDelta',ctypes.c_uint8)]
    
    mapper_init = False

    def _init_lib_if_needed():
        if not SBAMapper.mapper_init:
            SBA._init_lib_if_needed()
            SBAMapper.mapper_init = True

            # printSBAMapper
            SBAMapper.printSBAMapper = SBA.ml_lib.printSBAMapper
            SBAMapper.printSBAMapper.restype = None
            SBAMapper.printSBAMapper.argtype = [ctypes.POINTER(SBAMapper)]

            # doMapper
            SBAMapper.doMapper = SBA.ml_lib.doMapper
            SBAMapper.doMapper.restype = None
            SBAMapper.doMapper.argtype = [ctypes.POINTER(SBA), ctypes.POINTER(SBAMapper), ctypes.POINTER(SBA)]

    def __init__(self, numInputs: int,
            numOutputs: int,
            connectionLikelihood: float = 0.8,
            numActiveOutputs: int = None,
            connectionStrengthThreshold: int = 127,
            connectionStrengthDelta: int = 1):
        if numActiveOutputs is None:
            numActiveOutputs = int(numOutputs * 0.02)
        SBAMapper._init_lib_if_needed()
        self.numOutputs = (ctypes.c_uint64)(numOutputs)
        outputs = []
        for i in range(numOutputs):
            mo = SBAMapperOutput()
            inputBits = []
            strengths = []
            for j in range(numInputs):
                if random.uniform(0, 1) < connectionLikelihood:
                    inputBits.append(j)
                    strengths.append(random.randrange(256))
            ln = len(inputBits)
            mo.numConnections = len(inputBits)
            mo.inputBits = (ctypes.c_uint64 * ln)(*inputBits)
            mo.strengths = (ctypes.c_uint8 * ln)(*strengths)
            outputs.append(mo)
        self.outputs = (SBAMapperOutput * numOutputs)(*outputs)
        self.numActiveOutputs = (ctypes.c_uint64)(numActiveOutputs)
        self.connectionStrengthThreshold = (ctypes.c_uint8)(connectionStrengthThreshold)
        self.connectionStrengthDelta = (ctypes.c_uint8)(connectionStrengthDelta)
    
    def print_SBAMapper(self):
        SBAMapper.printSBAMapper(ctypes.pointer(self))
    
    def map(self, inputs: SBA) -> SBA:
        r = SBA(blank_size = self.numActiveOutputs)
        SBAMapper.doMapper(ctypes.pointer(r), ctypes.pointer(self), ctypes.pointer(inputs))
        return r

        

if __name__ == "__main__":
    m = SBAMapper(3, 2, 0.5, 2, 0, 1)
    m.print_SBAMapper()
    i = SBA(0, 1, 2)
    print(m.map(i))
    m.print_SBAMapper()