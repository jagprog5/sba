#!/usr/bin/env python3

import ctypes as c
import unittest
import numpy as np

# relative import. Prioritize importing from the repo BEFORE the system installed package
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src" / "py"))
from sba import *

class TestSBA(unittest.TestCase):
    def test_instantiation(self):
        a = SBA()
        self.assertEqual(a, [])
        a = SBA((1, 2, 3, 4))
        self.assertEqual(a, [1, 2, 3, 4])
        b = SBA(a)
        self.assertEqual(b, (1, 2, 3, 4))
        with self.assertRaises(SBAException):
            a = SBA(1, 2, 3, 0)
        with self.assertRaises(SBAException):
            a = SBA(2, 4, 6.2, 8)
        with self.assertRaises(SBAException):
            a = SBA(-4, -3, -2, -1)
        a = SBA(blank_size = 5)
        self.assertEqual(a.capacity, 5)
    
    def test_set_bit(self):
        a = SBA(1, 2, 3, 4)
        a.set_bit(0, True)
        self.assertEqual(a, [0, 1, 2, 3, 4])
        a.set_bit(1, True)
        self.assertEqual(a, [0, 1, 2, 3, 4])
        a.set_bit(3, False)
        self.assertEqual(a, [0, 1, 2, 4])
        a.set_bit(5, False)
        self.assertEqual(a, [0, 1, 2, 4])
        with self.assertRaises(SBAException):
            a.set_bit(-1, True)
    
    def test_add_sub(self):
        a = SBA(1, 2)
        self.assertEqual(a + 5, [1, 2, 5])
        self.assertNotEqual(a, [1, 2, 5])
        self.assertEqual(a + [4, 3], [1, 2, 3, 4])
        self.assertEqual(a + " str " + a, "[1, 2] str [1, 2]")
        b = SBA(2, 3)
        self.assertEqual(a + b, [1, 2, 3])

        self.assertEqual(a - 2, [1])
        self.assertNotEqual(a, [1])
        self.assertEqual(a - [2, 1], [])
        self.assertEqual(a - b, [1])
        
    def test_mul(self):
        a = SBA(1, 2)
        b = SBA(2, 3)
        self.assertEqual(a * b, [2])
        self.assertEqual(a * 1, True)
        a = SBA(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        for i in range(500):
            b = a * 0.5
            ln = len(b)
            self.assertTrue(ln >= 0 and ln <= len(a) and all(b[i] in a for i in range(ln)))
        self.assertEqual(a, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    def test_ops(self):
        a = SBA(1, 2)
        b = SBA(2, 3)
        self.assertEqual(SBA.and_bits(a, b), [2])
        self.assertEqual(SBA.and_size(a, b), 1)
        self.assertEqual(SBA.or_bits(a, b), [1, 2, 3])
        self.assertEqual(SBA.or_size(a, b), 3)
        self.assertEqual(SBA.xor_bits(a, b), [1, 3])
        self.assertEqual(SBA.xor_size(a, b), 2)
    
    def test_shift(self):
        a = SBA(1, 2)
        self.assertEqual(a << 1, [2, 3])
        self.assertEqual(a << -1, [0, 1])
        self.assertEqual(a >> 2, [0])
    
    def test_encoding(self):
        a = SBA.encode_linear(0.5, 3, 100)
        self.assertEqual(a, [49, 50, 51])
        a = SBA.encode_linear(1.0, 2, 10)
        self.assertEqual(a, [8, 9])
        a = SBA.encode_periodic(0, 1, 2, 10)
        self.assertEqual(a, [0, 1])
        a = SBA.encode_periodic(-1, 1, 2, 10)
        self.assertEqual(a, [0, 1])
    
    def test_numpy(self):
        arr = np.array([1, 2, 3, 4])
        with self.assertRaises(SBAException):
            a = SBA.from_np(arr)
        arr = arr.astype(np.uint32)
        a = SBA.from_np(arr, False)
        del a[0]
        self.assertTrue((arr == [2, 3, 4, 4]).all())
        self.assertEqual(a, [2, 3, 4])
        arr = np.array([1, 2, 3, 4], dtype=np.uint32)
        a = SBA.from_np(arr, True)
        del a[0]
        self.assertTrue((arr == [1, 2, 3, 4]).all())
        self.assertEqual(a, [2, 3, 4])

        arr_b = a.to_np(False)
        del a[0]
        self.assertTrue((arr_b == [3, 4, 4]).all())
        self.assertEqual(a, [3, 4])
        arr_b = a.to_np(True)
        del a[0]
        self.assertTrue((arr_b == [3, 4]).all())
        self.assertEqual(a, [4])
    
    def test_large_array(self):
        a = SBA()
        while a.size < 100000:
            a.set_bit(a.size, True)
        a.set_bit(0xFFFFFFFF, True)
        self.assertEqual(a[-1], 0xFFFFFFFF)
        a.set_bit(0, False)
        self.assertEqual(a[0], 1)
        while a.size > 0:
            del a[-1]
        
        
if __name__ == "__main__":
    SBA() # give lib import error rather than failed tests
    unittest.main()

