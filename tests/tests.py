#!/usr/bin/env python3

import ctypes as c
import unittest
import numpy as np
import argparse

class TestSBA(unittest.TestCase):
    def test_instantiation(self):
        a = SBA()
        self.assertEqual(a, [])
        a = SBA((4, 3, 2, 1))
        self.assertEqual(a, [4, 3, 2, 1])
        b = SBA(a)
        self.assertEqual(b, (4, 3, 2, 1))
        with self.assertRaises(SBAException):
            a = SBA(0, 3, 2, 1)
        with self.assertRaises(SBAException):
            a = SBA(2, 4, 6.2, 8)
        with self.assertRaises(SBAException):
            a = SBA(-1, -2, -3, -4)
        with self.assertRaises(SBAException):
            a = SBA(1, 1, 1, 1)
        with self.assertRaises(SBAException):
            a = SBA(0xFFFFFFFF + 1, 3, 2, 0)
        a = SBA(blank_cap = 5)
        self.assertEqual(a.capacity, 5)
    
    def test_set_bit(self):
        a = SBA(4, 3, 2, 1)
        a.set_bit(0, True)
        self.assertEqual(a, [4, 3, 2, 1, 0])
        a.set_bit(1, True)
        self.assertEqual(a, [4, 3, 2, 1, 0])
        a.set_bit(3, False)
        self.assertEqual(a, [4, 2, 1, 0])
        a.set_bit(5, False)
        self.assertEqual(a, [4, 2, 1, 0])
        with self.assertRaises(SBAException):
            a.set_bit(-1, True)
    
    def test_add_sub(self):
        a = SBA(2, 1)
        self.assertEqual(a + 5, [5, 2, 1])
        self.assertNotEqual(a, [5, 2, 1])
        self.assertEqual(a + [4, 3], [4, 3, 2, 1])
        self.assertEqual(a + " str " + a, "[2, 1] str [2, 1]")
        b = SBA(3, 2)
        self.assertEqual(a + b, [3, 2, 1])

        self.assertEqual(a - 2, [1])
        self.assertNotEqual(a, [1])
        self.assertEqual(a - [1, 2], [])
        self.assertEqual(a - b, [1])
        
    def test_mul(self):
        a = SBA(2, 1)
        b = SBA(3, 2)
        self.assertEqual(a * b, [2])
        self.assertEqual(a * 1, True)
        a = SBA(10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
        for i in range(500):
            b = a * 0.5
            ln = len(b)
            self.assertTrue(ln >= 0 and ln <= len(a) and all(b[i] in a for i in range(ln)))
        self.assertEqual(a, [10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    
    def test_slicing(self):
        a = SBA(250, 200, 50, 10, 2, 1)
        self.assertEqual(a[:], a)
        self.assertEqual(a[10:11], [10])
        self.assertEqual(a[0:3], [2, 1])
        self.assertEqual(a[50:300], [250, 200, 50])
        self.assertEqual(a[10:0], a[0:10])
        self.assertEqual(a[:200], [250, 200])
        self.assertEqual(a[3:], [2, 1])
    
    def test_ops(self):
        a = SBA(2, 1)
        b = SBA(3, 2)
        self.assertEqual(SBA.and_bits(a, b), [2])
        self.assertEqual(SBA.and_size(a, b), 1)
        self.assertEqual(SBA.or_bits(a, b), [3, 2, 1])
        self.assertEqual(SBA.or_size(a, b), 3)
        self.assertEqual(SBA.xor_bits(a, b), [3, 1])
        self.assertEqual(SBA.xor_size(a, b), 2)
    
    def test_shift(self):
        a = SBA(2, 1)
        self.assertEqual(a << 1, [3, 2])
        self.assertEqual(a << -1, [1, 0])
        self.assertEqual(a >> 2, [0])
    
    def test_encoding(self):
        a = SBA.encode_linear(0.5, 3, 100)
        self.assertEqual(a, [51, 50, 49])
        a = SBA.encode_linear(1.0, 2, 10)
        self.assertEqual(a, [9, 8])
        a = SBA.encode_periodic(0, 1, 2, 10)
        self.assertEqual(a, [1, 0])
        a = SBA.encode_periodic(-1, 1, 2, 10)
        self.assertEqual(a, [1, 0])
    
    def test_large_array(self):
        a = SBA()
        while a.size < 100000:
            a.set_bit(a.size, True)
        a.set_bit(0xFFFFFFFF, True)
        self.assertEqual(a[0], 0xFFFFFFFF)
        a.set_bit(0, False)
        self.assertEqual(a[-1], 1)
        while a.size > 0:
            del a[-1]
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Do tests on SBA module.')
    parser.add_argument("-l", "--local", action="store_true", help="Prioritize importing from this repo, " +
            "and not the system installed package with the same name.")
    args = parser.parse_args()
    relative_import = args.local
    if relative_import:
        import pathlib, sys
        sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src" / "py"))
    from sba import *

    SBA._init_lib_if_needed() # give lib import error rather than failed tests
    unittest.main()

