# python setup.py build_ext --inplace ; python.exe .\tests.py

import unittest
from sba import *
from array import array
import numpy as np
from time import time

def get_random_indicies(n=100,m=100):
    ''' at most n elements from m-1 to -m '''
    l = list(dict.fromkeys([SBA.rand_int() % m - (m >> 1) for i in range(n)]))
    l.sort(reverse=True)
    return l

class TestSBA(unittest.TestCase):
    def test_overloaded_ops(self):
        a = SBA([6, 4, 2])
        self.assertEqual(a + 3, [6, 4, 3, 2])
        self.assertEqual(a - 6, [4, 2])
        self.assertEqual(a * 6, True)
        self.assertEqual(a.get(6), True)
        self.assertTrue(4 in a)
        self.assertEqual(a - 6, [4, 2])
        self.assertEqual(a - SBA([6, 2]), [4])
        self.assertEqual(a & SBA([7, 6, 2]), [6, 2])
        self.assertEqual(a | SBA([7, 6, 2]), [7, 6, 4, 2])
        self.assertEqual(a ^ SBA([7, 6, 2]), [7, 4])

    def test_instantiation(self):
        with self.assertRaises(SBAException):
            SBA([1, 2, 3])
        with self.assertRaises(SBAException):
            SBA([1, 1, 1])
        with self.assertRaises(SBAException):
            SBA(['hi', 2, 1])
        self.assertEqual(SBA(5), [4, 3, 2, 1, 0])
        self.assertEqual(SBA(5, 2), [5, 4, 3, 2])
    
    def test_from_dense(self):
        a = memoryview(array('h', [0, 0, 0, 2, 0, -1]))
        self.assertEqual(SBA.from_dense(a), [2, 0])
        self.assertEqual(SBA.from_dense(a, True), [5, 3])
        self.assertEqual(SBA.from_dense(a, filter = lambda x: x > 0), [2])

    def test_large_array(self):
        a = SBA()
        for i in range(100000):
            a.set(i, True)
        self.assertEqual(len(a), 100000)
        self.assertEqual(a[0], 99999)
        for i in range(100000):
            a.set(i, False)
        self.assertEqual(len(a), 0)

    def test_from_capacity(self):
        a = SBA.from_capacity(5)
        self.assertEqual(a, [4, 3, 2, 1, 0])

    def test_from_np(self):
        arr = np.array([5, 2, 0], np.intc)
        a = SBA.from_np(arr, deep_copy=True)
        a[0] = 0
        self.assertEqual(a, [2, 0])
        self.assertTrue((arr == [5, 2, 0]).all())
        a = SBA.from_np(arr, deep_copy=False)
        arr = None
        self.assertEqual(a, [5, 2, 0])
        with self.assertRaises(SBAException): # not owner
            a[0] = 0

    def test_to_np(self):
        a = SBA([5, 2, 1])
        arr = a.to_np(give_ownership = False)
        self.assertTrue((arr == [5, 2, 1]).all())
        with self.assertRaises(ValueError): # array is read-only
            arr[1] = 2
        with self.assertRaises(SBAException): # can't modify buffer while being viewed
            a[-1] = 0
        arr = None
        a[-1] = 0
        self.assertEqual(a, [5, 2, 0])

        arr = a.to_np(give_ownership = True)
        self.assertTrue(a == []) # cleared reference to data
        self.assertTrue((arr == [5, 2, 0]).all())
        arr[0] = 0
        self.assertTrue((arr == [0, 2, 0]).all())

    def test_set(self):
        a = SBA([5, 2, 1])
        a[1] = 8
        self.assertEqual(a, [8, 5, 1])
        a[-1] = 0
        self.assertEqual(a, [8, 5, 0])
        with self.assertRaises(SBAException):
            a[10] = 0

    def test_get(self):
        a = SBA([i for i in range(100, 0, -2)])
        self.assertEqual(a[0], 100)
        self.assertEqual(a[-1], 2)
        self.assertEqual(a[6:], [6, 4, 2])
        self.assertEqual(a[:], a)
        self.assertEqual(a[:96], [100, 98, 96])

    def test_shift(self):
        a = SBA([5, 2, 1])
        a_cp = SBA(a)
        self.assertEqual(a << 2, [7, 4, 3])
        self.assertEqual(a, a_cp)
        self.assertEqual(a.shift(-4), [1, -2, -3])
        self.assertNotEqual(a, a_cp)

    def test_subsample(self):
        a = SBA(get_random_indicies())
        a_cp = a.cp()
        a.subsample(0.5)
        self.assertTrue(len(a) <= len(a_cp))
        self.assertTrue(all([i in a_cp for i in a]))
        self.assertEqual(a.subsample(0), [])

    def test_encoding(self):
        self.assertEqual(SBA.encode(0.5, 3, 100), [51, 50, 49])
        self.assertEqual(SBA.encode(1.0, 2, 10), [9, 8])
        self.assertEqual(SBA.encode(0, 2, 10, period=1), [1, 0])
        self.assertEqual(SBA.encode(-1, 2, 10, period=1), [1, 0])

    def test_comparison(self):
        self.assertTrue(SBA([5, 2, 1]) >= SBA([5, 2]))
        self.assertTrue(SBA([5, 2, 1]) >= SBA([5, 2, 1]))
        self.assertTrue(not SBA([5, 2, 1]) > SBA([5, 2, 1]))
        self.assertTrue(SBA([5, 2, 1]) >= SBA([5, 2]))
        self.assertTrue(SBA([9, 1]) == SBA([9, 1]))
        self.assertTrue(SBA([9]) != SBA([9, 1]))
        self.assertTrue(SBA([9, 1]) == [1, 9])
        self.assertTrue(SBA([9, 1]) != [" ", 9])
        self.assertTrue(not SBA([9, 1]) == [" ", 9])

    def test_bin_ops(self):
        a = SBA([3, 2, 1])
        b = SBA([2, 1, 0])
        self.assertEqual(a & b, [2, 1])
        self.assertEqual(len(a & b), SBA.andl(a, b))
        self.assertEqual(a | b, [3, 2, 1, 0])
        self.assertEqual(len(a | b), SBA.orl(a, b))
        self.assertEqual(a ^ b, [3, 0])
        self.assertEqual(len(a ^ b), SBA.xorl(a, b))
        self.assertEqual(a & [5, 7, 1], [False, False, True])
    
    def test_inplace_ops(self):
        a = SBA([3, 2, 1])
        b = SBA([2, 1, 0])
        self.assertEqual(b.andi(a), [2, 1])

    def test_and(self):
        a = SBA(get_random_indicies())
        b = SBA(get_random_indicies())
        c = SBA.andb(a, b)
        self.assertTrue(len(c) == SBA.andl(a, b))
        for i in c:
            self.assertTrue(i in a and i in b)
            a.set(i, False)
            b.set(i, False)
        for i in a:
            for j in b:
                self.assertTrue(i != j)
    
    def test_or(self):
        a = SBA(get_random_indicies())
        b = SBA(get_random_indicies())
        c = SBA.orb(a, b)
        self.assertTrue(len(c) == SBA.orl(a, b))
        for i in c:
            if i in a:
                a.set(i, False)
                if i in b:
                    b.set(i, False)
            elif i in b:
                b.set(i, False)
            else:
                self.assertTrue(False)
        self.assertTrue(len(a) == 0)
        self.assertTrue(len(b) == 0)
    
    def test_xor(self):
        a = SBA(get_random_indicies())
        b = SBA(get_random_indicies())
        c = SBA.xorb(a, b)
        self.assertTrue(len(c) == SBA.xorl(a, b))
        for i in c:
            if i in a:
                a.set(i, False)
                if i in b:
                    self.assertTrue(False)
            elif i in b:
                b.set(i, False)
            else:
                self.assertTrue(False)
        for i in a:
            self.assertTrue(i in b)
    
    def test_rm(self):
        a = SBA(get_random_indicies())
        a_cp = a.cp()
        rm = SBA(get_random_indicies())
        rm_cp = rm.cp()
        a.rm(rm)
        self.assertTrue(rm == rm_cp)
        for i in rm:
            self.assertTrue(not i in a)
        for i in SBA.andb(a, a_cp):
            self.assertTrue(not i in rm)

def speed_test():
    start = time()
    sub = 0
    for i in range(10000):
        substart = time()
        a = SBA(get_random_indicies(1000, 1000))
        b = SBA(get_random_indicies(1000, 1000))
        substop = time()
        sub += substop - substart
        SBA.andb(a, b)
        SBA.andl(a, b)
        SBA.orb(a, b)
        SBA.orl(a, b)
        SBA.xorb(a, b)
        SBA.xorl(a, b)
        a.cp().rm(b)
    stop = time()
    print(stop - start - sub)

if __name__ == "__main__":
    SBA.seed_rand()
    unittest.main()
    # speed_test()
