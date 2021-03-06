# python setup.py build_ext --inplace ; python.exe .\tests.py

import unittest
from sba import *
from array import array
import numpy as np
from timeit import default_timer as timer

def get_random_indices(n=100,m=100):
    ''' at most n elements from m-1 to -m '''
    l = list(dict.fromkeys([SBA.rand_int() % m - (m >> 1) for i in range(n)]))
    l.sort()
    return l

class Test(unittest.TestCase):
    def test_overloaded_ops(self):
        a = SBA.iterable([2, 4, 6])
        self.assertEqual(a + 3, [6, 4, 3, 2])
        self.assertEqual(a - 6, [4, 2])
        self.assertEqual(a * 6, True)
        self.assertEqual(a.get(6), True)
        self.assertTrue(4 in a)
        self.assertEqual(a - 6, [4, 2])
        self.assertEqual(a - SBA.iterable([2, 6]), [4])
        self.assertEqual(a & SBA.iterable([2, 6]), [2, 6])
        self.assertEqual(a | SBA.iterable([2, 7]), [2, 4, 6, 7])
        self.assertEqual(a ^ SBA.iterable([2, 7]), [4, 6, 7])

    def test_instantiation(self):
        with self.assertRaises(SBAException):
            SBA.iterable([3, 2, 1])
        with self.assertRaises(SBAException):
            SBA.iterable([1, 1, 1])
        with self.assertRaises(TypeError):
            SBA.iterable(['hi', 2, 1])
        with self.assertRaises(SBAException):
            SBA.iterable([0xFFFFFFFF + 1])
        self.assertEqual(SBA.length(5), [0, 1, 2, 3, 4])
        self.assertEqual(SBA.range(2, 5), [2, 3, 4, 5])

    def test_large_array(self):
        a = SBA()
        for i in range(100000):
            a.set(i, True)
        self.assertEqual(len(a), 100000)
        self.assertEqual(a[-1], 99999)
        for i in range(100000):
            a.set(i, False)
        self.assertEqual(len(a), 0)

    def test_buffer(self):
        arr = np.array([0, 2, 5], np.intc)
        a = SBA.buffer(arr, copy=True)
        a[0] = 5
        self.assertEqual(a, [2, 5])
        self.assertTrue((arr == [0, 2, 5]).all())
        a = SBA.buffer(arr, copy=False)
        arr = None
        self.assertEqual(a, [0, 2, 5])
        with self.assertRaises(SBAException): # not owner
            a[0] = 0

        arr = np.array([0, 2, 5], np.short)
        with self.assertRaises(SBAException):
            a = SBA.buffer(arr, copy=False)
        
        self.assertEqual(SBA.iterable([1, 0, 0, 1, 1], filter = lambda x : x != 0), [0, 3, 4])

    def test_to_np(self):
        a = SBA.iterable([1, 2, 5])
        arr = a.to_np(give_ownership = False)
        self.assertTrue((arr == [1, 2, 5]).all())
        with self.assertRaises(ValueError): # array is read-only
            arr[1] = 2
        with self.assertRaises(SBAException): # can't modify buffer while being viewed
            a[-1] = 0
        arr = None
        a[-1] = 0
        self.assertEqual(a, [0, 1, 2])

        arr = a.to_np(give_ownership = True)
        self.assertTrue(a == []) # cleared reference to data
        self.assertTrue((arr == [0, 1, 2]).all())
        arr[2] = 0
        self.assertTrue((arr == [0, 1, 0]).all())

    def test_set(self):
        a = SBA.iterable([1, 2, 5])
        a[1] = 8
        self.assertEqual(a, [1, 5, 8])
        a[0] = 0
        self.assertEqual(a, [0, 5, 8])
        with self.assertRaises(SBAException):
            a[10] = 0

    def test_get(self):
        a = SBA.iterable([i for i in range(0, 100, 2)])
        self.assertEqual(a[0], 0)
        self.assertEqual(a[-1], 98)
        self.assertEqual(a[:6], [6, 4, 2, 0])
        self.assertEqual(a[:], a)
        self.assertEqual(a[96:], [98, 96])

    def test_shift(self):
        a = SBA.iterable([1, 2, 5])
        a_cp = SBA.iterable(a)
        self.assertEqual(a << 2, [7, 4, 3])
        self.assertEqual(a, a_cp)
        a.shift(-4)
        self.assertEqual(a, [1, -2, -3])
        self.assertNotEqual(a, a_cp)

    def test_subsample(self):
        a = SBA.iterable(get_random_indices())
        a_cp = a.cp()
        a.subsample(0.5)
        self.assertTrue(len(a) <= len(a_cp))
        self.assertTrue(all([i in a_cp for i in a]))
        a_cp.subsample(0)
        self.assertTrue(len(a_cp) == 0)

    def test_encoding(self):
        self.assertEqual(SBA.encode(0.5, 3, 100), [51, 50, 49])
        self.assertEqual(SBA.encode(1.0, 2, 10), [9, 8])
        self.assertEqual(SBA.encode(0, 2, 10, period=1), [1, 0])
        self.assertEqual(SBA.encode(-1, 2, 10, period=1), [1, 0])

    def test_comparison(self):
        self.assertTrue(SBA.iterable([1, 2, 5]) >= SBA.iterable([2, 5]))
        self.assertTrue(SBA.iterable([1, 2, 5]) >= SBA.iterable([1, 2, 5]))
        self.assertTrue(not SBA.iterable([1, 2, 5]) > SBA.iterable([1, 2, 5]))
        self.assertTrue(SBA.iterable([1, 2, 5]) >= SBA.iterable([2, 5]))
        self.assertTrue(SBA.iterable([1, 9]) == SBA.iterable([1, 9]))
        self.assertTrue(SBA.iterable([9]) != SBA.iterable([1, 9]))
        self.assertTrue(SBA.iterable([1, 9]) == [1, 9])

    def test_bin_ops(self):
        a = SBA.iterable([1, 2, 3])
        b = SBA.iterable([0, 1, 2])
        self.assertEqual(a & b, [2, 1])
        self.assertEqual(len(a & b), SBA.andl(a, b))
        self.assertEqual(a | b, [1, 2, 3, 0])
        self.assertEqual(len(a | b), a.orl(b))
        self.assertEqual(a ^ b, [3, 0])
        self.assertEqual(len(a ^ b), SBA.xorl(a, b))
    
    def test_par(self):
        arr = np.array([SBA.length(i) for i in range(3)], dtype='object')           
        a = SBA.length(3)
        self.assertTrue((SBA.andp(a, arr) == [0, 1, 2]).all())
    
    def test_inplace_ops(self):
        a = SBA.iterable([1, 2, 3])
        b = SBA.iterable([0, 1, 2])
        self.assertEqual(b.andi(a), [2, 1])

    def test_and(self):
        a = SBA.iterable(get_random_indices())
        b = SBA.iterable(get_random_indices())
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
        a = SBA.iterable(get_random_indices())
        b = SBA.iterable(get_random_indices())
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
        a = SBA.iterable(get_random_indices())
        b = SBA.iterable(get_random_indices())
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
        a = SBA.iterable(get_random_indices())
        a_cp = a.cp()
        rm = SBA.iterable(get_random_indices())
        rm_cp = rm.cp()
        a.rm(rm)
        self.assertTrue(rm == rm_cp)
        for i in rm:
            self.assertTrue(not i in a)
        for i in SBA.andb(a, a_cp):
            self.assertTrue(not i in rm)

def speed_test():
    a = SBA.iterable(range(0, 100000, 3))
    b = SBA.iterable(range(0, 100000, 5))
    start = timer()
    SBA.andb(a, b)
    SBA.andl(a, b)
    SBA.orb(a, b)
    SBA.orl(a, b)
    SBA.xorb(a, b)
    SBA.xorl(a, b)
    stop = timer()
    print(round((stop - start) * 100000))

if __name__ == "__main__":
    SBA.seed_rand()
    unittest.main()
    # speed_test()
