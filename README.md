# Sparse Bit Array

A sparse bit array stores the indices of the bits that are ON. Made in cython.

## Overview

```python
>>> from sba import *
>>> a = SBA([40, 3, 0]) # an array with the bits 40, 3, and 0 set to ON
>>> a
[40 3 0]
>>> a.set_bit(3, False)
>>> a
[40 0]

>>> b = SBA([50, 3, 0])
>>> a & b # overloaded ops
[3 0]

>>> b.to_np() # convert to or from numpy ndarray
array([50,  3,  0], dtype=int32)

>>> memoryview(b)[0] # implements buffer protocol
50

>>> # Randomly flips bits OFF, where each bit has a 33% chance of remaining ON
>>> SBA([6, 5, 4, 3, 2, 1]) * (1 / 3)
[5 2]

>>> # encodes a float value by turning 3 bits ON in an array with a total size of 100
>>> SBA.encode_linear(0.0, 3, 100)
[2 1 0]
>>> SBA.encode_linear(0.5, 3, 100)
[51 50 49]
>>> SBA.encode_linear(1.0, 3, 100)
[99 98 97]
```
## Installation: PyPI
```bash
pip install sparse-bit-array
```
## Installation: GitHub
```bash
git clone https://github.com/jagprog5/sba.git
cd sba
```
Install repo to system:
```bash
python setup.py install
```
Install repo to directory and run tests:
```bash
python setup.py build_ext --inplace
python tests.py
```
