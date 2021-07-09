# Sparse Bit Array

A sparse bit array stores the indices of the bits that are ON. Made in cython.

## Overview

```python
>>> from sba import *
>>> z = SBA.iterable([0, 3, 40]) # an SBA with the bits at index 0, 3, and 40 set to ON
>>> z
[0, 3, 40]
>>> z.set(3, False)
>>> z
[0 40]


>>> y = SBA.iterable([0, 3, 40, 50])
>>> z & y # overloaded ops
[0 40]
>>> z.andl(y) # The number of ON bits shared between z AND y
2


>>> y.to_np() # numpy ndarray conversion
array([ 0,  3, 50], dtype=int32)

>>> memoryview(y)[0] # implements buffer protocol
0

>>> # Randomly flips bits OFF, where each bit has a 33% chance of remaining ON
>>> SBA.length(6) * (1 / 3)
[2 5]

>>> # encodes a float value by turning 3 bits ON in an array with a total size of 100
>>> SBA.encode(0.0, 3, 100)
[0 1 2]
>>> SBA.encode(0.5, 3, 100)
[49 50 51]
>>> SBA.encode(1.0, 3, 100)
[97 98 99]
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
