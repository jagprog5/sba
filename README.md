# Sparse Bit Array

A sparse bit array stores the indices of the bits that are ON. Made in cython.

A python stub is provided for type hinting and autocomplete. Make sure to use pylance if you're using vscode.

## Overview

```python
>>> from sba import *
>>> a = SBA.from_iterable([0, 3, 40]) # an array with the bits at index 0, 3, and 40 set to ON
>>> a
[0, 3, 40]
>>> a.set(3, False)
>>> a
[0 40]


>>> b = SBA.from_iterable([0, 3, 50])
>>> a & b # overloaded ops
[0 3]

>>> b.to_buffer() # convert to or from numpy ndarray
array([ 0,  3, 50], dtype=int32)

>>> memoryview(b)[0] # implements buffer protocol
0

>>> # Randomly flips bits OFF, where each bit has a 33% chance of remaining ON
>>> SBA([1, 2, 3, 4, 5, 6]) * (1 / 3)
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
