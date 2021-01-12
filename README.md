# Sparse Bit Arrays

A sparse bit array stores the indices of the bits that are ON for an arbitrarily long array.

This package is a C lib wrapper.

## Overview

```python
from sba import *

>>> a = SBA(0, 3, 40) # an array with the bits 0, 3, and 40 set to ON
>>> a
[0, 3, 40]
>>> a.set_bit(3, False)
>>> a
[0, 40]

>>> arr = a.to_np() # Conversion to and from numpy arrays
>>> arr
array([ 0,  3, 40], dtype=uint32)

>>> b = a + [3, 2] # overloaded operators
>>> b
[0, 2, 3, 40]
>>> a + (b >> 2)
[0, 1, 38, 40]

# Randomly flips bits OFF, where each bit has a 33% chance of remaining ON
>>> SBA(1, 2, 3, 4, 5, 6) * (1 / 3)
[2, 5]

# encodes a float value by turning 3 bits ON in an array with a total size of 100
>>> SBA.encode_linear(0.0, 3, 100)
[0, 1, 2]
>>> SBA.encode_linear(0.5, 3, 100)
[49, 50, 51]
>>> SBA.encode_linear(1.0, 3, 100)
[97, 98, 99]
```

## Installation: pypi

The package is a source distribution, not a binary distribution. Windows users may need to do [this](https://docs.microsoft.com/en-us/answers/questions/136595/error-microsoft-visual-c-140-or-greater-is-require.html).


```bash
pip install sparse-bit-arrays
```
## Installation: GitHub
```bash
git clone https://github.com/jagprog5/sba.git
cd sba
make install
```
The installation can then be tested with:
```bash
make test
```