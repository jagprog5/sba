# Sparse Bit Arrays

A sparse bit array stores the indices of the bits that are ON for an arbitrarily lengthed array.

This package is a wrapper for a C backend.

```python
from sba import *

# an array with the bits 0, 3, and 40 set to ON
b = SBA(0, 3, 40)

b.set_bit(3, False) # [0, 40]

# Makes a COPY of b, turn on bits 2 and 3, then places the result in c. b is NOT modified
c = b + [3, 2]

xored = a ^ b # [2, 3]

# Randomly flips bits OFF, where each bit has an 80% chance of remaining ON
c *= 0.8

# encode the value 0.5 by turning 3 bits ON in an array with the total size of 100
d = SBA.encode_linear(0.5, 3, 100) # [49, 50, 51]
d = SBA.encode_linear(0, 3, 100) # [0, 1, 2]
d = SBA.encode_linear(1.0, 3, 100) # [97, 98, 99]
```

A bunch of operators are defined (+, -, *, &, |, <<, [i], etc.), except for the complement operator, since it doesn't make sense in the context of SBAs.