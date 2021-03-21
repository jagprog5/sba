from sba import SBA
import numpy as np



# a = np.arange(10, dtype=np.uintc)
# a[0] = 0xFFFFFFFF + 1
# print(a)
# SBA.from_buffer(a)

# c = np.arange(1, dtype=np.uint)

# b = np.frombuffer(a, dtype=np.uint)

# print(np.append(a, c))
# print(a)
# print(b)

# narr = np.arange(10, dtype=np.uint)
# narr.flags.writeable = False
# barr = np.frombuffer(narr, dtype=np.uint)

# barr[0] = 1
# print(narr)
# print(barr)

# barr = narr

# barr[0] = 5
# print(narr)
# print(barr)

# i = [1, 2, 3, 4]
# a = np.array(i)
# i[0] = 5
# print(a)

# a = SBA.from_iterable([15, 2, 0])
# print(a[0])
# arr = np.frombuffer(a, dtype=np.uint)
# b = SBA.from_np(arr)
# print(b)

# a = SBA.from_capacity(6)
# print(a.to_np().flags)
# print(np.frombuffer(a, dtype=np.uint))


# narr = np.arange(10, dtype=np.dtype("I"))
# SBA.from_np(narr)
# a = SBA(narr)
# cdef int [:] narr_view = narr
# sba.test_buffer(narr)

# a = np.ndarray([1, 2, 3, 4], dtype=np.uint)
# SBA.test_buffer(a)
# a = SBA(5)
# a = SBA([5, 2, 1])
# print(memoryview(a).tolist())
# print(memoryview(a)[-1])
# print(memoryview(a)[0])
# print(np.asarray(a))
# print(a.to_np())
# a.add_row()