# Automatic promotion from void*

```python
cdef int* thing
thing = PyMem_Malloc(5 * sizeof(thing[0]))
```

yields:

```
Cannot assign type 'void *' to 'int *'
```

This can be fixed by adding a cast:

```
cdef int* thing
thing = <int*> PyMem_Malloc(5 * sizeof(thing[0]))
        ^^^^^^
```

However, in c, casting the output from malloc is discouraged. Why does cython necessitate this?