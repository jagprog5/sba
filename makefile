PYTHON=python3
SHARED_LIBRARY=src/py/c-build/sba_lib.so

cython:
	$(PYTHON) setup.py build_ext --inplace

clean:
	rm -v *.c *.so *.pyd

install:
	$(PYTHON) -m pip install .

#test: # requires install
#	./tests/tests.py

# no need to install on system. Test within repo:
#test-local: $(SHARED_LIBRARY)
#	./tests/tests.py --local