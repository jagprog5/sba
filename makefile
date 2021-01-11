PYTHON=python3
SHARED_LIBRARY=src/py/c-build/sba_lib.so

install:
	$(PYTHON) -m pip install .

clean:
	rm -rf build dist src/py/__pycache__ sparse_bit_arrays* src/py/c-build/*.so

test: # requires install
	./tests/tests.py

# no need to install on system. Test within repo:
test-local: $(SHARED_LIBRARY)
	./tests/tests.py --local

$(SHARED_LIBRARY):
	gcc -o $(SHARED_LIBRARY) -Isrc/c src/c/sba.c -fPIC -shared