PYTHON=python3

clean:
	rm -rf build dist src/py/__pycache__ sparse_bit_arrays* src/py/c-build/*.so

install:
	$(PYTHON) -m pip install .

test:
	$(PYTHON) tests/tests.py

# for running the repo without installing it on the system
build-local:
	gcc -o src/py/c-build/sba_lib.so -Isrc/c src/c/sba.c -fPIC -shared