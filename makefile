PYTHON=python3

clean:
	rm -rf build dist src/py/__pycache__ sparse_bit_arrays*

install:
	$(PYTHON) -m pip install .

test:
	$(PYTHON) tests/tests.py
