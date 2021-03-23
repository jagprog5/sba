from setuptools import setup
from Cython.Build import cythonize
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    ext_modules = cythonize("sba.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
    name = "sparse-bit-array",
    version = "1.0.0",
    author = "John Giorshev",
    description = "sparse bit array",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/jagprog5/sba",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires = ">=3.6"
)

 