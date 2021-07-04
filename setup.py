from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
import numpy

class compiler_specific_args_build_ext(build_ext):
    RELEVANT_EXTENSIONS = ['sba']
    def build_extensions(self):
        compiler = self.compiler.compiler_type
        for ext in self.extensions:
            if ext.name in self.RELEVANT_EXTENSIONS:
                if compiler == "msvc":
                    ext.extra_compile_args = ["/openmp"]
                elif compiler == "gcc":
                    ext.extra_compile_args = "-fopenmp"
                    ext.extra_link_args = "-fopenmp"
                else:
                    raise Exception("unknown compiler: " + compiler)
        build_ext.build_extensions(self)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    cmdclass = { 'build_ext': compiler_specific_args_build_ext },
    ext_modules = cythonize(
        [
            Extension(
                name = "sba",
                sources = ["sba.pyx"],
            )
        ],
        annotate = True,
        language_level=3,
    ),
    data_files = [('sba', ['py.typed', 'sba.pyi', 'sba.pxd']),],
    include_dirs=[numpy.get_include()],
    name = "sparse-bit-array",
    version = "1.1.1",
    author = "John Giorshev",
    description = "sparse bit array",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/jagprog5/sba",
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires = ">=3.6",
)

