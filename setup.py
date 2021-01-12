from setuptools import setup, Extension
from pathlib import Path

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sparse-bit-arrays",
    version="0.0.3",
    author="John Giorshev",
    description="sparse bit arrays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jagprog5/sba",
    packages=['sba'],
    package_dir={'sba': 'src/py'},
    # setuptools is complete and utter garbage, f****** piece of s***
    # how hard can it be to ensure a header file is added to sdist?
    package_data={'sba': [str(Path(__file__).absolute().parents[0] / "src" / "c" / "sba.h")]},
    ext_modules = [Extension(name = "sba.c-build.sba_lib",
                            sources = ["src/c/sba.c"],
                            include_dirs = ["src/c"])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
