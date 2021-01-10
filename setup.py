from setuptools import setup, Extension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sparse-bit-arrays-jagprog5-TEST", # sparse-bit-arrays
    version="0.0.1",
    author="John Giorshev",
    description="sparse bit arrays",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jagprog5/sba",
    packages=['sba'],
    package_dir={'sba': 'src/py'},
    ext_modules = [Extension(name = "sba.c-build.sba_lib", # keep same relative dir compared to sba.py
                            sources = ["src/c/sba.c"])],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
