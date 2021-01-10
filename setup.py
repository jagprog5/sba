import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparse-bit-arrays-jagprog5-TEST", # sparse-bit-arrays
    version="0.0.1",
    author="John Giorshev",
    description="sparse bit array",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jagprog5/sba",
    packages=['sba'],
    package_dir={'sba': 'src/py'},
    package_data={'sba': ['lib/sba_lib.so']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
