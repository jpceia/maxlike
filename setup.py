import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize


__version__ = '2.3.3'
__package__ = 'maxlike'


with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("README.md") as f:
    __doc__ = f.read()


setup(
    name=__package__,
    version=__version__,
    author='joao ceia',
    author_email='joao.p.ceia@gmail.com',
    packages=['maxlike', 'maxlike.tensor', 'maxlike.func', 'maxlike.analytics'],
    url='https://github.com/jpceia/maxlike',
    license='',
    description="Python package to model count data",
    long_description=__doc__,
    long_description_content_type="text/markdown",
    install_requires=required,
    include_dirs = [np.get_include()],
    ext_modules=cythonize("maxlike/tensor/ctensor.pyx")
)
