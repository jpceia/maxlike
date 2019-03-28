import os
from setuptools import setup
from Cython.Build import cythonize


__version__ = '2.3.1'
__package__ = 'maxlike'


with open("requirements.txt") as f:
    required = f.read().splitlines()


with open("README.md") as f:
    __doc__ = f.read()


def numpy_include():
    import numpy
    return numpy.get_include()



setup(
    name=__package__,
    version=__version__,
    author='joao ceia',
    author_email='joao.p.ceia@gmail.com',
    packages=['maxlike', 'maxlike.tensor', 'maxlike.func', 'maxlike.analytics'],
    url='https://github.com/jpceia/maxlike',
    license='',
    description=__doc__,
    install_requires=required,
    include_dirs = [numpy_include()],
    ext_modules=cythonize("maxlike/tensor/ctensor.pyx")
)
