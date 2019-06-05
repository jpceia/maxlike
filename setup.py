import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize


__version__ = '2.3.6'
__package__ = 'maxlike'

here = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(here, "requirements.txt")) as f:
    required = f.read().splitlines()

with open(os.path.join(here, "README.md")) as f:
    __doc__ = f.read()


setup(
    name=__package__,
    version=__version__,
    author='joao ceia',
    author_email='joao.p.ceia@gmail.com',
    packages=['maxlike', 'maxlike.tensor', 'maxlike.func'],
    url='https://github.com/jpceia/maxlike',
    license='nolicense',
    description="Python package to model count data",
    long_description=__doc__,
    long_description_content_type="text/markdown",
    install_requires=required,
    tests_require=["nose", "coverage"],
    test_suite="nose.collector",
    include_dirs = [np.get_include()],
    ext_modules=cythonize(
        os.path.join(here, "maxlike/tensor/ctensor.pyx")),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7'
    ]
)
