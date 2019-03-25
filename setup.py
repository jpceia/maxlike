import os
from setuptools import setup


__version__ = '2.3.1'
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
    description=__doc__,
    install_requires=required
)
