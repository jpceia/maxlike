import os
from setuptools import setup


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
    python_requires=">=3.9",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13'
    ]
)
