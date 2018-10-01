from distutils.core import setup

setup(
    name='maxlike',
    version='2.1',
    author='joao ceia',
    author_email='joao.p.ceia@gmail.com',
    packages=['maxlike'],
    url='https://github.com/jpceia/maxlike',
    license='',
    description="""
        Python module to fit statistical models to observed data through
        maximum likelihood estimation. """,
    requires=[
        "Numpy (>= 1.14.0)",
        "Pandas (>= 0.23.0)",
        "Scipy (>= 1.1.0)",
        "Six (>= 1.11.0)",
    ],
)
