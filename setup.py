from distutils.core import setup

setup(
    name='maxlike',
    version='2.0',
    packages=['maxlike'],
    install_requires = ['numpy'],
    url='https://github.com/jpceia/maxlike',
    license='',
    author='jpceia',
    author_email='joao.p.ceia@gmail.com',
    description="""
        Python module to fit statistical models to observed data through
        maximum likelihood estimation."""
)
