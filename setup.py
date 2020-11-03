from distutils.core import setup

setup(name='py_vollib_vectorized',
      version='0.1',
      description='A vectorized approach to calculating Implied Volatility and Greeks',
      author='Marc Demers',
      author_email='demers.marc@gmail.com',
      # url='https://www.python.org/sigs/distutils-sig/',
      packages=['py_vollib', 'numba'],
      )
