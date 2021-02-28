import io
import os
from distutils.core import setup

from setuptools import find_packages

dir = os.path.dirname(__file__)

with io.open(os.path.join(dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='py_vollib_vectorized',
      version='0.1.1',
      description='A fast, vectorized approach to calculating Implied Volatility and Greeks using the Black, Black-Scholes and Black-Scholes-Merton pricing.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/marcdemers/py_vollib_vectorized',
      author='Marc Demers',
      author_email='demers.marc@gmail.com',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Financial and Insurance Industry',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Topic :: Software Development :: Libraries',
          'Programming Language :: Python :: 3.3',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: Implementation',
          'License :: OSI Approved :: MIT License',
          'Topic :: Office/Business :: Financial',
          'Topic :: Office/Business :: Financial :: Investment',
      ],
      install_requires=['py_vollib>=1.0.1', 'numba>=0.51.0', 'py_lets_be_rational', 'numpy', 'pandas', 'scipy'],
      packages=find_packages()
      )
