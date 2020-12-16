# py_vollib_vectorized


## Introduction

The `py_vollib_vectorized` package makes pricing thousands of option contracts and calculating greeks fast and effortless.
It is built on top of the `py_vollib` library.
Upon import, it will automatically patch the corresponding `py_vollib` functions so as to support vectorization.
Inputs can then be passed as `numpy.array`, `pandas.Series` or `pandas.DataFrame`.

On top of vectorization, modifications to py_vollib include additional `numba` speedups; as such, `numba` *is* required.

## Installation

    pip install py_vollib_vectorized
    
## Requirements

* Written for Python 3.5+
* Requires py_vollib, numba, numpy, pandas, scipy

## Code samples

#### Patching `py_vollib`

```python
# The usual py_vollib syntax

import numpy as np
import pandas as pd

from py_vollib.black_scholes import black_scholes
flag = 'c'  # 'c' for call, 'p' for put
S = 100  # Underlying asset price
K = 90  # Strike
t = 0.5  # (Annualized) time-to-expiration
r = 0.01  # Interest free rate
iv = 0.2  # Implied Volatility

option_price = black_scholes(flag, S, K, t, r, iv)  # 12.111581435

# This library keeps the same syntax, but you can pass as input any iterable of values.
# This includes list, tuple, numpy.array, pd.Series, pd.DataFrame (with only a single column).
# Note that you must pass a value for each contract as *no broadcasting* is done on the inputs.


# Patch the original py_vollib library by importing py_vollib_vectorized
import py_vollib_vectorized  # The same functions now accept vectors as input!

# Note that the input arguments are broadcasted.
# You can specify ints, floats, tuples, lists, numpy arrays or Series.

flag = ['c', 'p']  # 'c' for call, 'p' for put
S = (100, 100)  # Underlying asset prices
K = [90]  # Strikes
t = pd.Series([0.5, 0.6])  # (Annualized) times-to-expiration
r = np.array([0.01])  # Interest free rates
iv = 0.2  # Implied Volatilities

option_price = black_scholes(flag, S, K, t, r, iv, return_as='array')  
# array([12.111581435, 1.66270456231])


# We also define other utility functions to get all contract greeks in one call.

from py_vollib_vectorized import get_all_greeks

greeks = get_all_greeks(flag, S, K, t, r, iv, model='black_scholes', return_as='dict')

#   {'delta': array([ 0.80263679, -0.21293214]),
#    'gamma': array([0.0196385, 0.01875498]),
#    'theta': array([-0.01263557, -0.00964498]),
#    'rho': array([0.34073321, -0.13994668]),
#    'vega': array([0.19626478, 0.22493816])}

```

See the documentation and the tests directory for more details on calculating IV, option prices, or greeks.

## Benchmarking

Compared to looping through contracts or to using built-in pandas functionality, this library is very memory efficient and scales fast and well to a large number of contracts.

![Performance of the py_vollib_vectorized libary](Isolated.png "Title")


## Acknowledgements

This library optimizes the `py_vollib` codebase, itself built upon Peter Jäckel's *Let's be rational* methodology.
