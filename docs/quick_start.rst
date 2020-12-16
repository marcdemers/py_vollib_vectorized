:github_url: https://github.com/marcdemers/py_vollib_vectorized

Quick start
============

Philosophy
-------------------------

The philosophy of py_vollib_vectorized is to provide seamless integration of vectorization into the py_vollib library.
py_vollib_vectorized also possesses an API which simplifies the process of obtaining all option greeks for option contracts.


Monkey-patching
------------------------

Upon import, `py_vollib_vectorized` monkey-patches (i.e. replaces) all relevant functions in py_vollib to make them accept lists, tuples, numpy arrays or pandas Series.
The calculations are therefore much faster and more memory efficient, which is in some cases a benefit, in others a necessity.

The example below shows that the monkey-patch is applied to the `black` function from `py_vollib.`
You can confirm this by printing the function definition.

.. code-block:: python

    >>> from py_vollib.black import black
    >>> black # check if the monkey-patch is applied.
    Vectorized <vectorized_black()>
    >>> import py_vollib_vectorized
    >>> flag = 'c'  # 'c' for call, 'p' for put
    >>> S = 95  # price of the underlying
    >>> K = 100  # strike
    >>> t = .2  # annualized time to expiration
    >>> r = .2  # interest-free rate
    >>> sigma = .2  # implied volatility
    >>> black_scholes(flag, S, K, t, r, sigma, return_as='numpy')
    array([2.89558836])


Data Format
------------------------

All input arguments are raveled. In order to avoid conflicts or mispricing, you should supply 0- or 1-dimensional arrays or Series to :underline:`all` functions.
By default, all input arguments are also broadcasted to the largest input argument.
If you supply unbroadcastable inputs (e.g. a 2-item list and a 3-item list), a ValueError is generated.

You can supply the inputs as ints, floats, lists, tuples, numpy arrays or pandas Series, or a mix of all of those.
You can also ask to return the result in a specific format(see documentation of the specific functions for the accepted formats).


.. code-block:: python

    >>> from py_vollib.black_scholes.implied_volatility import implied_volatility
    >>> import py_vollib_vectorized
    >>> price = 0.2
    >>> flag = ['c', 'p']  # 'c' for call, 'p' for put
    >>> S = (95, 10)  # price of the underlying
    >>> K = 100  # strike
    >>> t = pd.Series([.2])  # annualized time to expiration
    >>> r = .2  # interest-free rate
    >>> sigma = .2  # implied volatility
    >>> implied_volatility(price, S, K, t, r, flag, return_as='series')
    0    0.034543
    1         NaN
    Name: IV, dtype: float64


Here, the put contract, with the underlying price of 10 and the strike at 100, was found to be below intrinsic price.
py_vollib_vectorized returns contracts with below intrinsic or above maximum price as NaNs.




