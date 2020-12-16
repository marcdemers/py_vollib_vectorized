:github_url: https://github.com/marcdemers/py_vollib_vectorized

Quick start
============

Philosophy
-------------------------

The philosophy of :obj:`py_vollib_vectorized` is to provide seamless integration of vectorization into the `py_vollib <http://www.vollib.org/documentation/python/1.0.2/>`_ library.
:obj:`py_vollib_vectorized` also possesses an API which simplifies the process of obtaining all option greeks for option contracts.


Monkey-patching
------------------------

Upon import, :obj:`py_vollib_vectorized` monkey-patches (i.e. replaces) all relevant functions in :obj:`py_vollib` to make them accept floats as well as :obj:`list`, :obj:`tuple`, :obj:`numpy.array` or :obj:`pandas.Series`.
The calculations are therefore much faster and more memory efficient, which is in some cases a benefit, in others a necessity.

The example below shows that the monkey-patch is applied to the :meth:`py_vollib.black.black` function.
You can confirm this by printing the function definition.

.. code-block:: python

    >>> from py_vollib.black import black
    >>> import py_vollib_vectorized
    >>> black # check if the monkey-patch is applied.
    Vectorized <vectorized_black()>
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

All input arguments are raveled.
In order to avoid conflicts or mispricing, you should supply 0- or 1-dimensional arrays or :obj:`pandas.Series` to :underline:`all` functions.
By default, all input arguments are broadcasted to the largest input argument.
If you supply unbroadcastable inputs (e.g. a 2-item list and a 3-item list), a :obj:`ValueError` is generated.

Again, you can supply the inputs as :obj:`int`, :obj:`float`, :obj:`list`, :obj:`tuple`, :obj:`numpy.array` or :obj:`pandas.Series`, or a mix of all of those.
You can also ask to return the result in a specific format (see documentation of the specific functions for the accepted formats).


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


Here, the put contract is below the intrinsic price.
Contracts below intrinsic or above maximum price are returned as NaNs.




