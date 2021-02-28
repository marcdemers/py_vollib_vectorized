import numpy as np
import pandas as pd

from ._model_calls import _black_scholes_merton_vectorized_call, _black_vectorized_call, _black_scholes_vectorized_call
from .util.data_format import _preprocess_flags, maybe_format_data_and_broadcast, _validate_data


def vectorized_black(flag, F, K, t, r, sigma, *, return_as="dataframe", dtype=np.float64):
    """
    Price a Future option using the Black model.
    Broadcasting is applied on the inputs.

    :param F: The price of the underlying asset.
    :param K: The strike price.
    :param sigma: The Implied Volatility (as a decimal, i.e. 0.10 for 10%).
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param return_as: To return as a :obj:`pd.Series` object, use "series". To return as a :obj:`pd.DataFrame` object, use "dataframe". Any other value will return a :obj:`numpy.array` object.
    :param dtype: Data type.
    :return: The price of the option.
    >>> import py_vollib.black
    >>> import py_vollib_vectorized
    >>> flag = ['c', 'p']
    >>> F = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> sigma = .2
    >>> py_vollib.black.black(flag, F, K, t, r, sigma, return_as='numpy')
    array([1.53408169, 1.38409245])
    >>> py_vollib_vectorized.vectorized_black(flag, F, K, t, r, sigma, return_as='numpy')  # equivalent
    array([1.53408169, 1.38409245])
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    F, K, sigma, t, flag = maybe_format_data_and_broadcast(F, K, sigma, t, flag, dtype=dtype)
    _validate_data(F, K, sigma, t, flag)

    prices = _black_vectorized_call(F, K, sigma, t, flag)
    prices = np.ascontiguousarray(prices)

    if return_as == "series":
        return pd.Series(prices, name="Price")
    elif return_as == "dataframe":
        return pd.DataFrame(prices, columns=["Price"])
    return prices


def vectorized_black_scholes(flag, S, K, t, r, sigma, *, return_as="dataframe", dtype=np.float64):
    """
    Price an option using the Black-Scholes model.
    Broadcasting is applied on the inputs.

    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The underlying asset price
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The interest free rate.
    :param sigma: The Implied Volatility (as a decimal, i.e. 0.10 for 10%).
    :param return_as: To return as a :obj:`pd.Series` object, use "series". To return as a :obj:`pd.DataFrame` object, use "dataframe". Any other value will return a :obj:`numpy.array` object.
    :param dtype: Data type.
    :return: The price of the option.
    >>> import py_vollib.black_scholes
    >>> import py_vollib_vectorized
    >>> flag = ['c', 'p']
    >>> S = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> sigma = .2
    >>> py_vollib.black_scholes.black_scholes(flag, S, K, t, r, sigma, return_as='numpy')
    array([2.89558836, 0.61109351])
    >>> py_vollib_vectorized.vectorized_black_scholes(flag, S, K, t, r, sigma, return_as='numpy')  # equivalent
    array([2.89558836, 0.61109351])
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, sigma, t, r, flag = maybe_format_data_and_broadcast(S, K, sigma, t, r, flag, dtype=dtype)
    _validate_data(S, K, sigma, t, r, flag)

    prices = _black_scholes_vectorized_call(flag, S, K, t, r, sigma)
    prices = np.ascontiguousarray(prices)

    if return_as == "series":
        return pd.Series(prices, name="Price")
    elif return_as == "dataframe":
        return pd.DataFrame(prices, columns=["Price"])
    return prices


def vectorized_black_scholes_merton(flag, S, K, t, r, sigma, q, *, return_as="dataframe", dtype=np.float64):
    """
    Price an option using the Black-Scholes-Merton model.
    Broadcasting is applied on the inputs.

    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The underlying asset price
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The interest free rate.
    :param sigma: The Implied Volatility (as a decimal, i.e. 0.10 for 10%).
    :param q: The annualized continuous dividend yield.
    :param return_as: To return as a :obj:`pd.Series` object, use "series". To return as a :obj:`pd.DataFrame` object, use "dataframe". Any other value will return a :obj:`numpy.array` object.
    :param dtype: Data type.
    :return: The price of the option.
    >>> import py_vollib.black_scholes_merton
    >>> import py_vollib_vectorized
    >>> flag = ['c', 'p']
    >>> S = [95, 99]
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> sigma = .2
    >>> q = 0
    >>> py_vollib.black_scholes_merton.black_scholes_merton(flag, S, K, t, r, sigma, q, return_as='numpy')
    array([2.89558836, 0.23536284])
    >>> py_vollib_vectorized.vectorized_black_scholes_merton(flag, S, K, t, r, sigma, q, return_as='numpy')  # equivalent
    array([2.89558836, 0.23536284])
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    flag, S, K, t, r, sigma, q = maybe_format_data_and_broadcast(flag, S, K, t, r, sigma, q, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma, q)

    prices = _black_scholes_merton_vectorized_call(flag, S, K, t, r, sigma, q)
    prices = np.ascontiguousarray(prices)

    if return_as == "series":
        return pd.Series(prices, name="Price")
    elif return_as == "dataframe":
        return pd.DataFrame(prices, columns=["Price"])
    return prices
