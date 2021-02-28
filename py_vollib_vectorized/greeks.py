import numpy as np
import pandas as pd

from ._numerical_greeks import numerical_delta_black, numerical_theta_black, \
    numerical_vega_black, numerical_rho_black, numerical_gamma_black
from ._numerical_greeks import numerical_delta_black_scholes, numerical_theta_black_scholes, \
    numerical_vega_black_scholes, numerical_rho_black_scholes, numerical_gamma_black_scholes
from ._numerical_greeks import numerical_delta_black_scholes_merton, numerical_theta_black_scholes_merton, \
    numerical_vega_black_scholes_merton, numerical_rho_black_scholes_merton, numerical_gamma_black_scholes_merton
from .util.data_format import _preprocess_flags, maybe_format_data_and_broadcast, _validate_data

def delta(flag, S, K, t, r, sigma, q=None, *, model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Return the delta of a contract, as specified by the pricing model `model`.
    Broadcasting is applied on the inputs.

    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of 'black', 'black_scholes' or 'black_scholes_merton'.
    :param return_as: To return as a :obj:`pd.Series` object, use "series". To return as a :obj:`pd.DataFrame` object, use "dataframe". Any other value will return a :obj:`numpy.array` object.
    :param dtype: Data type.
    :return: :obj:`pd.Series`, :obj:`pd.DataFrame` or :obj:`numpy.array` object containing the delta for each contract.
    >>> import py_vollib.black_scholes.greeks.numerical
    >>> import py_vollib_vectorized
    >>> flag = ['c', 'p']
    >>> S = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> sigma = .2
    >>> py_vollib.black_scholes.greeks.numerical.delta(flag, S, K, t, r, sigma, return_as='numpy')
    array([ 0.46750566, -0.1364465 ])
    >>> py_vollib_vectorized.vectorized_delta(flag, S, K, t, r, sigma, model='black_scholes', return_as='numpy')  # equivalent
    array([ 0.46750566, -0.1364465 ])
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma, flag = maybe_format_data_and_broadcast(S, K, t, r, sigma, flag, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        delta = numerical_delta_black(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes":
        b = r
        delta = numerical_delta_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        q = maybe_format_data_and_broadcast(q, dtype=dtype)[0]
        S, K, t, r, sigma, q = maybe_format_data_and_broadcast(S, K, t, r, sigma, q,
                                                               dtype=dtype)  # recheck to make sure q matches

        _validate_data(r, q)
        b = r - q
        delta = numerical_delta_black_scholes_merton(flag, S, K, t, r, sigma, b)

    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    delta = np.ascontiguousarray(delta)

    if return_as == "series":
        return pd.Series(delta, name="delta")
    elif return_as == "dataframe":
        return pd.DataFrame(delta, columns=["delta"])
    return delta


def theta(flag, S, K, t, r, sigma, q=None, *,  model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Return the theta of a contract, as specified by the pricing model `model`.
    Broadcasting is applied on the inputs.

    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of 'black', 'black_scholes' or 'black_scholes_merton'.
    :param return_as: To return as a :obj:`pd.Series` object, use "series". To return as a :obj:`pd.DataFrame` object, use "dataframe". Any other value will return a :obj:`numpy.array` object.
    :param dtype: Data type.
    :return: :obj:`pd.Series`, :obj:`pd.DataFrame` or :obj:`numpy.array` object containing the theta for each contract.
    >>> import py_vollib.black_scholes.greeks.numerical
    >>> import py_vollib_vectorized
    >>> flag = ['c', 'p']
    >>> S = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> sigma = .2
    >>> py_vollib.black_scholes.greeks.numerical.theta(flag, S, K, t, r, sigma, return_as='numpy')
    array([-0.04589963, -0.00533543])
    >>> py_vollib_vectorized.vectorized_theta(flag, S, K, t, r, sigma, model='black_scholes', return_as='numpy')  # equivalent
    array([-0.04589963, -0.00533543])
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma, flag = maybe_format_data_and_broadcast(S, K, t, r, sigma, flag, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        theta = numerical_theta_black(flag, S, K, t, r, sigma, b)

    elif model == "black_scholes":
        b = r
        theta = numerical_theta_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        S, K, t, r, sigma, q = maybe_format_data_and_broadcast(S, K, t, r, sigma, q,
                                                               dtype=dtype)  # recheck to make sure q matches
        _validate_data(r, q)
        b = r - q
        theta = numerical_theta_black_scholes_merton(flag, S, K, t, r, sigma, b)

    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    theta = np.ascontiguousarray(theta)

    if return_as == "series":
        return pd.Series(theta, name="theta")
    elif return_as == "dataframe":
        return pd.DataFrame(theta, columns=["theta"])
    return theta


def vega(flag, S, K, t, r, sigma, q=None, *, model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Return the vega of a contract, as specified by the pricing model `model`.
    Broadcasting is applied on the inputs.

    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of 'black', 'black_scholes' or 'black_scholes_merton'.
    :param return_as: To return as a :obj:`pd.Series` object, use "series". To return as a :obj:`pd.DataFrame` object, use "dataframe". Any other value will return a :obj:`numpy.array` object.
    :param dtype: Data type.
    :return: :obj:`pd.Series`, :obj:`pd.DataFrame` or :obj:`numpy.array` object containing the vega for each contract.
    >>> import py_vollib.black_scholes.greeks.numerical
    >>> import py_vollib_vectorized
    >>> flag = ['c', 'p']
    >>> S = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> sigma = .2
    >>> py_vollib.black_scholes.greeks.numerical.vega(flag, S, K, t, r, sigma, return_as='numpy')
    array([0.16892575, 0.0928379 ])
    >>> py_vollib_vectorized.vectorized_vega(flag, S, K, t, r, sigma, model='black_scholes', return_as='numpy')  # equivalent
    array([0.16892575, 0.0928379 ])
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma, flag = maybe_format_data_and_broadcast(S, K, t, r, sigma, flag, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        vega = numerical_vega_black(flag, S, K, t, r, sigma, b)

    elif model == "black_scholes":
        b = r
        vega = numerical_vega_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        S, K, t, r, sigma, q = maybe_format_data_and_broadcast(S, K, t, r, sigma, q,
                                                               dtype=dtype)  # recheck to make sure q matches
        _validate_data(r, q)
        b = r - q
        vega = numerical_vega_black_scholes_merton(flag, S, K, t, r, sigma, b)

    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    vega = np.ascontiguousarray(vega)

    if return_as == "series":
        return pd.Series(vega, name="vega")
    elif return_as == "dataframe":
        return pd.DataFrame(vega, columns=["vega"])
    return vega


def rho(flag, S, K, t, r, sigma, q=None, *, model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Return the rho of a contract, as specified by the pricing model `model`.
    Broadcasting is applied on the inputs.

    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of 'black', 'black_scholes' or 'black_scholes_merton'.
    :param return_as: To return as a :obj:`pd.Series` object, use "series". To return as a :obj:`pd.DataFrame` object, use "dataframe". Any other value will return a :obj:`numpy.array` object.
    :param dtype: Data type.
    :return: :obj:`pd.Series`, :obj:`pd.DataFrame` or :obj:`numpy.array` object containing the rho for each contract.
    >>> import py_vollib.black_scholes.greeks.numerical
    >>> import py_vollib_vectorized
    >>> flag = ['c', 'p']
    >>> S = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> sigma = .2
    >>> py_vollib.black_scholes.greeks.numerical.rho(flag, S, K, t, r, sigma, return_as='numpy')
    array([ 0.0830349 , -0.02715114])
    >>> py_vollib_vectorized.vectorized_rho(flag, S, K, t, r, sigma, model='black_scholes', return_as='numpy')  # equivalent
    array([ 0.0830349 , -0.02715114])
    """

    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma, flag = maybe_format_data_and_broadcast(S, K, t, r, sigma, flag, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        rho = numerical_rho_black(flag, S, K, t, r, sigma, b)

    elif model == "black_scholes":
        b = r
        rho = numerical_rho_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        q = maybe_format_data_and_broadcast(q, dtype=dtype)[0]

        _validate_data(r, q)
        b = r - q
        rho = numerical_rho_black_scholes_merton(flag, S, K, t, r, sigma, b)

    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    rho = np.ascontiguousarray(rho)

    if return_as == "series":
        return pd.Series(rho, name="rho")
    elif return_as == "dataframe":
        return pd.DataFrame(rho, columns=["rho"])
    return rho


def gamma(flag, S, K, t, r, sigma, q=None, *,  model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Return the gamma of a contract, as specified by the pricing model `model`.
    Broadcasting is applied on the inputs.

    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of 'black', 'black_scholes' or 'black_scholes_merton'.
    :param return_as: To return as a :obj:`pd.Series` object, use "series". To return as a :obj:`pd.DataFrame` object, use "dataframe". Any other value will return a :obj:`numpy.array` object.
    :param dtype: Data type.
    :return: :obj:`pd.Series`, :obj:`pd.DataFrame` or :obj:`numpy.array` object containing the gamma for each contract.
    >>> import py_vollib.black_scholes.greeks.numerical
    >>> import py_vollib_vectorized
    >>> flag = ['c', 'p']
    >>> S = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> sigma = .2
    >>> py_vollib.black_scholes.greeks.numerical.gamma(flag, S, K, t, r, sigma, return_as='numpy')
    array([0.0467948, 0.0257394])
    >>> py_vollib_vectorized.vectorized_gamma(flag, S, K, t, r, sigma, model='black_scholes', return_as='numpy')  # equivalent
    array([0.0467948, 0.0257394])
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma, flag = maybe_format_data_and_broadcast(S, K, t, r, sigma, flag, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        # black scholes, it calls the black_scholes function and not the black function.
        gamma = numerical_gamma_black(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes":
        b = r
        gamma = numerical_gamma_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        S, K, t, r, sigma, q = maybe_format_data_and_broadcast(S, K, t, r, sigma, q,
                                                               dtype=dtype)  # recheck to make sure q matches
        _validate_data(r, q)
        b = r - q
        gamma = numerical_gamma_black_scholes_merton(flag, S, K, t, r, sigma, b)

    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    gamma = np.ascontiguousarray(gamma)

    if return_as == "series":
        return pd.Series(gamma, name="gamma")
    elif return_as == "dataframe":
        return pd.DataFrame(gamma, columns=["gamma"])
    return gamma
