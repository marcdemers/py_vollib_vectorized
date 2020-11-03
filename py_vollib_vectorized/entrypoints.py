import numpy as np
from py_vollib.helpers import binary_flag

from .package_delta import black_scholes
from .package_delta import numerical_delta, numerical_theta, numerical_vega, numerical_rho, numerical_gamma

def _preprocess_flags(flags):
    return np.array([binary_flag[f] for f in flags], dtype=np.float64)

def all_greeks(flag, S, K, t, r, sigma):
    b = r
    flag = _preprocess_flags(flag)
    greeks = {
        "delta": numerical_delta(flag, S, K, t, r, sigma, b),
        "gamma": numerical_gamma(flag, S, K, t, r, sigma, b),
        "theta": numerical_theta(flag, S, K, t, r, sigma, b),
        "rho": numerical_rho(flag, S, K, t, r, sigma, b),
        "vega": numerical_vega(flag, S, K, t, r, sigma, b)
    }
    return greeks


# TODO for delta of black-scholes-merton, use another pricing function `f
def delta(flag, S, K, t, r, sigma):
    """Return Black-Scholes delta of an option.

    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    """
    f = lambda flag, S, K, t, r, sigma, b: black_scholes(flag, S, K, t, r, sigma)
    # f = black_scholes
    b = r
    flag = _preprocess_flags(flag)

    return numerical_delta(flag, S, K, t, r, sigma, b)


def theta(flag, S, K, t, r, sigma):
    """Return Black-Scholes theta of an option.

    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    """
    b = r
    flag = _preprocess_flags(flag)

    return numerical_theta(flag, S, K, t, r, sigma, b)


# TODO in all the entrypoint functions, like delta, theta, etc... we need to fix the function problem of black_scoles vs black_schols_merton pricing functions and the `b`parameter
def vega(flag, S, K, t, r, sigma):
    """Return Black-Scholes vega of an option.
    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    """

    b = r
    flag = _preprocess_flags(flag)

    return numerical_vega(flag, S, K, t, r, sigma, b)


def rho(flag, S, K, t, r, sigma):
    """Return Black-Scholes rho of an option.
    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    """

    b = r
    flag = _preprocess_flags(flag)

    return numerical_rho(flag, S, K, t, r, sigma, b)


def gamma(flag, S, K, t, r, sigma):
    """Return Black-Scholes gamma of an option.
    :param S: underlying asset price
    :type S: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    :param r: risk-free interest rate
    :type r: float
    :param flag: 'c' or 'p' for call or put.
    :type flag: str
    """

    b = r
    flag = _preprocess_flags(flag)

    return numerical_gamma(flag, S, K, t, r, sigma, b)
