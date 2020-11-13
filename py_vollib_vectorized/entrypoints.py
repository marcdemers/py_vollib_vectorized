import json

import numpy as np
import pandas as pd
from py_vollib.helpers import binary_flag
from py_vollib.helpers.constants import FLOAT_MAX, MINUS_FLOAT_MAX
from py_vollib.helpers.exceptions import PriceIsAboveMaximum, PriceIsBelowIntrinsic

from .package import implied_volatility_from_a_transformed_rational_guess, forward_price
from .package_greeks_numerical import numerical_delta_black_scholes, numerical_theta_black_scholes, \
    numerical_vega_black_scholes, numerical_rho_black_scholes, numerical_gamma_black_scholes


def _preprocess_flags(flags, dtype):
    return np.array([binary_flag[f] for f in flags], dtype=dtype)


def _maybe_format_data(data, dtype):
    if isinstance(data, (int, float)):
        return np.array([data], dtype=dtype)
    elif isinstance(data, pd.Series):
        return data.values.astype(dtype)
    elif isinstance(data, (list, tuple)):
        return np.array(data, dtype=dtype)
    elif isinstance(data, np.ndarray):
        return data.astype(dtype=dtype)
    raise ValueError(f"Data type {type(data)} unsupported, must be in: list, tuple, np.array, pd.Series.")


def maybe_format_data(*all_data, dtype):
    return tuple(np.ravel(_maybe_format_data(d, dtype=dtype)) for d in all_data)


def _validate_data(*all_data):
    number_elems = [len(x) for x in all_data]
    check = np.all([x == number_elems[0] for x in number_elems])
    if not check:
        raise ValueError(
            f"All input values must contain the same number of elements. Found number of elements = {number_elems}")


######## IV
def implied_volatility_vectorized(price, S, K, t, r, flag, on_error="raise", dtype=np.float64, **kwargs):
    # TODO documentation, should r be annualized, etc.
    flag = _preprocess_flags(flag, dtype)
    price, S, K, t, r = maybe_format_data(price, S, K, t, r, dtype=dtype)
    _validate_data(price, S, K, t, r, flag)

    deflater = np.exp((-r * t))
    undiscounted_option_price = price / deflater
    F = forward_price(S, t, r)
    sigma_calc = implied_volatility_from_a_transformed_rational_guess(undiscounted_option_price, F, K, t,
                                                                      flag)

    # TODO send user warning instead of error, and fix the kwarg on_error
    if np.any(sigma_calc == FLOAT_MAX):
        raise PriceIsAboveMaximum()
    elif np.any(sigma_calc == MINUS_FLOAT_MAX):
        raise PriceIsBelowIntrinsic()
    return sigma_calc


####### GREEKS
def all_greeks(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
    flag = _preprocess_flags(flag, dtype=dtype)
    price, S, K, t, r = maybe_format_data(S, K, t, r, dtype=dtype)
    _validate_data(price, S, K, t, r, flag)

    if model == "black":
        b = 0
    elif model == "black_scholes":
        b = r
        greeks = {
            "delta": numerical_delta_black_scholes(flag, S, K, t, r, sigma, b),
            "gamma": numerical_gamma_black_scholes(flag, S, K, t, r, sigma, b),
            "theta": numerical_theta_black_scholes(flag, S, K, t, r, sigma, b),
            "rho": numerical_rho_black_scholes(flag, S, K, t, r, sigma, b),
            "vega": numerical_vega_black_scholes(flag, S, K, t, r, sigma, b)
        }
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        b = r - q
    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "dataframe":
        return pd.DataFrame.from_dict(greeks)
    elif return_as == "json":
        return json.dumps({k: v.tolist() for k, v in greeks.items()})
    return greeks


# TODO for delta of black-scholes-merton, use another pricing function `f
def delta(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
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
    flag = _preprocess_flags(flag, dtype=dtype)
    price, S, K, t, r = maybe_format_data(S, K, t, r, dtype=dtype)
    _validate_data(price, S, K, t, r, flag)

    if model == "black":
        b = 0
    elif model == "black_scholes":
        b = r
        delta = numerical_delta_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        b = r - q
    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "series":
        return pd.Series(delta, name="delta")
    elif return_as == "dataframe":
        return pd.DataFrame(delta, columns=["delta"])
    return delta


def theta(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
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
    flag = _preprocess_flags(flag, dtype=dtype)
    price, S, K, t, r = maybe_format_data(S, K, t, r, dtype=dtype)
    _validate_data(price, S, K, t, r, flag)

    if model == "black":
        b = 0
    elif model == "black_scholes":
        b = r
        theta = numerical_theta_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        b = r - q
    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "series":
        return pd.Series(theta, name="theta")
    elif return_as == "dataframe":
        return pd.DataFrame(theta, columns=["theta"])
    return theta


# TODO in all the entrypoint functions, like delta, theta, etc... we need to fix the function problem of black_scoles vs black_schols_merton pricing functions and the `b`parameter
def vega(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
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
    flag = _preprocess_flags(flag, dtype=dtype)
    price, S, K, t, r = maybe_format_data(S, K, t, r, dtype=dtype)
    _validate_data(price, S, K, t, r, flag)

    if model == "black":
        b = 0
    elif model == "black_scholes":
        b = r
        vega = numerical_vega_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        b = r - q
    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "series":
        return pd.Series(vega, name="vega")
    elif return_as == "dataframe":
        return pd.DataFrame(vega, columns=["vega"])
    return vega


def rho(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
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

    flag = _preprocess_flags(flag, dtype=dtype)
    price, S, K, t, r = maybe_format_data(S, K, t, r, dtype=dtype)
    _validate_data(price, S, K, t, r, flag)

    if model == "black":
        b = 0
    elif model == "black_scholes":
        b = r
        rho = numerical_rho_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:  # TODO on each greek add a check for q is in right format
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        b = r - q
    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "series":
        return pd.Series(rho, name="rho")
    elif return_as == "dataframe":
        return pd.DataFrame(rho, columns=["rho"])
    return rho


def gamma(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
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
    flag = _preprocess_flags(flag, dtype=dtype)
    price, S, K, t, r = maybe_format_data(S, K, t, r, dtype=dtype)
    _validate_data(price, S, K, t, r, flag)

    if model == "black":
        b = 0
    elif model == "black_scholes":
        b = r
        gamma = numerical_gamma_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        b = r - q
    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "series":
        return pd.Series(gamma, name="gamma")
    elif return_as == "dataframe":
        return pd.DataFrame(gamma, columns=["gamma"])
    return gamma
