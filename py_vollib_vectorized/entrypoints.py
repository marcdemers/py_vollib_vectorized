import json
import warnings

import numpy as np
import pandas as pd
from py_vollib.helpers import binary_flag
from py_vollib.helpers.constants import FLOAT_MAX, MINUS_FLOAT_MAX
from py_vollib.helpers.exceptions import PriceIsAboveMaximum, PriceIsBelowIntrinsic

from .iv_models import implied_volatility_from_a_transformed_rational_guess, forward_price
from .numerical_greeks import numerical_delta_black_scholes, numerical_theta_black_scholes, \
    numerical_vega_black_scholes, numerical_rho_black_scholes, numerical_gamma_black_scholes
from .numerical_greeks import numerical_delta_black_scholes_merton, numerical_theta_black_scholes_merton, \
    numerical_vega_black_scholes_merton, numerical_rho_black_scholes_merton, numerical_gamma_black_scholes_merton

from .model_calls import _black_scholes_merton_vectorized_call, _black_vectorized_call, _black_scholes_vectorized_call


def _preprocess_flags(flags, dtype):
    return np.array([binary_flag[f] for f in flags], dtype=dtype)

#TODO examples for each function

def _maybe_format_data(data, dtype):
    if isinstance(data, (int, float)):
        return np.array([data], dtype=dtype)
    elif isinstance(data, pd.Series):
        return data.values.astype(dtype)
    elif isinstance(data, pd.DataFrame):
        assert data.shape[1] == 1, "You passed a `pandas.DataFrame` object that contains more than (1) column!"
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


def _check_below_and_above_intrinsic(Ks, Fs, qs, prices, on_error):
    _intrinsic = np.zeros(shape=Ks.shape)
    _intrinsic[qs < 0] = (Ks - Fs)[qs < 0]
    _intrinsic[qs > 0] = (Fs - Ks)[qs > 0]
    _intrinsic = np.abs(np.maximum(_intrinsic, 0.))

    _max_price = np.zeros(shape=Ks.shape)
    _max_price[qs < 0] = Ks[qs < 0]
    _max_price[qs > 0] = Fs[qs > 0]

    _below_intrinsic_array, _above_max_price = [], []

    if on_error != "ignore":
        if np.any(prices < _intrinsic):
            _below_intrinsic_array = np.argwhere(prices < _intrinsic).ravel().tolist()
            if on_error == "warn":
                warnings.warn(
                    "Found Below Intrinsic contracts at index {}".format(_below_intrinsic_array),
                    stacklevel=2)
            elif on_error == "raise":
                raise PriceIsBelowIntrinsic
        if np.any(prices >= _max_price):
            _above_max_price = np.argwhere(prices >= _max_price).ravel().tolist()
            if on_error == "warn":
                warnings.warn(
                    "Found Above Maximum Price contracts at index {}".format(_above_max_price),
                    stacklevel=2)
            elif on_error == "raise":
                raise PriceIsAboveMaximum

    return _below_intrinsic_array, _above_max_price

def _check_minus_above_float(values, on_error):
    _below_float_min_array, _above_float_max_array = [], []
    if on_error != "ignore":
        if np.any(values == FLOAT_MAX):
            _above_float_max_array = np.argwhere(values == FLOAT_MAX).ravel().tolist()
            if on_error == "warn":
                warnings.warn(
                    "Found PriceAboveMaximum contracts at index {}".format(_above_float_max_array)
                )
            elif on_error == "raise":
                raise PriceIsAboveMaximum()
        elif np.any(values == MINUS_FLOAT_MAX):
            _below_float_min_array = np.argwhere(values == MINUS_FLOAT_MAX).ravel().tolist()
            if on_error == "warn":
                warnings.warn(
                    "Found PriceBelowMaximum contracts at index {}".format(_below_float_min_array)
                )
            elif on_error == "raise":
                PriceIsBelowIntrinsic()
    return _below_float_min_array, _above_float_max_array

######## IV


def implied_volatility_vectorized(price, S, K, t, r, flag, q=None, on_error="warn",
                                  model="black_scholes", return_as="dataframe",
                                  dtype=np.float64, **kwargs):
    """
    An extremely fast, efficient and accurate Implied Volatility calculator for option/future contracts.
    Inputs can be lists, tuples, floats, `pandas.Series`, or `numpy.arrays`.
    No broadcasting is done on the inputs, all dimensions must match.
    :param price: The price of the option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param q: The annualized continuous dividend yield.
    :param on_error: Either "raise", "warn" or "ignore".
    :param model: Must be one of "black_scholes" or "black_scholes_merton". Use `implied_volatility_vectorized_black` for the Black model.
    :param return_as: To return as a `pandas.Series` object, use "series". To return as a `pd.DataFrame` object, use
    "dataframe". Any other value will return a `numpy.array` object.
    :param dtype: Data type.
    :param kwargs: Other keyword arguments are ignored.
    :return: `pd.Series`, `pd.DataFrame` or `numpy.array` object containing the implied volatility for each contract.
    """
    flag = _preprocess_flags(flag, dtype)
    price, S, K, t, r = maybe_format_data(price, S, K, t, r, dtype=dtype)
    _validate_data(price, S, K, t, r, flag)

    assert on_error in ["warn", "ignore", "raise"], "`on_error` must be 'warn', 'ignore' or 'raise'."

    deflater = np.exp(-r * t)

    if np.any(deflater == 0) and on_error != "ignore":
        if on_error == "warn":
            warnings.warn(
                "Unexpected value encountered in Interest Free Rate (r) or Annualized time to expiration (t). Are you sure the time to expiration is annualized?",
                stacklevel=2)
        elif on_error == "raise":
            raise ValueError(
                "Unexpected value encountered in Interest Free Rate (r) or Annualized time to expiration (t).")

    undiscounted_option_price = price / deflater

    if model == "black":
        F = S
    elif model == "black_scholes":
        F = forward_price(S, t, r)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        q = maybe_format_data(q, dtype=dtype)[0]
        _validate_data(r, q)
        F = S * np.exp((r - q) * t)
    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    below_intrinsic, above_max_price = _check_below_and_above_intrinsic(K, F, flag, undiscounted_option_price, on_error)

    sigma_calc = implied_volatility_from_a_transformed_rational_guess(undiscounted_option_price, F, K, t, flag)

    below_min_float, above_max_float = _check_minus_above_float(sigma_calc, on_error)

    # postprocess the results
    sigma_calc[below_intrinsic] = np.nan
    sigma_calc[above_max_price] = np.nan
    sigma_calc[below_min_float] = np.nan
    sigma_calc[above_max_float] = np.nan

    if return_as == "series":
        return pd.Series(sigma_calc, name="IV")
    elif return_as == "dataframe":
        return pd.DataFrame(sigma_calc, columns=["IV"])
    return sigma_calc


def implied_volatility_vectorized_black(price, F, K, r, t, flag, on_error="warn", return_as="dataframe",
                                        dtype=np.float64, **kwargs):
    """
    An extremely fast, efficient and accurate Implied Volatility calculator for option/future contracts.
    Inputs can be lists, tuples, floats, `pandas.Series`, or `numpy.arrays`.
    No broadcasting is done on the inputs, all dimensions must match.
    This method should only be used in the black model of pricing.
    Argument order is kept consistent with that of the `py_vollib` package.
    :param price: The price of the option.
    :param F: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param on_error: Either "raise", "warn" or "ignore".
    :param return_as: To return as a `pandas.Series` object, use "series". To return as a `pd.DataFrame` object, use
    "dataframe". Any other value will return a `numpy.array` object.
    :param dtype: Data type.
    :param kwargs: Other keyword arguments are ignored.
    :return: `pd.Series`, `pd.DataFrame` or `numpy.array` object containing the implied volatility for each contract.
    """
    return implied_volatility_vectorized(price, F, K, t, r, flag, model="black", on_error=on_error, return_as=return_as,
                                         dtype=dtype)


####### GREEKS
def get_all_greeks(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Utility function that returns all contract greeks, as specified by the pricing model `model`.
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of "black", "black_scholes" or "black_scholes_merton".
    :param return_as: To return as a `pd.DataFrame` object, use "dataframe". To return as a `json` object, use "json".
    Any other value will return a python dictionary.
    :param dtype: Data type.
    :return: `pd.DataFrame`, `json` string, or `dict` object containing the greeks for each contract.
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma = maybe_format_data(S, K, t, r, sigma, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
    elif model == "black_scholes":
        b = r
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        q = maybe_format_data(q, dtype=dtype)[0]
        _validate_data(r, q)
        b = r - q
    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    greeks = {
        "delta": numerical_delta_black_scholes_merton(flag, S, K, t, r, sigma, b),
        "gamma": numerical_gamma_black_scholes_merton(flag, S, K, t, r, sigma, b),
        "theta": numerical_theta_black_scholes_merton(flag, S, K, t, r, sigma, b),
        "rho": numerical_rho_black_scholes_merton(flag, S, K, t, r, sigma, b),
        "vega": numerical_vega_black_scholes_merton(flag, S, K, t, r, sigma, b)
    }

    if return_as == "dataframe":
        return pd.DataFrame.from_dict(greeks)
    elif return_as == "json":
        return json.dumps({k: v.tolist() for k, v in greeks.items()})
    return greeks


def delta(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Return the delta of a contract, as specified by the pricing model `model`.
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of "black", "black_scholes" or "black_scholes_merton".
    :param return_as: To return as a `pandas.Series` object, use "series". To return as a `pd.DataFrame` object, use
    "dataframe". Any other value will return a `numpy.array` object.
    :param dtype: Data type.
    :return: `pd.Series`, `pd.DataFrame` or `numpy.array` object containing the delta for each contract.
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma = maybe_format_data(S, K, t, r, sigma, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        delta = numerical_delta_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes":
        b = r
        delta = numerical_delta_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        q = maybe_format_data(q, dtype=dtype)[0]
        _validate_data(r, q)
        b = r - q
        delta = numerical_delta_black_scholes_merton(flag, S, K, t, r, sigma, b)

    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "series":
        return pd.Series(delta, name="delta")
    elif return_as == "dataframe":
        return pd.DataFrame(delta, columns=["delta"])
    return delta


def theta(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Return the theta of a contract, as specified by the pricing model `model`.
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of "black", "black_scholes" or "black_scholes_merton".
    :param return_as: To return as a `pandas.Series` object, use "series". To return as a `pd.DataFrame` object, use
    "dataframe". Any other value will return a `numpy.array` object.
    :param dtype: Data type.
    :return: `pd.Series`, `pd.DataFrame` or `numpy.array` object containing the theta for each contract.
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma = maybe_format_data(S, K, t, r, sigma, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        theta = numerical_theta_black_scholes(flag, S, K, t, r, sigma, b)

    elif model == "black_scholes":
        b = r
        theta = numerical_theta_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        q = maybe_format_data(q, dtype=dtype)[0]
        _validate_data(r, q)
        b = r - q
        theta = numerical_theta_black_scholes_merton(flag, S, K, t, r, sigma, b)

    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "series":
        return pd.Series(theta, name="theta")
    elif return_as == "dataframe":
        return pd.DataFrame(theta, columns=["theta"])
    return theta


def vega(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Return the vega of a contract, as specified by the pricing model `model`.
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of "black", "black_scholes" or "black_scholes_merton".
    :param return_as: To return as a `pandas.Series` object, use "series". To return as a `pd.DataFrame` object, use
    "dataframe". Any other value will return a `numpy.array` object.
    :param dtype: Data type.
    :return: `pd.Series`, `pd.DataFrame` or `numpy.array` object containing the vega for each contract.
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma = maybe_format_data(S, K, t, r, sigma, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        vega = numerical_vega_black_scholes(flag, S, K, t, r, sigma, b)

    elif model == "black_scholes":
        b = r
        vega = numerical_vega_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        b = r - q
        q = maybe_format_data(q, dtype=dtype)[0]
        _validate_data(r, q)
        vega = numerical_vega_black_scholes_merton(flag, S, K, t, r, sigma, b)

    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "series":
        return pd.Series(vega, name="vega")
    elif return_as == "dataframe":
        return pd.DataFrame(vega, columns=["vega"])
    return vega


def rho(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Return the rho of a contract, as specified by the pricing model `model`.
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of "black", "black_scholes" or "black_scholes_merton".
    :param return_as: To return as a `pandas.Series` object, use "series". To return as a `pd.DataFrame` object, use
    "dataframe". Any other value will return a `numpy.array` object.
    :param dtype: Data type.
    :return: `pd.Series`, `pd.DataFrame` or `numpy.array` object containing the rho for each contract.
    """

    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma = maybe_format_data(S, K, t, r, sigma, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        rho = numerical_rho_black_scholes(flag, S, K, t, r, sigma, b)

    elif model == "black_scholes":
        b = r
        rho = numerical_rho_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        q = maybe_format_data(q, dtype=dtype)[0]
        _validate_data(r, q)
        b = r - q
        rho = numerical_rho_black_scholes_merton(flag, S, K, t, r, sigma, b)

    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "series":
        return pd.Series(rho, name="rho")
    elif return_as == "dataframe":
        return pd.DataFrame(rho, columns=["rho"])
    return rho


def gamma(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Return the gamma of a contract, as specified by the pricing model `model`.
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of "black", "black_scholes" or "black_scholes_merton".
    :param return_as: To return as a `pandas.Series` object, use "series". To return as a `pd.DataFrame` object, use
    "dataframe". Any other value will return a `numpy.array` object.
    :param dtype: Data type.
    :return: `pd.Series`, `pd.DataFrame` or `numpy.array` object containing the gamma for each contract.
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma = maybe_format_data(S, K, t, r, sigma, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        # TODO for these models are we certain that black schoels behaves same as black? because in the call to numerical
        # black scholes, it calls the black_scholes function and not the black function.
        gamma = numerical_gamma_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes":
        b = r
        gamma = numerical_gamma_black_scholes(flag, S, K, t, r, sigma, b)
    elif model == "black_scholes_merton":
        if q is None:
            raise ValueError("Must pass a `q` to black scholes merton model (annualized continuous dividend yield).")
        q = maybe_format_data(q, dtype=dtype)[0]
        _validate_data(r, q)
        b = r - q
        gamma = numerical_gamma_black_scholes_merton(flag, S, K, t, r, sigma, b)

    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "series":
        return pd.Series(gamma, name="gamma")
    elif return_as == "dataframe":
        return pd.DataFrame(gamma, columns=["gamma"])
    return gamma


###### IV models

def black_vectorized(F, K, sigma, t, flag, return_as="dataframe", dtype=np.float64):
    flag = _preprocess_flags(flag, dtype=dtype)
    F, K, sigma, t = maybe_format_data(F, K, sigma, t, dtype=dtype)
    _validate_data(F, K, sigma, t, flag)

    prices = _black_vectorized_call(F, K, sigma, t, flag)

    if return_as == "series":
        return pd.Series(prices, name="Price")
    elif return_as == "dataframe":
        return pd.DataFrame(prices, columns=["Price"])
    return np.array(prices)


def black_scholes_vectorized(flag, S, K, t, r, sigma, return_as="dataframe", dtype=np.float64):
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, sigma, t, r = maybe_format_data(S, K, sigma, t, r, dtype=dtype)
    _validate_data(S, K, sigma, t, r, flag)

    prices = _black_scholes_vectorized_call(flag, S, K, t, r, sigma)

    if return_as == "series":
        return pd.Series(prices, name="Price")
    elif return_as == "dataframe":
        return pd.DataFrame(prices, columns=["Price"])
    return np.array(prices)


def black_scholes_merton_vectorized(flag, S, K, t, r, sigma, q, return_as="dataframe", dtype=np.float64):
    flag = _preprocess_flags(flag, dtype=dtype)
    flag, S, K, t, r, sigma, q = maybe_format_data(flag, S, K, t, r, sigma, q, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma, q)

    prices = _black_scholes_merton_vectorized_call(flag, S, K, t, r, sigma, q)

    if return_as == "series":
        return pd.Series(prices, name="Price")
    elif return_as == "dataframe":
        return pd.DataFrame(prices, columns=["Price"])
    return np.array(prices)
