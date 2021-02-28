import warnings
import numpy as np
import pandas as pd

from ._iv_models import implied_volatility_from_a_transformed_rational_guess, forward_price
from .util.data_format import _preprocess_flags, maybe_format_data_and_broadcast, _validate_data, _check_below_and_above_intrinsic, _check_minus_above_float

def vectorized_implied_volatility(price, S, K, t, r, flag, q=None, *, on_error="warn",
                                  model="black_scholes", return_as="dataframe",
                                  dtype=np.float64, **kwargs):
    """
    An extremely fast, efficient and accurate Implied Volatility calculator for option/future contracts.
    Inputs can be lists, tuples, floats, :obj:`pd.Series`, or `numpy.arrays`.
    Broadcasting is applied on the inputs.

    :param price: The price of the option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param q: The annualized continuous dividend yield.
    :param on_error: Either "raise", "warn" or "ignore".
    :param model: Must be one of "black_scholes" or "black_scholes_merton". Use `vectorized_implied_volatility_black` for the Black model.
    :param return_as: To return as a :obj:`pd.Series` object, use "series". To return as a :obj:`pd.DataFrame` object, use "dataframe". Any other value will return a :obj:`numpy.array` object.
    :param dtype: Data type.
    :param kwargs: Other keyword arguments are ignored.
    :return: :obj:`pd.Series`, :obj:`pd.DataFrame` or :obj:`numpy.array` object containing the implied volatility for each contract.
    >>> import py_vollib.black_scholes_merton.implied_volatility
    >>> import py_vollib_vectorized
    >>> price = 0.10
    >>> S = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> flag = ['c', 'p']
    >>> py_vollib.black_scholes_merton.implied_volatility.implied_volatility(price, S, K, t, r, flag, q=0, return_as='numpy')
    array([0.02621257, 0.12585767])
    >>> py_vollib_vectorized.vectorized_implied_volatility(price, S, K, t, r, flag, q=0, model='black_scholes_merton',return_as='numpy')  # equivalent
    array([0.02621257, 0.12585767])
    """
    flag = _preprocess_flags(flag, dtype)
    price, S, K, t, r, flag = maybe_format_data_and_broadcast(price, S, K, t, r, flag, dtype=dtype)
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
        price, S, K, t, r, q = maybe_format_data_and_broadcast(price, S, K, t, r, q,
                                                               dtype=dtype)  # recheck to make sure q matches
        _validate_data(r, q)
        F = S * np.exp((r - q) * t)
    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    below_intrinsic, above_max_price = _check_below_and_above_intrinsic(K, F, flag, undiscounted_option_price, on_error)

    sigma_calc = implied_volatility_from_a_transformed_rational_guess(undiscounted_option_price, F, K, t, flag)
    sigma_calc = np.ascontiguousarray(sigma_calc)

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


def vectorized_implied_volatility_black(price, F, K, r, t, flag, *, on_error="warn", return_as="dataframe",
                                        dtype=np.float64, **kwargs):
    """
    An extremely fast, efficient and accurate Implied Volatility calculator for option/future contracts.
    Inputs can be lists, tuples, floats, :obj:`pd.Series`, or `numpy.arrays`.
    Broadcasting is applied on the inputs.
    This method should only be used in the black model of pricing.
    Argument order is kept consistent with that of the `py_vollib` package.

    :param price: The price of the option.
    :param F: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param on_error: Either "raise", "warn" or "ignore".
    :param return_as: To return as a :obj:`pd.Series` object, use "series". To return as a :obj:`pd.DataFrame` object, use "dataframe". Any other value will return a :obj:`numpy.array` object.
    :param dtype: Data type.
    :param kwargs: Other keyword arguments are ignored.
    :return: :obj:`pd.Series`, :obj:`pd.DataFrame` or :obj:`numpy.array` object containing the implied volatility for each contract.
    >>> import py_vollib.black.implied_volatility
    >>> import py_vollib_vectorized
    >>> price = 0.10
    >>> F = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> flag = ['c', 'p']
    >>> py_vollib.black.implied_volatility.implied_volatility(price, F, K, r, t, flag, return_as='numpy')
    array([0.02621257, 0.12585767])
    >>> py_vollib_vectorized.vectorized_implied_volatility_black(price, F, K, r, t, flag, return_as='numpy')  # equivalent
    array([0.02621257, 0.12585767])
    """
    return vectorized_implied_volatility(price, F, K, t, r, flag, model="black", on_error=on_error, return_as=return_as,
                                         dtype=dtype)
