import json

import numpy as np
import pandas as pd

from ._numerical_greeks import numerical_delta_black, numerical_theta_black, \
    numerical_vega_black, numerical_rho_black, numerical_gamma_black
from ._numerical_greeks import numerical_delta_black_scholes, numerical_theta_black_scholes, \
    numerical_vega_black_scholes, numerical_rho_black_scholes, numerical_gamma_black_scholes
from ._numerical_greeks import numerical_delta_black_scholes_merton, numerical_theta_black_scholes_merton, \
    numerical_vega_black_scholes_merton, numerical_rho_black_scholes_merton, numerical_gamma_black_scholes_merton
from .util.data_format import _preprocess_flags, maybe_format_data_and_broadcast, _validate_data


def get_all_greeks(flag, S, K, t, r, sigma, q=None, model="black_scholes", return_as="dataframe", dtype=np.float64):
    """
    Utility function that returns all contract greeks, as specified by the pricing model `model`.
    Broadcasting is applied on the inputs.

    :param flag: For each contract, this should be specified as `c` for a call option and `p` for a put option.
    :param S: The price of the underlying asset.
    :param K: The strike price.
    :param t: The annualized time to expiration. Must be positive. For small TTEs, use a small value (1e-3).
    :param r: The Interest Free Rate.
    :param sigma: The Implied Volatility.
    :param q: The annualized continuous dividend yield.
    :param model: Must be one of "black", "black_scholes" or "black_scholes_merton".
    :param return_as: To return as a `pd.DataFrame` object, use "dataframe". To return as a `json` object, use "json". Any other value will return a python dictionary.
    :param dtype: Data type.
    :return: `pd.DataFrame`, `json` string, or `dict` object containing the greeks for each contract.
    >>> flag = ['c', 'p']
    >>> S = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> sigma = .2
    >>> get_all_greeks(flag, S, K, t, r, sigma, model='black_scholes', return_as='numpy')
    {'delta': array([ 0.46750566, -0.1364465 ]),
     'gamma': array([0.0467948, 0.0257394]),
     'theta': array([-0.04589963, -0.00533543]),
     'rho': array([ 0.0830349 , -0.02715114]),
     'vega': array([0.16892575, 0.0928379 ])}
    """
    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, sigma, flag = maybe_format_data_and_broadcast(S, K, t, r, sigma, flag, dtype=dtype)
    _validate_data(flag, S, K, t, r, sigma)

    if model == "black":
        b = 0
        greeks = {
            "delta": numerical_delta_black(flag, S, K, t, r, sigma, b),
            "gamma": numerical_gamma_black(flag, S, K, t, r, sigma, b),
            "theta": numerical_theta_black(flag, S, K, t, r, sigma, b),
            "rho": numerical_rho_black(flag, S, K, t, r, sigma, b),
            "vega": numerical_vega_black(flag, S, K, t, r, sigma, b)
        }
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
        S, K, t, r, sigma, q = maybe_format_data_and_broadcast(S, K, t, r, sigma, q,
                                                               dtype=dtype)  # recheck to make sure q matches
        _validate_data(r, q)
        b = r - q
        greeks = {
            "delta": numerical_delta_black_scholes_merton(flag, S, K, t, r, sigma, b),
            "gamma": numerical_gamma_black_scholes_merton(flag, S, K, t, r, sigma, b),
            "theta": numerical_theta_black_scholes_merton(flag, S, K, t, r, sigma, b),
            "rho": numerical_rho_black_scholes_merton(flag, S, K, t, r, sigma, b),
            "vega": numerical_vega_black_scholes_merton(flag, S, K, t, r, sigma, b)
        }
    else:
        raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

    if return_as == "dataframe":
        return pd.DataFrame.from_dict(greeks)
    elif return_as == "json":
        return json.dumps({k: v.tolist() for k, v in greeks.items()})
    return greeks
