import warnings
import json

import numpy as np
import pandas as pd

from .models import vectorized_black, vectorized_black_scholes, vectorized_black_scholes_merton
from .implied_volatility import vectorized_implied_volatility_black, vectorized_implied_volatility

from ._numerical_greeks import numerical_delta_black, numerical_theta_black, \
    numerical_vega_black, numerical_rho_black, numerical_gamma_black
from ._numerical_greeks import numerical_delta_black_scholes, numerical_theta_black_scholes, \
    numerical_vega_black_scholes, numerical_rho_black_scholes, numerical_gamma_black_scholes
from ._numerical_greeks import numerical_delta_black_scholes_merton, numerical_theta_black_scholes_merton, \
    numerical_vega_black_scholes_merton, numerical_rho_black_scholes_merton, numerical_gamma_black_scholes_merton
from .util.data_format import _preprocess_flags, maybe_format_data_and_broadcast, _validate_data, _validate_df_col


def price_dataframe(df, *, flag_col=None, underlying_price_col=None, strike_col=None, annualized_tte_col=None,
                    riskfree_rate_col=None,
                    sigma_col=None, price_col=None, dividend_col=None,
                    model="black_scholes", inplace=False, dtype=np.float64):
    """
    Utility function to price a DataFrame of option contracts by specifying the columns corresponding to each value.
    This function automatically calculates option price, option implied volatility and greeks in one call.
    Specifying a `sigma_col` will return the option prices and greeks.
    Specifying a `price_col` will return implied volatilities and greeks.
    Specifying both will return only greeks.

    :param df: Input DataFrame.
    :type df: pd.DataFrame
    :param flag_col: Column containing the flags ('c' for call, 'p' for puts)
    :type flag_col: str
    :param underlying_price_col: Column containing the price of the underlying.
    :type underlying_price_col: str
    :param strike_col: Column containing the strike price.
    :type strike_col: str
    :param annualized_tte_col: Column containing the annualized time to expiration.
    :type annualized_tte_col: str
    :param riskfree_rate_col: Column containing the risk-free rate.
    :type riskfree_rate_col: str
    :param sigma_col: Column containing the implied volatility (if unspecified, will be calculated).
    :type sigma_col: str
    :param price_col: Column containing the price of the option (if unspecified, will be calculated).
    :type price_col: str
    :param dividend_col: Column containing the implied volatility (only for Black-Scholes-Merton).
    :type dividend_col: str
    :param model: Must be one of 'black', 'black_scholes' or 'black_scholes_merton'.
    :type model: str
    :param inplace: Whether to modify the input dataframe inplace (columns will be added) or return a :obj:`pd.DataFrame` with the result.
    :type inplace: bool
    :param dtype: Data type
    :type dtype: dtype
    :return: None if inplace is True or a :obj:`pd.DataFrame` object containing the desired calculations if inplace is False.
    :rtype: :obj:`pd.DataFrame`
    >>> df = pd.DataFrame()
    >>> df['Flag'] = ['c', 'p']
    >>> df['S'] = 95
    >>> df['K'] = [100, 90]
    >>> df['T'] = 0.2
    >>> df['R'] = 0.2
    >>> df['IV'] = 0.2
    >>> price_dataframe(df, flag_col='Flag', underlying_price_col='S', strike_col='K', annualized_tte_col='T', riskfree_rate_col='R', sigma_col='IV', model='black_scholes', inplace=False)
        Price 	    delta 	    gamma 	    theta 	    rho 	    vega
    0 	2.895588 	0.467506 	0.046795 	-0.045900 	0.083035 	0.168926
    1 	0.611094 	-0.136447 	0.025739 	-0.005335 	-0.027151 	0.092838
    """
    assert flag_col is not None, "You must specify a `flag_col` argument!"
    assert underlying_price_col is not None, "You must specify a `underlying_price_col` argument!"
    assert strike_col is not None, "You must specify a `strike_col` argument!"
    assert annualized_tte_col is not None, "You must specify a `annualized_tte_col` argument!"
    assert riskfree_rate_col is not None, "You must specify a `riskfree_rate_col` argument!"

    for col in [flag_col, underlying_price_col, strike_col, annualized_tte_col, riskfree_rate_col]:
        _validate_df_col(col, df)

    flag = df[flag_col]
    S = df[underlying_price_col]
    K = df[strike_col]
    t = df[annualized_tte_col]
    r = df[riskfree_rate_col]

    flag = _preprocess_flags(flag, dtype=dtype)
    S, K, t, r, flag = maybe_format_data_and_broadcast(S, K, t, r, flag, dtype=dtype)
    _validate_data(flag, S, K, t, r)

    _px_calc, _iv_calc, _greek_calc = False, False, False
    if price_col is not None and sigma_col is not None:
        warnings.warn(
            "Both `price_col` and `sigma_col` were specified. If these two calculations were not calculated using the same "
            "library, this may lead to inconsistencies.",
            stacklevel=2)
        _validate_df_col(price_col, df)
        _validate_df_col(sigma_col, df)
        price, sigma = df[price_col], df[sigma_col]
        _greek_calc = True
        S, K, t, r, flag, price, sigma = maybe_format_data_and_broadcast(S, K, t, r, flag, price, sigma, dtype=dtype)
        _validate_data(flag, S, K, t, r, price, sigma)
    elif price_col is not None:
        _validate_df_col(price_col, df)
        price = df[price_col]
        _iv_calc, _greek_calc = True, True
        S, K, t, r, flag, price = maybe_format_data_and_broadcast(S, K, t, r, flag, price, dtype=dtype)
        _validate_data(flag, S, K, t, r, price)
    elif sigma_col is not None:
        _validate_df_col(sigma_col, df)
        sigma = df[sigma_col]
        _px_calc, _greek_calc = True, True
        S, K, t, r, flag, sigma = maybe_format_data_and_broadcast(S, K, t, r, flag, sigma, dtype=dtype)
        _validate_data(flag, S, K, t, r, sigma)
    else:
        raise ValueError("You must specify either `sigma_col`, `price_col`, or both!")

    q = None
    if model == "black_scholes_merton":
        if dividend_col is not None:
            _validate_df_col(dividend_col, df)
            q = df[dividend_col]
            S, K, t, r, q = maybe_format_data_and_broadcast(S, K, t, r, q,
                                                            dtype=dtype)  # recheck to make sure q matches
            _validate_data(S, K, t, r, q)
        else:
            raise ValueError("You must specify a `dividend_col` value with `black_scholes_merton` model.")

    output_df = pd.DataFrame(index=df.index)

    if _px_calc:
        if model == "black":
            price = vectorized_black(flag, S, K, t, r, sigma)
        elif model == "black_scholes":
            price = vectorized_black_scholes(flag, S, K, t, r, sigma)
        elif model == "black_scholes_merton":
            price = vectorized_black_scholes_merton(flag, S, K, t, r, sigma, q)
        else:
            raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

        if inplace:
            df["Price"] = price
        else:
            output_df["Price"] = price

    if _iv_calc:
        if model == "black":
            sigma = vectorized_implied_volatility_black(
                price=price,
                F=S,
                K=K,
                t=t,
                r=r,
                flag=flag,
            )
        elif model in ["black_scholes", "black_scholes_merton"]:
            sigma = vectorized_implied_volatility(
                price=price,
                S=S,
                K=K,
                t=t,
                r=r,
                flag=flag,
                q=q,
                model=model
            )
        else:
            raise ValueError("Model must be one of: `black`, `black_scholes`, `black_scholes_merton`")

        if inplace:
            df["IV"] = sigma
        else:
            output_df["IV"] = sigma

    if _greek_calc:
        greeks = get_all_greeks(flag, S, K, t, r, sigma, q, model=model, return_as="dataframe")

        for col in greeks:
            if inplace:
                df[col] = greeks[col]
            else:
                output_df[col] = greeks[col]

    if not inplace:
        return output_df


def get_all_greeks(flag, S, K, t, r, sigma, q=None, *,  model="black_scholes", return_as="dataframe", dtype=np.float64):
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
    :param model: Must be one of 'black', 'black_scholes' or 'black_scholes_merton'.
    :param return_as: To return as a :obj:`pd.DataFrame` object, use "dataframe". To return as a `json` object, use "json". Any other value will return a python dictionary.
    :param dtype: Data type.
    :return: :obj:`pd.DataFrame`, :obj:`json` string, or :obj:`dict` object containing the greeks for each contract.
    >>> flag = ['c', 'p']
    >>> S = 95
    >>> K = [100, 90]
    >>> t = .2
    >>> r = .2
    >>> sigma = .2
    >>> get_all_greeks(flag, S, K, t, r, sigma, model='black_scholes', return_as='dict')
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
        b = r - r
        greeks = {
            "delta": numerical_delta_black_scholes(flag, S, K, t, r, sigma, b),
            "gamma": numerical_gamma_black_scholes(flag, S, K, t, r, sigma, b),
            "theta": numerical_theta_black_scholes(flag, S, K, t, r, sigma, b),
            "rho": numerical_rho_black_scholes(flag, S, K, t, r, sigma, b),
            "vega": numerical_vega_black_scholes(flag, S, K, t, r, sigma, b)
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
