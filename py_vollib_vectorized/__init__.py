from .entrypoints import implied_volatility_vectorized, get_all_greeks, delta as vectorized_delta,\
    gamma as vectorized_gamma, rho as vectorized_rho, vega as vectorized_vega, theta as vectorized_theta

#TODO the readme file
#TODO small test suite


from py_vollib.black_scholes.implied_volatility import implied_volatility as orig_implied_volatility
from py_vollib.black_scholes.greeks.numerical import delta

import py_vollib
from functools import partial

# monkey-patch py_vollib fn's
#TODO use functools partial to specify the black vs black scholes vs merton

py_vollib.black_scholes.implied_volatility.implied_volatility = implied_volatility_vectorized
# orig_implied_volatility = implied_volatility_vectorized


from .entrypoints import get_all_greeks

py_vollib.black.greeks.numerical.delta = partial(vectorized_delta, model="black")
py_vollib.black.greeks.numerical.gamma = partial(vectorized_gamma, model="black")
py_vollib.black.greeks.numerical.rho = partial(vectorized_rho, model="black")
py_vollib.black.greeks.numerical.theta = partial(vectorized_theta, model="black")
py_vollib.black.greeks.numerical.vega = partial(vectorized_vega, model="black")

py_vollib.black_scholes.greeks.numerical.delta = partial(vectorized_delta, model="black_scholes")
py_vollib.black_scholes.greeks.numerical.gamma = partial(vectorized_gamma, model = "black_scholes")
py_vollib.black_scholes.greeks.numerical.rho = partial(vectorized_rho, model = "black_scholes")
py_vollib.black_scholes.greeks.numerical.theta = partial(vectorized_theta, model = "black_scholes")
py_vollib.black_scholes.greeks.numerical.vega = partial(vectorized_vega, model = "black_scholes")

py_vollib.black_scholes_merton.greeks.numerical.delta = partial(vectorized_delta, model="black_scholes_merton")
py_vollib.black_scholes_merton.greeks.numerical.gamma = partial(vectorized_gamma, model="black_scholes_merton")
py_vollib.black_scholes_merton.greeks.numerical.rho = partial(vectorized_rho, model="black_scholes_merton")
py_vollib.black_scholes_merton.greeks.numerical.theta = partial(vectorized_theta, model="black_scholes_merton")
py_vollib.black_scholes_merton.greeks.numerical.vega = partial(vectorized_vega, model="black_scholes_merton")


