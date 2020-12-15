
from functools import partial, update_wrapper

from .implied_volatility import vectorized_implied_volatility_black, vectorized_implied_volatility
from .api import get_all_greeks
from .greeks import delta as vectorized_delta, \
    gamma as vectorized_gamma, rho as vectorized_rho, vega as vectorized_vega, theta as vectorized_theta

from .models import vectorized_black, vectorized_black_scholes, vectorized_black_scholes_merton


# TODO the readme file
#TODO AOT compilation and remove dependency on numba
#todo repr partial arguments of function, verify for vectorized_black for instance
class repr_partial(partial):
    def __repr__(self):
        return "Vectorized <{fn}({args}{kwargs})>".format(fn=self.func.__name__,
                                                          args=", ".join(repr(a) for a in self.args),
                                                          kwargs=", ".join(
                                                              [str(k) + "=" + str(v) for k, v in self.keywords.items()])
                                                          )


# ## apply monkeypatches

# IVs

try:
    import py_vollib
except ImportError:
    raise ImportError("You must have py_vollib installed to use this library.")

#TODO override pkg_ref like black and black scholes as well

import py_vollib.black.implied_volatility

py_vollib.black.implied_volatility.implied_volatility = repr_partial(vectorized_implied_volatility_black, model="black")
update_wrapper(py_vollib.black.implied_volatility.implied_volatility, vectorized_implied_volatility_black)

import py_vollib.black_scholes.implied_volatility

py_vollib.black_scholes.implied_volatility.implied_volatility = repr_partial(vectorized_implied_volatility,
                                                                             model="black_scholes")
update_wrapper(py_vollib.black_scholes.implied_volatility.implied_volatility, vectorized_implied_volatility)

import py_vollib.black_scholes_merton.implied_volatility

py_vollib.black_scholes_merton.implied_volatility.implied_volatility = repr_partial(vectorized_implied_volatility,
                                                                                    model="black_scholes_merton")
update_wrapper(py_vollib.black_scholes_merton.implied_volatility.implied_volatility, vectorized_implied_volatility)

## Greeks

import py_vollib.black.greeks.numerical

py_vollib.black.greeks.numerical.delta = repr_partial(vectorized_delta, model="black")
py_vollib.black.greeks.numerical.gamma = repr_partial(vectorized_gamma, model="black")
py_vollib.black.greeks.numerical.rho = repr_partial(vectorized_rho, model="black")
py_vollib.black.greeks.numerical.theta = repr_partial(vectorized_theta, model="black")
py_vollib.black.greeks.numerical.vega = repr_partial(vectorized_vega, model="black")

import py_vollib.black_scholes.greeks.numerical

py_vollib.black_scholes.greeks.numerical.delta = repr_partial(vectorized_delta, model="black_scholes")
py_vollib.black_scholes.greeks.numerical.gamma = repr_partial(vectorized_gamma, model="black_scholes")
py_vollib.black_scholes.greeks.numerical.rho = repr_partial(vectorized_rho, model="black_scholes")
py_vollib.black_scholes.greeks.numerical.theta = repr_partial(vectorized_theta, model="black_scholes")
py_vollib.black_scholes.greeks.numerical.vega = repr_partial(vectorized_vega, model="black_scholes")

import py_vollib.black_scholes_merton.greeks.numerical

py_vollib.black_scholes_merton.greeks.numerical.delta = repr_partial(vectorized_delta, model="black_scholes_merton")
py_vollib.black_scholes_merton.greeks.numerical.gamma = repr_partial(vectorized_gamma, model="black_scholes_merton")
py_vollib.black_scholes_merton.greeks.numerical.rho = repr_partial(vectorized_rho, model="black_scholes_merton")
py_vollib.black_scholes_merton.greeks.numerical.theta = repr_partial(vectorized_theta, model="black_scholes_merton")
py_vollib.black_scholes_merton.greeks.numerical.vega = repr_partial(vectorized_vega, model="black_scholes_merton")

#TODO
## pkg_ref
import py_vollib.black
py_vollib.black.black = repr_partial(vectorized_black)

import py_vollib.black_scholes
py_vollib.black_scholes.black_scholes = repr_partial(vectorized_black_scholes)

import py_vollib.black_scholes_merton
py_vollib.black_scholes_merton.black_scholes_merton = repr_partial(vectorized_black_scholes_merton)



