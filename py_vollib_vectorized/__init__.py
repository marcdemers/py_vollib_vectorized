
from functools import partial, update_wrapper

from .entrypoints import implied_volatility_vectorized_black, implied_volatility_vectorized
from .entrypoints import get_all_greeks
from .entrypoints import delta as vectorized_delta, \
    gamma as vectorized_gamma, rho as vectorized_rho, vega as vectorized_vega, theta as vectorized_theta

from .entrypoints import black_vectorized, black_scholes_vectorized, black_scholes_merton_vectorized

# TODO the readme file

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

#TODO override models like black and black scholes as well

import py_vollib.black.implied_volatility

py_vollib.black.implied_volatility.implied_volatility = repr_partial(implied_volatility_vectorized_black, model="black")
update_wrapper(py_vollib.black.implied_volatility.implied_volatility, implied_volatility_vectorized_black)

import py_vollib.black_scholes.implied_volatility

py_vollib.black_scholes.implied_volatility.implied_volatility = repr_partial(implied_volatility_vectorized,
                                                                             model="black_scholes")
update_wrapper(py_vollib.black_scholes.implied_volatility.implied_volatility, implied_volatility_vectorized)

import py_vollib.black_scholes_merton.implied_volatility

py_vollib.black_scholes_merton.implied_volatility.implied_volatility = repr_partial(implied_volatility_vectorized,
                                                                                    model="black_scholes_merton")
update_wrapper(py_vollib.black_scholes_merton.implied_volatility.implied_volatility, implied_volatility_vectorized)

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


## Models
import py_vollib.black.black
py_vollib.black.black = repr_partial(black_vectorized)

import py_vollib.black_scholes.black_scholes
py_vollib.black_scholes.black_scholes = repr_partial(black_scholes_vectorized)

import py_vollib.black_scholes_merton.black_scholes_merton
py_vollib.black_scholes_merton.black_scholes_merton = repr_partial(black_scholes_merton_vectorized)



