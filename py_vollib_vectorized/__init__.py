#TODO AOT compilation and remove dependency on numba
import inspect

from functools import partial, update_wrapper

from .api import get_all_greeks, price_dataframe

from .implied_volatility import vectorized_implied_volatility_black, vectorized_implied_volatility
from .greeks import delta as vectorized_delta, \
    gamma as vectorized_gamma, rho as vectorized_rho, vega as vectorized_vega, theta as vectorized_theta
from .models import vectorized_black, vectorized_black_scholes, vectorized_black_scholes_merton

__version__ = '0.1'

__all__ = [
    'price_dataframe'
    'get_all_greeks',
    'vectorized_implied_volatility_black',
    'vectorized_implied_volatility',
    'vectorized_delta',
    'vectorized_gamma',
    'vectorized_rho',
    'vectorized_vega',
    'vectorized_theta',
    'vectorized_black',
    'vectorized_black_scholes',
    'vectorized_black_scholes_merton',
    '__version__'
]




class repr_partial(partial):
    def __repr__(self):
        return "Vectorized implementation of <{fn}({args}{kwargs})>".format(fn=self.func.__name__.replace("vectorized_", ""),
                                                          args=", ".join(repr(a) for a in self.args),
                                                          kwargs=", ".join(
                                                              [p.name for p in inspect.signature(self.func).parameters.values() if
                                                               p.kind == p.POSITIONAL_OR_KEYWORD] +
                                                              [str(k) + "=" + str(v) for k, v in self.keywords.items()]
                                                          )
                                                          )


# ## apply monkeypatches

# IVs

try:
    import py_vollib
except ImportError:
    raise ImportError("You must have py_vollib installed to use this library.")

import py_vollib.black.implied_volatility

py_vollib.black.implied_volatility.__dict__["implied_volatility"] = repr_partial(vectorized_implied_volatility_black,
                                                                          model="black")
update_wrapper(py_vollib.black.implied_volatility.implied_volatility, vectorized_implied_volatility_black)

import py_vollib.black_scholes.implied_volatility

py_vollib.black_scholes.implied_volatility.__dict__["implied_volatility"] = repr_partial(vectorized_implied_volatility,
                                                                                 model="black_scholes")
update_wrapper(py_vollib.black_scholes.implied_volatility.implied_volatility, vectorized_implied_volatility)

import py_vollib.black_scholes_merton.implied_volatility

py_vollib.black_scholes_merton.implied_volatility.__dict__["implied_volatility"] = repr_partial(vectorized_implied_volatility,
                                                                                         model="black_scholes_merton")
update_wrapper(py_vollib.black_scholes_merton.implied_volatility.implied_volatility, vectorized_implied_volatility)


## Greeks

import py_vollib.black.greeks.numerical

py_vollib.black.greeks.numerical.__dict__["delta"] = repr_partial(vectorized_delta, model="black")
update_wrapper(py_vollib.black.greeks.numerical.delta, vectorized_delta)
py_vollib.black.greeks.numerical.__dict__["gamma"] = repr_partial(vectorized_gamma, model="black")
update_wrapper(py_vollib.black.greeks.numerical.gamma, vectorized_gamma)
py_vollib.black.greeks.numerical.__dict__["rho"] = repr_partial(vectorized_rho, model="black")
update_wrapper(py_vollib.black.greeks.numerical.rho, vectorized_rho)
py_vollib.black.greeks.numerical.__dict__["theta"] = repr_partial(vectorized_theta, model="black")
update_wrapper(py_vollib.black.greeks.numerical.theta, vectorized_theta)
py_vollib.black.greeks.numerical.__dict__["vega"] = repr_partial(vectorized_vega, model="black")
update_wrapper(py_vollib.black.greeks.numerical.vega, vectorized_vega)

import py_vollib.black_scholes.greeks.numerical

py_vollib.black_scholes.greeks.numerical.__dict__["delta"] = repr_partial(vectorized_delta, model="black_scholes")
update_wrapper(py_vollib.black_scholes.greeks.numerical.delta, vectorized_delta)
py_vollib.black_scholes.greeks.numerical.__dict__["gamma"] = repr_partial(vectorized_gamma, model="black_scholes")
update_wrapper(py_vollib.black_scholes.greeks.numerical.gamma, vectorized_gamma)
py_vollib.black_scholes.greeks.numerical.__dict__["rho"] = repr_partial(vectorized_rho, model="black_scholes")
update_wrapper(py_vollib.black_scholes.greeks.numerical.rho, vectorized_rho)
py_vollib.black_scholes.greeks.numerical.__dict__["theta"] = repr_partial(vectorized_theta, model="black_scholes")
update_wrapper(py_vollib.black_scholes.greeks.numerical.theta, vectorized_theta)
py_vollib.black_scholes.greeks.numerical.__dict__["vega"] = repr_partial(vectorized_vega, model="black_scholes")
update_wrapper(py_vollib.black_scholes.greeks.numerical.vega, vectorized_vega)

import py_vollib.black_scholes_merton.greeks.numerical

py_vollib.black_scholes_merton.greeks.numerical.__dict__["delta"] = repr_partial(vectorized_delta, model="black_scholes_merton")
update_wrapper(py_vollib.black_scholes_merton.greeks.numerical.delta, vectorized_delta)
py_vollib.black_scholes_merton.greeks.numerical.__dict__["gamma"] = repr_partial(vectorized_gamma, model="black_scholes_merton")
update_wrapper(py_vollib.black_scholes_merton.greeks.numerical.gamma, vectorized_gamma)
py_vollib.black_scholes_merton.greeks.numerical.__dict__["rho"] = repr_partial(vectorized_rho, model="black_scholes_merton")
update_wrapper(py_vollib.black_scholes_merton.greeks.numerical.rho, vectorized_rho)
py_vollib.black_scholes_merton.greeks.numerical.__dict__["theta"] = repr_partial(vectorized_theta, model="black_scholes_merton")
update_wrapper(py_vollib.black_scholes_merton.greeks.numerical.theta, vectorized_theta)
py_vollib.black_scholes_merton.greeks.numerical.__dict__["vega"] = repr_partial(vectorized_vega, model="black_scholes_merton")
update_wrapper(py_vollib.black_scholes_merton.greeks.numerical.vega, vectorized_vega)

## models
import py_vollib.black

py_vollib.black.__dict__["black"] = repr_partial(vectorized_black)
update_wrapper(py_vollib.black.black, vectorized_black)

import py_vollib.black_scholes
py_vollib.black_scholes.__dict__["black_scholes"] = repr_partial(vectorized_black_scholes)
update_wrapper(py_vollib.black_scholes.black_scholes, vectorized_black_scholes)

import py_vollib.black_scholes_merton
py_vollib.black_scholes_merton.__dict__["black_scholes_merton"] = repr_partial(vectorized_black_scholes_merton)
update_wrapper(py_vollib.black_scholes_merton.black_scholes_merton, vectorized_black_scholes_merton)



