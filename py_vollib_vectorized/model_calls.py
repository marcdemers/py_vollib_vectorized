import numba
import numpy as np
from py_lets_be_rational.constants import *
from py_lets_be_rational.rationalcubic import *

from .greeks_helpers import _asymptotic_expansion_of_normalized_black_call, _normalised_intrinsic_call, \
    _small_t_expansion_of_normalized_black_call, _normalized_black_call_using_norm_cdf, \
    _normalised_black_call_using_erfcx

implied_volatility_maximum_iterations = 2
asymptotic_expansion_accuracy_threshold = -10
small_t_expansion_of_normalized_black_threshold = 2 * SIXTEENTH_ROOT_DBL_EPSILON

dS = .01

from .jit_helper import maybe_jit


@maybe_jit()
def normalised_black_call(x, s):
    """
    :param x:
    :type x: float
    :param s:
    :type x: float
    :return:
    :rtype: float
    """
    if x > 0:
        return _normalised_intrinsic_call(x) + normalised_black_call(-x, s)
    ax = np.abs(x)
    if s <= ax * DENORMALIZATION_CUTOFF:
        return _normalised_intrinsic_call(x)
    # Denote h := x/s and t := s/2. We evaluate the condition |h|>|η|, i.e., h<η  &&  t < τ+|h|-|η|  avoiding any
    # divisions by s , where η = asymptotic_expansion_accuracy_threshold  and τ =
    # small_t_expansion_of_normalized_black_threshold .
    if x < s * asymptotic_expansion_accuracy_threshold and 0.5 * s * s + x < s * (
            small_t_expansion_of_normalized_black_threshold + asymptotic_expansion_accuracy_threshold):
        # Region 1.
        return _asymptotic_expansion_of_normalized_black_call(x / s, 0.5 * s)
    if 0.5 * s < small_t_expansion_of_normalized_black_threshold:
        # Region 2.
        return _small_t_expansion_of_normalized_black_call(x / s, 0.5 * s)
    # When b is more than, say, about 85% of b_max=exp(x/2), then b is dominated by the first of the two terms in the
    #  Black formula, and we retain more accuracy by not attempting to combine the two terms in any way. We evaluate
    # the condition h+t>0.85  avoiding any divisions by s.
    if x + 0.5 * s * s > s * 0.85:
        # Region 3.
        return _normalized_black_call_using_norm_cdf(x, s)
    # Region 4.
    return _normalised_black_call_using_erfcx(x / s, 0.5 * s)


@maybe_jit()
def normalised_black(x, s, q) -> float:
    """
    :param x:
    :type x: float
    :param s:
    :type s: float
    :param q: q=±1
    :type q: float
    :return:
    :rtype: float
    """
    out = normalised_black_call(-x if q < 0 else x, s)  # Reciprocal-strike call-put equivalence
    return out


@maybe_jit()
def undiscounted_black(F, K, sigma, t, flag) -> np.float64:
    """Calculate the **undiscounted** Black option price.
    :param F: underlying futures price
    :type F: float
    :param K: strike price
    :type K: float
    :param sigma: annualized standard deviation, or volatility
    :type sigma: float
    :param t: time to expiration in years
    :type t: float
    >>> F = 100
    >>> K = 100
    >>> sigma = .2
    >>> flag = 'c'
    >>> t = .5
    # >>> undiscounted_black(F, K, sigma, t, flag)
    5.637197779701664

    """

    q = flag
    # F = float(F)
    # K = float(K)
    # sigma = float(sigma)
    # t = float(t)
    out = black(F, K, sigma, t, q)
    return out


# @njit(cache=True)
@maybe_jit()
def black_scholes(flag, S, K, t, r, sigma):
    """Return the Black-Scholes option price.
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


    >>> c = black_scholes('c',100,90,.5,.01,.2)
    >>> abs(c - 12.111581435) < .000001
    True
    >>> p = black_scholes('p',100,90,.5,.01,.2)
    >>> abs(p - 1.66270456231) < .000001
    True
    """

    deflater = np.exp(-r * t)
    F = S / deflater

    # flag=float(flag)
    # F = float(F)
    # K = float(K)
    # sigma = float(sigma)
    # t = float(t)

    out = undiscounted_black(F, K, sigma, t, flag)
    return out * deflater


@maybe_jit()
def black_scholes_merton(flag, S, K, t, r, sigma, q):
    """Return the Black-Scholes-Merton option price.
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
    :param q: annualized continuous dividend rate
    :type q: float
    From Espen Haug, The Complete Guide To Option Pricing Formulas
    Page 4
    >>> S=100
    >>> K=95
    >>> q=.05
    >>> t = 0.5
    >>> r = 0.1
    >>> sigma = 0.2
    >>> p_published_value = 2.4648
    >>> p_calc = black_scholes_merton('p', S, K, t, r, sigma, q)
    >>> abs(p_published_value - p_calc) < 0.0001
    True
    """

    F = S * np.exp((r - q) * t)
    deflater = np.exp(-r * t)
    out = black(F, K, sigma, t, flag)
    return out * deflater


@maybe_jit()
def black(F: float, K: float, sigma: float, T: float, q: float) -> float:
    """
    :param F:
    :type F: float
    :param K:
    :type K: float
    :param sigma:
    :type sigma: float
    :param T:
    :type T: float
    :param q: q=±1
    :type q: float
    :return:
    :rtype: float
    """
    intrinsic = np.abs(np.maximum((K - F if q < 0 else F - K), 0.0))
    # Map in-the-money to out-of-the-money
    if q * (F - K) > 0:
        return intrinsic + black(F, K, sigma, T, -q)
    return np.maximum(intrinsic,
                      (np.sqrt(F) * np.sqrt(K)) * normalised_black(np.log(F / K), sigma * np.sqrt(T), q)
                      )
