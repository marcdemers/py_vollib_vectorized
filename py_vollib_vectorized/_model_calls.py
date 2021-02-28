import numpy as np
from py_lets_be_rational.constants import *
from py_lets_be_rational.rationalcubic import *

from py_vollib_vectorized.util.greeks_helpers import _asymptotic_expansion_of_normalized_black_call, _normalised_intrinsic_call, \
    _small_t_expansion_of_normalized_black_call, _normalized_black_call_using_norm_cdf, \
    _normalised_black_call_using_erfcx

implied_volatility_maximum_iterations = 2
asymptotic_expansion_accuracy_threshold = -10
small_t_expansion_of_normalized_black_threshold = 2 * SIXTEENTH_ROOT_DBL_EPSILON

dS = .01

from py_vollib_vectorized.util.jit_helper import maybe_jit


@maybe_jit()
def normalised_black_call(x, s):
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
def normalised_black(x, s, q):
    return normalised_black_call(-x if q < 0 else x, s)  # Reciprocal-strike call-put equivalence


@maybe_jit()
def undiscounted_black(F, K, sigma, t, flag):
    q = flag
    return black(F, K, sigma, t, q)


@maybe_jit()
def black_scholes(flag, S, K, t, r, sigma):
    deflater = np.exp(-r * t)
    F = S / deflater
    return undiscounted_black(F, K, sigma, t, flag) * deflater


@maybe_jit()
def black_scholes_merton(flag, S, K, t, r, sigma, q):
    F = S * np.exp((r - q) * t)
    deflater = np.exp(-r * t)
    return black(F, K, sigma, t, flag) * deflater


@maybe_jit()
def black(F, K, sigma, T, q):
    intrinsic = np.abs(np.maximum((K - F if q < 0 else F - K), 0.0))
    # Map in-the-money to out-of-the-money
    if q * (F - K) > 0:
        return intrinsic + black(F, K, sigma, T, -q)
    return np.maximum(intrinsic,
                      (np.sqrt(F) * np.sqrt(K)) * normalised_black(np.log(F / K), sigma * np.sqrt(T), q)
                      )


@maybe_jit()
def _black_scholes_vectorized_call(flags, Ss, Ks, ts, rs, sigmas):
    prices = []
    for q, S, K, t, r, sigma in zip(flags, Ss, Ks, ts, rs, sigmas):
        prices.append(black_scholes(q, S, K, t, r, sigma))
    return prices


@maybe_jit()
def _black_vectorized_call(Fs, Ks, sigmas, ts, flag):
    prices = []
    for F, K, sigma, T, q in zip(Fs, Ks, sigmas, ts, flag):
        prices.append(black(F, K, sigma, T, q))
    return prices


@maybe_jit()
def _black_scholes_merton_vectorized_call(flags, Ss, Ks, ts, rs, sigmas, qs):
    prices = []
    for f, S, K, t, r, sigma, q in zip(flags, Ss, Ks, ts, rs, sigmas, qs):
        prices.append(black_scholes_merton(f, S, K, t, r, sigma, q))
    return prices
