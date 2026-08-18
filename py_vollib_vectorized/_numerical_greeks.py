import numpy as np

from py_vollib_vectorized.util.jit_helper import maybe_jit
from ._model_calls import black, black_scholes, black_scholes_merton, discounted_black

dS = .01

#### BLACK
# 注意: Black-76 模型 b=0 (cost of carry 持有成本), 已内嵌于 discounted_black() 的折现逻辑中。
# bs 参数为签名一致性占位, 不参与计算, 使 greeks.py 可统一传递 b 参数。

@maybe_jit()
def numerical_delta_black(flags, Fs, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    deltas = np.empty(n)
    for i, (flag, F, K, t, r, sigma) in enumerate(zip(flags, Fs, Ks, ts, rs, sigmas)):
        if t == 0.0:
            if F == K:
                if flag > 0:  # call option
                    delta = 0.5
                if flag < 0:  # put option
                    delta = -0.5
            elif F > K:
                if flag > 0:  # call option
                    delta = 1.0
                if flag < 0:  # put option
                    delta = 0.0
            else:
                if flag > 0:  # call option
                    delta = 0.0
                if flag < 0:  # put option
                    delta = -1.0
        else:
            delta = (discounted_black(F+dS, K, sigma, t, flag, r) - discounted_black(F - dS, K, sigma, t, flag, r)) / (
                    2 * dS)
        deltas[i] = delta
    return deltas


@maybe_jit()
def numerical_theta_black(flags, Fs, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    thetas = np.empty(n)
    for i, (flag, F, K, t, r, sigma) in enumerate(zip(flags, Fs, Ks, ts, rs, sigmas)):
        if t <= 1. / 365.:
            theta = discounted_black(F, K, sigma, 0.00001, flag, r) - discounted_black(F, K, sigma, t, flag, r)
        else:
            theta = discounted_black(F, K, sigma, t - 1./365., flag, r) - discounted_black(F, K, sigma, t, flag, r)
        thetas[i] = theta
    return thetas


@maybe_jit()
def numerical_vega_black(flags, Fs, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    vegas = np.empty(n)
    for i, (flag, F, K, t, r, sigma) in enumerate(zip(flags, Fs, Ks, ts, rs, sigmas)):
        vega = (discounted_black(F, K, sigma + 0.01, t, flag, r) - discounted_black(F, K, sigma - 0.01, t, flag, r)) / 2.
        vegas[i] = vega
    return vegas


@maybe_jit()
def numerical_rho_black(flags, Fs, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    rhos = np.empty(n)
    for i, (flag, F, K, t, r, sigma) in enumerate(zip(flags, Fs, Ks, ts, rs, sigmas)):
        # black() 不含 r 参数，rho 通过 deflater = exp(-r*t) 体现
        undiscounted = black(F, K, sigma, t, flag)
        rho = (undiscounted * np.exp(-(r + 0.01) * t) -
               undiscounted * np.exp(-(r - 0.01) * t)) / 2.
        rhos[i] = rho
    return rhos


@maybe_jit()
def numerical_gamma_black(flags, Fs, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    gammas = np.empty(n)
    for i, (flag, F, K, t, r, sigma) in enumerate(zip(flags, Fs, Ks, ts, rs, sigmas)):
        if t == 0:
            gamma = np.inf if F == K else 0.0
        else:
            gamma = (discounted_black(F + dS, K, sigma, t, flag, r) - 2. * discounted_black(F, K, sigma, t, flag, r) + \
                     discounted_black(F - dS, K, sigma, t, flag, r)) / dS ** 2.
        gammas[i] = gamma
    return gammas

#### BLACK SCHOLES

@maybe_jit()
def numerical_delta_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    deltas = np.empty(n)
    for i, (flag, S, K, t, r, sigma, b) in enumerate(zip(flags, Ss, Ks, ts, rs, sigmas, bs)):
        if t == 0.0:
            if S == K:
                if flag > 0:  # call option
                    delta = 0.5
                if flag < 0:  # put option
                    delta = -0.5
            elif S > K:
                if flag > 0:  # call option
                    delta = 1.0
                if flag < 0:  # put option
                    delta = 0.0
            else:
                if flag > 0:  # call option
                    delta = 0.0
                if flag < 0:  # put option
                    delta = -1.0
        else:
            delta = (black_scholes(flag, S + dS, K, t, r, sigma) - black_scholes(flag, S - dS, K, t, r, sigma)) / (
                    2 * dS)
        deltas[i] = delta
    return deltas


@maybe_jit()
def numerical_theta_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    thetas = np.empty(n)
    for i, (flag, S, K, t, r, sigma, b) in enumerate(zip(flags, Ss, Ks, ts, rs, sigmas, bs)):
        if t <= 1. / 365.:
            theta = black_scholes(flag, S, K, 0.00001, r, sigma) - black_scholes(flag, S, K, t, r, sigma)
        else:
            theta = black_scholes(flag, S, K, t - 1. / 365., r, sigma) - black_scholes(flag, S, K, t, r, sigma)
        thetas[i] = theta
    return thetas


@maybe_jit()
def numerical_vega_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    vegas = np.empty(n)
    for i, (flag, S, K, t, r, sigma, b) in enumerate(zip(flags, Ss, Ks, ts, rs, sigmas, bs)):
        vega = (black_scholes(flag, S, K, t, r, sigma + 0.01) - black_scholes(flag, S, K, t, r, sigma - 0.01)) / 2.
        vegas[i] = vega
    return vegas


@maybe_jit()
def numerical_rho_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    rhos = np.empty(n)
    for i, (flag, S, K, t, r, sigma, b) in enumerate(zip(flags, Ss, Ks, ts, rs, sigmas, bs)):
        rho = (black_scholes(flag, S, K, t, r + 0.01, sigma) - black_scholes(flag, S, K, t, r - 0.01, sigma)) / 2.
        rhos[i] = rho
    return rhos


@maybe_jit()
def numerical_gamma_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    gammas = np.empty(n)
    for i, (flag, S, K, t, r, sigma, b) in enumerate(zip(flags, Ss, Ks, ts, rs, sigmas, bs)):
        if t == 0:
            gamma = np.inf if S == K else 0.0
        else:
            gamma = (black_scholes(flag, S + dS, K, t, r, sigma) - 2. * black_scholes(flag, S, K, t, r, sigma) + \
                     black_scholes(flag, S - dS, K, t, r, sigma)) / dS ** 2.
        gammas[i] = gamma
    return gammas


### BLACK SCHOLES MERTON


@maybe_jit()
def numerical_delta_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    deltas = np.empty(n)
    for i, (flag, S, K, t, r, sigma, b) in enumerate(zip(flags, Ss, Ks, ts, rs, sigmas, bs)):
        if t == 0.0:
            if S == K:
                if flag > 0:  # call option
                    delta = 0.5
                if flag < 0:  # put option
                    delta = -0.5
            elif S > K:
                if flag > 0:  # call option
                    delta = 1.0
                if flag < 0:  # put option
                    delta = 0.0
            else:
                if flag > 0:  # call option
                    delta = 0.0
                if flag < 0:  # put option
                    delta = -1.0
        else:
            delta = (black_scholes_merton(flag, S + dS, K, t, r, sigma, r - b) - black_scholes_merton(flag, S - dS, K,
                                                                                                      t, r,
                                                                                                      sigma,
                                                                                                      r - b)) / (
                            2 * dS)
        deltas[i] = delta
    return deltas


@maybe_jit()
def numerical_theta_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    thetas = np.empty(n)
    for i, (flag, S, K, t, r, sigma, b) in enumerate(zip(flags, Ss, Ks, ts, rs, sigmas, bs)):
        if t <= 1. / 365.:
            theta = black_scholes_merton(flag, S, K, 0.00001, r, sigma, r - b) - black_scholes_merton(flag, S, K, t, r,
                                                                                                      sigma,
                                                                                                      r - b)
        else:
            theta = black_scholes_merton(flag, S, K, t - 1. / 365., r, sigma, r - b) - black_scholes_merton(flag, S, K,
                                                                                                            t,
                                                                                                            r, sigma,
                                                                                                            r - b)
        thetas[i] = theta
    return thetas


@maybe_jit()
def numerical_vega_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    vegas = np.empty(n)
    for i, (flag, S, K, t, r, sigma, b) in enumerate(zip(flags, Ss, Ks, ts, rs, sigmas, bs)):
        vega = (black_scholes_merton(flag, S, K, t, r, sigma + 0.01, r - b) - black_scholes_merton(flag, S, K, t, r,
                                                                                                   sigma - 0.01,
                                                                                                   r - b)) / 2.
        vegas[i] = vega
    return vegas


@maybe_jit()
def numerical_rho_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    rhos = np.empty(n)
    for i, (flag, S, K, t, r, sigma, b) in enumerate(zip(flags, Ss, Ks, ts, rs, sigmas, bs)):
        rho = (black_scholes_merton(flag, S, K, t, r + 0.01, sigma, r - b) - black_scholes_merton(flag, S, K, t,
                                                                                                   r - 0.01,
                                                                                                   sigma,
                                                                                                   r - b)) / 2.
        rhos[i] = rho
    return rhos


@maybe_jit()
def numerical_gamma_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    n = len(flags)
    gammas = np.empty(n)
    for i, (flag, S, K, t, r, sigma, b) in enumerate(zip(flags, Ss, Ks, ts, rs, sigmas, bs)):
        if t == 0:
            gamma = np.inf if S == K else 0.0
        else:
            gamma = (black_scholes_merton(flag, S + dS, K, t, r, sigma, r - b) - 2. * black_scholes_merton(flag, S, K,
                                                                                                           t, r,
                                                                                                           sigma,
                                                                                                           r - b) + \
                     black_scholes_merton(flag, S - dS, K, t, r, sigma, r - b)) / dS ** 2.
        gammas[i] = gamma
    return gammas
