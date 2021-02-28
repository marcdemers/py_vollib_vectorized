import numpy as np

from py_vollib_vectorized.util.jit_helper import maybe_jit
from ._model_calls import black, black_scholes, black_scholes_merton

dS = .01

#### BLACK

@maybe_jit()
def numerical_delta_black(flags, Fs, Ks, ts, rs, sigmas):
    deltas = []
    for flag, F, K, t, r, sigma in zip(flags, Fs, Ks, ts, rs, sigmas):
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
            delta = (black(F+dS, K, sigma, t, flag) - black(F - dS, K, sigma, t, flag)) / (
                    2 * dS)
        deltas.append(delta)
    return deltas


@maybe_jit()
def numerical_theta_black(flags, Fs, Ks, ts, rs, sigmas):
    thetas = []
    for flag, F, K, t, r, sigma in zip(flags, Fs, Ks, ts, rs, sigmas):
        if t <= 1. / 365.:
            theta = black(F, K, sigma, 0.00001, flag) - black(F, K, sigma, t, flag)
        else:
            theta = black(F, K, sigma, t - 1./365., flag) - black(F, K, sigma, t, flag)
        thetas.append(theta)
    return thetas


@maybe_jit()
def numerical_vega_black(flags, Fs, Ks, ts, rs, sigmas):
    vegas = []

    for flag, F, K, t, r, sigma in zip(flags, Fs, Ks, ts, rs, sigmas):
        vega = (black(F, K, sigma + 0.01, t, flag) - black(F, K, sigma - 0.01, t, flag)) / 2.
        vegas.append(vega)
    return vegas


@maybe_jit()
def numerical_rho_black(flags, Fs, Ks, ts, rs, sigmas):
    rhos = []

    for flag, F, K, t, r, sigma in zip(flags, Fs, Ks, ts, rs, sigmas):
        black(F, K, sigma. t, flag)
        rho = (black(flag, F, K, t, r + 0.01, sigma) - black(flag, F, K, t, r - 0.01, sigma)) / 2.
        rhos.append(rho)

    return rhos


@maybe_jit()
def numerical_gamma_black(flags, Fs, Ks, ts, rs, sigmas):
    gammas = []

    for flag, F, K, t, r, sigma in zip(flags, Fs, Ks, ts, rs, sigmas):
        if t == 0:
            gamma = np.inf if F == K else 0.0
        else:
            gamma = (black(flag, F + dS, K, t, r, sigma) - 2. * black(flag, F, K, t, r, sigma) + \
                     black(flag, F - dS, K, t, r, sigma)) / dS ** 2.

        gammas.append(gamma)
    return gammas

#### BLACK SCHOLES

@maybe_jit()
def numerical_delta_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    deltas = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
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
        deltas.append(delta)
    return deltas


@maybe_jit()
def numerical_theta_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    thetas = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t <= 1. / 365.:
            theta = black_scholes(flag, S, K, 0.00001, r, sigma) - black_scholes(flag, S, K, t, r, sigma)
        else:
            theta = black_scholes(flag, S, K, t - 1. / 365., r, sigma) - black_scholes(flag, S, K, t, r, sigma)
        thetas.append(theta)
    return thetas


@maybe_jit()
def numerical_vega_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    vegas = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        vega = (black_scholes(flag, S, K, t, r, sigma + 0.01) - black_scholes(flag, S, K, t, r, sigma - 0.01)) / 2.
        vegas.append(vega)
    return vegas


@maybe_jit()
def numerical_rho_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    rhos = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        rho = (black_scholes(flag, S, K, t, r + 0.01, sigma) - black_scholes(flag, S, K, t, r - 0.01, sigma)) / 2.
        rhos.append(rho)

    return rhos


@maybe_jit()
def numerical_gamma_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    gammas = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t == 0:
            gamma = np.inf if S == K else 0.0
        else:
            gamma = (black_scholes(flag, S + dS, K, t, r, sigma) - 2. * black_scholes(flag, S, K, t, r, sigma) + \
                     black_scholes(flag, S - dS, K, t, r, sigma)) / dS ** 2.

        gammas.append(gamma)
    return gammas


### BLACK SCHOLES MERTON


@maybe_jit()
def numerical_delta_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    deltas = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
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
        deltas.append(delta)
    return deltas


@maybe_jit()
def numerical_theta_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    thetas = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t <= 1. / 365.:
            theta = black_scholes_merton(flag, S, K, 0.00001, r, sigma, r - b) - black_scholes_merton(flag, S, K, t, r,
                                                                                                      sigma,
                                                                                                      r - b)
        else:
            theta = black_scholes_merton(flag, S, K, t - 1. / 365., r, sigma, r - b) - black_scholes_merton(flag, S, K,
                                                                                                            t,
                                                                                                            r, sigma,
                                                                                                            r - b)
        thetas.append(theta)
    return thetas


@maybe_jit()
def numerical_vega_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    vegas = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        vega = (black_scholes_merton(flag, S, K, t, r, sigma + 0.01, r - b) - black_scholes_merton(flag, S, K, t, r,
                                                                                                   sigma - 0.01,
                                                                                                   r - b)) / 2.
        vegas.append(vega)
    return vegas


@maybe_jit()
def numerical_rho_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    rhos = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        rho = (black_scholes_merton(flag, S, K, t, r + 0.01, sigma, r - b + 0.01) - black_scholes_merton(flag, S, K, t,
                                                                                                         r - 0.01,
                                                                                                         sigma,
                                                                                                         r - b - 0.01)) / 2.
        rhos.append(rho)

    return rhos


@maybe_jit()
def numerical_gamma_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    gammas = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t == 0:
            gamma = np.inf if S == K else 0.0
        else:
            gamma = (black_scholes_merton(flag, S + dS, K, t, r, sigma, r - b) - 2. * black_scholes_merton(flag, S, K,
                                                                                                           t, r,
                                                                                                           sigma,
                                                                                                           r - b) + \
                     black_scholes_merton(flag, S - dS, K, t, r, sigma, r - b)) / dS ** 2.

        gammas.append(gamma)
    return gammas
