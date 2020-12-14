from py_vollib_vectorized.util.jit_helper import maybe_jit


import numpy as np
import py_lets_be_rational

from py_lets_be_rational.lets_be_rational import DENORMALIZATION_CUTOFF, DBL_MIN, DBL_MAX, \
    normalised_black_call, normalised_vega, DBL_EPSILON, SQRT_DBL_MAX

implied_volatility_maximum_iterations = 2


def forward_price(S, t, r):
    """Calculate the forward price of an underlying asset."""
    return S / np.exp(-r * t)


@maybe_jit()
def implied_volatility_from_a_transformed_rational_guess(price, F, K, T, q):
    return implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        price, F, K, T, q, implied_volatility_maximum_iterations)


@maybe_jit()
def implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
        prices, Fs, strikes, ttes, qs, N
):
    ivs = []
    for K, F, q, price, T in zip(strikes, Fs, qs, prices, ttes):
        intrinsic = np.abs(max(K - F if q < 0 else F - K, 0.0))

        x = np.log(F / K)

        # Map in-the-money to out-of-the-money
        if q * x > 0:
            price = np.abs(max(price - intrinsic, 0.0))
            q = -q
        iv = _unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(
            price / (np.sqrt(F) * np.sqrt(K)), x, q, N
        ) / np.sqrt(
            T
        )
        ivs.append(iv)
    return np.array(ivs)


@maybe_jit()
def _unchecked_normalised_implied_volatility_from_a_transformed_rational_guess_with_limited_iterations(beta, x, q, N):
    """
    See http://en.wikipedia.org/wiki/Householder%27s_method for a detailed explanation of the third order Householder iteration.
    Given the objective function g(s) whose root x such that 0 = g(s) we seek, iterate
        s_n+1  =  s_n  -  (g/g') · [ 1 - (g''/g')·(g/g') ] / [ 1 - (g/g')·( (g''/g') - (g'''/g')·(g/g')/6 ) ]
    Denoting  newton:=-(g/g'), halley:=(g''/g'), and hh3:=(g'''/g'), this reads
        s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ]
    NOTE that this function returns 0 when beta<intrinsic without any safety checks.
    :param beta:
    :type beta: float
    :param x:
    :type x: float
    :param q: q=±1
    :type q:
    :param N:
    :type N: int
    :return:
    :rtype: float
    """
    # Subtract intrinsic.
    if q * x > 0:
        beta = np.abs(max(beta - py_lets_be_rational.lets_be_rational._normalised_intrinsic(x, q), 0.))
        q = -q
    # Map puts to calls
    if q < 0:
        x = -x
        q = -q
    if beta <= 0:  # For negative or zero prices we return 0.
        return 0
    if beta < DENORMALIZATION_CUTOFF:  # For positive but denormalized (a.k.a. 'subnormal') prices, we return 0 since it would be impossible to converge to full machine accuracy anyway.
        return 0
    b_max = np.exp(0.5 * x)
    #     if beta >= b_max:
    #         raise AboveMaximumException
    iterations = 0
    direction_reversal_count = 0
    f = -DBL_MAX
    s = -DBL_MAX
    ds = s
    ds_previous = 0
    s_left = DBL_MIN
    s_right = DBL_MAX
    # The temptation is great to use the optimised form b_c = exp(x/2)/2-exp(-x/2)·Phi(sqrt(-2·x)) but that would require implementing all of the above types of round-off and over/underflow handling for this expression, too.
    s_c = np.sqrt(np.abs(2 * x))
    b_c = normalised_black_call(x, s_c)
    v_c = normalised_vega(x, s_c)
    # Four branches.
    if beta < b_c:
        s_l = s_c - b_c / v_c
        b_l = normalised_black_call(x, s_l)
        if beta < b_l:
            f_lower_map_l, d_f_lower_map_l_d_beta, d2_f_lower_map_l_d_beta2 = py_lets_be_rational.lets_be_rational._compute_f_lower_map_and_first_two_derivatives(
                x, s_l)
            r_ll = py_lets_be_rational.lets_be_rational.convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
                0., b_l, 0., f_lower_map_l, 1., d_f_lower_map_l_d_beta, d2_f_lower_map_l_d_beta2, True)
            f = py_lets_be_rational.lets_be_rational.rational_cubic_interpolation(beta, 0., b_l, 0., f_lower_map_l, 1.,
                                                                                  d_f_lower_map_l_d_beta, r_ll)
            if not (f > 0):  # This can happen due to roundoff truncation for extreme values such as |x|>500.
                # We switch to quadratic interpolation using f(0)≡0, f(b_l), and f'(0)≡1 to specify the quadratic.
                t = beta / b_l
                f = (f_lower_map_l * t + b_l * (1 - t)) * t

            s = py_lets_be_rational.lets_be_rational._inverse_f_lower_map(x, f)
            s_right = s_l
            #
            # In this branch, which comprises the lowest segment, the objective function is
            #     g(s) = 1/ln(b(x,s)) - 1/ln(beta)
            #          ≡ 1/ln(b(s)) - 1/ln(beta)
            # This makes
            #              g'               =   -b'/(b·ln(b)²)
            #              newton = -g/g'   =   (ln(beta)-ln(b))·ln(b)/ln(beta)·b/b'
            #              halley = g''/g'  =   b''/b'  -  b'/b·(1+2/ln(b))
            #              hh3    = g'''/g' =   b'''/b' +  2(b'/b)²·(1+3/ln(b)·(1+1/ln(b)))  -  3(b''/b)·(1+2/ln(b))
            #
            # The Householder(3) iteration is
            #     s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ]
            #
            while (iterations < N and np.abs(ds) > DBL_EPSILON * s):
                if ds * ds_previous < 0:
                    direction_reversal_count += 1
                if iterations > 0 and (3 == direction_reversal_count or not (s > s_left and s < s_right)):
                    # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
                    # NOTE that this can only really happen for very extreme values of |x|, such as |x| = |ln(F/K)| > 500.
                    s = 0.5 * (s_left + s_right)
                    if s_right - s_left <= DBL_EPSILON * s:
                        break
                    direction_reversal_count = 0
                    ds = 0
                ds_previous = ds
                b = py_lets_be_rational.lets_be_rational.normalised_black_call(x, s)
                bp = py_lets_be_rational.lets_be_rational.normalised_vega(x, s)
                if b > beta and s < s_right:
                    s_right = s
                elif b < beta and s > s_left:
                    s_left = s  # Tighten the bracket if applicable.
                if b <= 0 or bp <= 0:  # Numerical underflow. Switch to binary nesting for this iteration.
                    ds = 0.5 * (s_left + s_right) - s
                else:
                    ln_b = np.log(b)
                    ln_beta = np.log(beta)
                    bpob = bp / b
                    h = x / s
                    b_halley = h * h / s - s / 4
                    newton = (ln_beta - ln_b) * ln_b / ln_beta / bpob
                    halley = b_halley - bpob * (1 + 2 / ln_b)
                    b_hh3 = b_halley * b_halley - 3 * py_lets_be_rational.lets_be_rational._square(h / s) - 0.25
                    hh3 = b_hh3 + 2 * py_lets_be_rational.lets_be_rational._square(bpob) * (
                            1 + 3 / ln_b * (1 + 1 / ln_b)) - 3 * b_halley * bpob * (1 + 2 / ln_b)
                    ds = newton * py_lets_be_rational.lets_be_rational._householder_factor(newton, halley, hh3)
                ds = max(-0.5 * s, ds)
                s += ds
                iterations += 1
            return s
        else:
            v_l = py_lets_be_rational.lets_be_rational.normalised_vega(x, s_l)
            r_lm = py_lets_be_rational.lets_be_rational.convex_rational_cubic_control_parameter_to_fit_second_derivative_at_right_side(
                b_l, b_c, s_l, s_c, 1 / v_l, 1 / v_c, 0.0, False)
            s = py_lets_be_rational.lets_be_rational.rational_cubic_interpolation(beta, b_l, b_c, s_l, s_c, 1 / v_l,
                                                                                  1 / v_c, r_lm)
            s_left = s_l
            s_right = s_c
    else:
        s_h = s_c + (b_max - b_c) / v_c if v_c > DBL_MIN else s_c
        b_h = py_lets_be_rational.lets_be_rational.normalised_black_call(x, s_h)
        if beta <= b_h:
            v_h = py_lets_be_rational.lets_be_rational.normalised_vega(x, s_h)
            r_hm = py_lets_be_rational.lets_be_rational.convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
                b_c, b_h, s_c, s_h, 1 / v_c, 1 / v_h, 0.0, False)
            s = py_lets_be_rational.lets_be_rational.rational_cubic_interpolation(beta, b_c, b_h, s_c, s_h, 1 / v_c,
                                                                                  1 / v_h, r_hm)
            s_left = s_c
            s_right = s_h
        else:
            f_upper_map_h, d_f_upper_map_h_d_beta, d2_f_upper_map_h_d_beta2 = py_lets_be_rational.lets_be_rational._compute_f_upper_map_and_first_two_derivatives(
                x, s_h)
            if d2_f_upper_map_h_d_beta2 > -SQRT_DBL_MAX < SQRT_DBL_MAX:
                r_hh = py_lets_be_rational.lets_be_rational.convex_rational_cubic_control_parameter_to_fit_second_derivative_at_left_side(
                    b_h, b_max, f_upper_map_h, 0., d_f_upper_map_h_d_beta, -0.5, d2_f_upper_map_h_d_beta2, True)
                f = py_lets_be_rational.lets_be_rational.rational_cubic_interpolation(beta, b_h, b_max, f_upper_map_h,
                                                                                      0., d_f_upper_map_h_d_beta, -0.5,
                                                                                      r_hh)
            if f <= 0:
                h = b_max - b_h
                t = (beta - b_h) / h
                f = (f_upper_map_h * (1 - t) + 0.5 * h * t) * (
                        1 - t)  # We switch to quadratic interpolation using f(b_h), f(b_max)≡0, and f'(b_max)≡-1/2 to specify the quadratic.
            s = py_lets_be_rational.lets_be_rational._inverse_f_upper_map(f)
            s_left = s_h
            if beta > 0.5 * b_max:  # Else we better drop through and let the objective function be g(s) = b(x,s)-beta.
                #
                # In this branch, which comprises the upper segment, the objective function is
                #     g(s) = ln(b_max-beta)-ln(b_max-b(x,s))
                #          ≡ ln((b_max-beta)/(b_max-b(s)))
                # This makes
                #              g'               =   b'/(b_max-b)
                #              newton = -g/g'   =   ln((b_max-b)/(b_max-beta))·(b_max-b)/b'
                #              halley = g''/g'  =   b''/b'  +  b'/(b_max-b)
                #              hh3    = g'''/g' =   b'''/b' +  g'·(2g'+3b''/b')
                # and the iteration is
                #     s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ].
                #
                while iterations < N and np.abs(ds) > DBL_EPSILON * s:
                    if ds * ds_previous < 0:
                        direction_reversal_count += 1
                    if iterations > 0 and (3 == direction_reversal_count or not (s > s_left and s < s_right)):
                        # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
                        # NOTE that this can only really happen for very extreme values of |x|, such as |x| = |ln(F/K)| > 500.
                        s = 0.5 * (s_left + s_right)
                    if (s_right - s_left <= DBL_EPSILON * s):
                        break
                    direction_reversal_count = 0
                    ds = 0
                    ds_previous = ds
                    b = py_lets_be_rational.lets_be_rational.normalised_black_call(x, s)
                    bp = py_lets_be_rational.lets_be_rational.normalised_vega(x, s)
                    if b > beta and s < s_right:
                        s_right = s
                    elif b < beta and s > s_left:
                        s_left = s  # Tighten the bracket if applicable.
                    if b >= b_max or bp <= DBL_MIN:  # Numerical underflow. Switch to binary nesting for this iteration.
                        ds = 0.5 * (s_left + s_right) - s
                    else:
                        b_max_minus_b = b_max - b
                        g = np.log((b_max - beta) / b_max_minus_b)
                        gp = bp / b_max_minus_b
                        b_halley = py_lets_be_rational.lets_be_rational._square(x / s) / s - s / 4
                        b_hh3 = b_halley * b_halley - 3 * py_lets_be_rational.lets_be_rational._square(
                            x / (s * s)) - 0.25
                        newton = -g / gp
                        halley = b_halley + gp
                        hh3 = b_hh3 + gp * (2 * gp + 3 * b_halley)
                        ds = newton * py_lets_be_rational.lets_be_rational._householder_factor(newton, halley, hh3)
                    ds = max(-0.5 * s, ds)
                    s += ds
                    iterations += 1
                return s
    # In this branch, which comprises the two middle segments, the objective function is g(s) = b(x,s)-beta, or g(s) = b(s) - beta, for short.
    # This makes
    #              newton = -g/g'   =  -(b-beta)/b'
    #              halley = g''/g'  =    b''/b'    =  x²/s³-s/4
    #              hh3    = g'''/g' =    b'''/b'   =  halley² - 3·(x/s²)² - 1/4
    # and the iteration is
    #     s_n+1  =  s_n  +  newton · [ 1 + halley·newton/2 ] / [ 1 + newton·( halley + hh3·newton/6 ) ].
    #
    while iterations < N and np.abs(ds) > DBL_EPSILON * s:
        if ds * ds_previous < 0:
            direction_reversal_count += 1
        if iterations > 0 and (3 == direction_reversal_count or not (s > s_left and s < s_right)):
            # If looping inefficently, or the forecast step takes us outside the bracket, or onto its edges, switch to binary nesting.
            # NOTE that this can only really happen for very extreme values of |x|, such as |x| = |ln(F/K)| > 500.
            s = 0.5 * (s_left + s_right)
            if s_right - s_left <= DBL_EPSILON * s:
                break
            direction_reversal_count = 0
            ds = 0
        ds_previous = ds
        b = py_lets_be_rational.lets_be_rational.normalised_black_call(x, s)
        bp = py_lets_be_rational.lets_be_rational.normalised_vega(x, s)
        if b > beta and s < s_right:
            s_right = s
        elif b < beta and s > s_left:
            s_left = s  # Tighten the bracket if applicable.
        newton = (beta - b) / bp
        halley = py_lets_be_rational.lets_be_rational._square(x / s) / s - s / 4
        hh3 = halley * halley - 3 * py_lets_be_rational.lets_be_rational._square(x / (s * s)) - 0.25
        ds = max(-0.5 * s, newton * py_lets_be_rational.lets_be_rational._householder_factor(newton, halley, hh3))
        s += ds
        iterations += 1
    return s
