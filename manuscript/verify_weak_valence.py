"""Reproduce the weak-valence comparison at the full first-election mandates."""

from functools import cache
from math import exp, inf, pi, sqrt

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad_vec


# Entries are the incumbent, challenger bloc, incumbent bloc, and swing voter.
BASELINES = np.array([2.0, 0.0, 2.0, 1.005])
WEIGHTS = np.array([2.0, 0.0, 0.0, 1.0])
SECOND_MONITORING = 0.05
DISCOUNT = 0.25
FIRST_VALENCE = np.array([-0.5, 0.0, 0.5])
VALENCE_PROBABILITIES = np.array([0.1, 0.8, 0.1])


def flow(party_baseline, mandate, a, b, monitoring):
    mean_policy = party_baseline + mandate * a
    response_payoff = 2 * WEIGHTS * mandate - mandate**2
    return (2 * WEIGHTS * a * mean_policy - (mean_policy - BASELINES)**2
            + monitoring * b**2 * response_payoff)


def winning_offer(a, v):
    """Largest terminal y = 1 + ell winning the policy ballot, if feasible."""
    if 4 * v + 4.02 * a + 0.02 <= 0:
        return 2.0
    discriminant = (2.01 * a)**2 - 0.08 * v
    if discriminant < 0:
        return None
    lower = (-2.01 * a - sqrt(discriminant)) / (2 * v)
    upper = (-2.01 * a + sqrt(discriminant)) / (2 * v)
    if max(lower, 1.0) <= min(upper, 2.0):
        return min(upper, 2.0)
    return None


def terminal(a, state, use_record):
    b = 0.5 + 25 * state * a
    v = a**2 + SECOND_MONITORING * b**2
    y = winning_offer(a, v) if use_record else None
    if y is None:
        democracy = flow(2, 1, a, b, SECOND_MONITORING)
    else:
        democracy = flow(0, 1 - y, a, b, SECOND_MONITORING)
    autocracy = flow(2, 2, a, b, SECOND_MONITORING)
    return np.r_[democracy, autocracy, float(y is not None)]


def switch_points(state):
    a = Polynomial([0, 1])
    v = a**2 + SECOND_MONITORING * (0.5 + 25 * state * a)**2
    boundaries = [v * y**2 + 2.01 * a * y + 0.02 for y in (1, 2)]
    boundaries.append((2.01 * a)**2 - 0.08 * v)
    return sorted({root.real for expression in boundaries for root in expression.roots()
                   if abs(root.imag) < 1e-12})


@cache
def trial(policy, state, use_record):
    if policy == 0:
        return terminal(0.0, state, use_record)
    sd = 0.01 * abs(policy) / sqrt(0.26 * policy**2 + 0.0625)
    boundaries = [point / sd for point in switch_points(state)] if use_record else []
    endpoints = [-inf, *boundaries, inf]

    def integrand(z):
        return terminal(sd * z, state, use_record) * exp(-z**2 / 2) / sqrt(2 * pi)

    return sum(quad_vec(integrand, lo, hi, epsabs=1e-12, epsrel=1e-12)[0]
               for lo, hi in zip(endpoints, endpoints[1:]))


def continuation(baseline, mandate, monitoring, use_record):
    value = np.zeros(9)
    for state in (-1, 1):
        no_news = trial(baseline, state, use_record)
        news = trial(baseline + 0.5 * mandate * state, state, use_record)
        value += 0.5 * ((1 - monitoring) * no_news + monitoring * news)
    return value


def evaluate(monitoring, use_record):
    left = continuation(0, -1, monitoring, use_record)
    right = continuation(2, 1, monitoring, use_record)
    current_left = flow(0, -1, 0, 0.5, monitoring)
    current_right = flow(2, 1, 0, 0.5, monitoring)
    concede = current_left + DISCOUNT * left[:4]
    entrench = current_right + DISCOUNT * right[4:8]
    win = current_right + DISCOUNT * right[:4]

    bloc_margins = (concede - entrench)[1:3]
    assert bloc_margins[0] > 0.5 > bloc_margins[1]
    q = 0.5  # Only the challenger bloc endorses, meeting one of two equiprobable quorums.
    gain = q * (entrench[0] - concede[0]) - 2.40
    success = q if gain > 0 else 0.0
    old_margin = (1 - success) * (current_right[-1] - current_left[-1])
    young_margin = win[-1] - ((1 - success) * concede[-1] + success * entrench[-1])
    votes_left = np.array([old_margin, young_margin])[:, None] + FIRST_VALENCE <= 0
    assert np.array_equal(votes_left[0], votes_left[1])
    first_win = votes_left[0] @ VALENCE_PROBABILITIES
    return gain, first_win, left[-1], old_margin, young_margin, *bloc_margins


if __name__ == "__main__":
    print("S  ballot  refusal_gain  P(L1)  P(L2|L1_seated)  old  young  bloc_L  bloc_R")
    for monitoring in (0.30, 0.38):
        for use_record in (False, True):
            values = evaluate(monitoring, use_record)
            assert abs(values[1] - 0.10) < 1e-12
            assert (values[0] > 0) == (use_record and monitoring == 0.38)
            ballot = "responsive" if use_record else "fixed"
            print(f"{monitoring:.2f}  {ballot:10s}", *(f"{x:.8f}" for x in values))
