"""Exact best-response checks for the two-election game with continuous mandates.

Run with Python 3.10+ in this folder. No external packages are needed.
"""
from fractions import Fraction as F
from functools import cache
from itertools import combinations, product

from bayesian_horizons import (
    DISCOUNT, PRIOR, SIGN, WORLDS, certificates, election_payoffs, flow, means,
    play_after_certificate,
)
from verify_bayesian_mandates import BASE, ERROR_RADIUS

HALF = F(1, 2)
COST = F("2.40")
FINAL_S = F(".05")
ALL_RECORDS = tuple(h for n in range(1, 5) for h in combinations(PRIOR, n))
LEFT, RIGHT = 0, 1
# For a right incumbent, only the challenger's bloc enforces, at a one-bloc quorum.
ENFORCERS = ((LEFT,), ())


def minmax_quadratic(fn, lo, hi):
    """Exact extrema, and least derivative, on an interval known to be quadratic."""
    u, mid, w = (fn(x) for x in (lo, (lo + hi) / 2, hi))
    a = 2 * (w + u - 2 * mid)
    b = 4 * mid - w - 3 * u
    values = [u, w]
    if a:
        x = -b / (2 * a)
        if 0 < x < 1:
            values.append(u + b * x + a * x * x)
    return min(values), max(values), min(b, b + 2 * a) / (hi - lo)


def final_election(h, fixed_neutral=False):
    """Check the neutral certificate and responsive mandates at every final record."""
    a, b = means(h)
    dispersion = a * a + FINAL_S * b * b
    if a >= 0:
        # The neutral voter likes L best at zero. The two endpoints of R suffice
        # because the voter's utility from R is concave in its mandate weight.
        assert F(".02") + 4 * a > 0
        assert F(".02") + F("2.01") * a + dispersion > 0
    else:
        # R's mandate raises its own payoff and weakly improves its electoral
        # prospects. At r=1 the neutral vote keeps its sign over every L offer.
        lo, hi, _ = minmax_quadratic(
            lambda x: F(".02") + F("2.01") * a * x + dispersion * x * x, F(1), F(2)
        )
        assert lo > 0 or hi < 0
    policies = [flow(j, SIGN[j], h, FINAL_S) for j in (LEFT, RIGHT)]
    ballot = (False, True, True) if fixed_neutral else certificates(*policies)
    ordinary, _ = election_payoffs(*policies, ballot)
    autocracy = [flow(j, 2 * SIGN[j], h, FINAL_S) for j in (LEFT, RIGHT)]
    return (ordinary, *autocracy)


FINAL = {(h, fixed): final_election(h, fixed) for h in ALL_RECORDS for fixed in (False, True)}


@cache
def posterior_distribution(x, state):
    centers = {i: x * (WORLDS[i][0] + WORLDS[i][1] * state) for i in PRIOR}
    ends = sorted({v + sign * ERROR_RADIUS for v in centers.values() for sign in (-1, 1)})
    result = {}
    for lo, hi in zip(ends, ends[1:]):
        mid = (lo + hi) / 2
        record = tuple(i for i in PRIOR if abs(mid - centers[i]) < ERROR_RADIUS)
        if record:
            mass = (hi - lo) * len(record) / (2 * ERROR_RADIUS * len(PRIOR))
            result[record] = result.get(record, 0) + mass
    assert sum(result.values()) == 1
    return result


@cache
def trial(j, magnitude, s):
    result = {}
    for state in (-1, 1):
        for forecast, mass in ((0, (1 - s) / 2), (state, s / 2)):
            if not mass:
                continue
            x = BASE[j] + SIGN[j] * magnitude * forecast / 2
            for record, prob in posterior_distribution(x, state).items():
                result[record] = result.get(record, 0) + mass * prob
    return result


def breakpoints(j):
    cuts = {F(0), F(1)}
    for state in (-1, 1):
        for i, k in combinations(PRIOR, 2):
            difference = WORLDS[i][0] - WORLDS[k][0] + (WORLDS[i][1] - WORLDS[k][1]) * state
            if not difference:
                continue
            for shift in (-2 * ERROR_RADIUS, 0, 2 * ERROR_RADIUS):
                point = (shift / difference - BASE[j]) / (SIGN[j] * HALF * state)
                if 0 < point < 1:
                    cuts.add(point)
    return sorted(cuts)


@cache
def installed(j, magnitude, regime, s, fixed_neutral=False):
    current = flow(j, SIGN[j] * magnitude, PRIOR, s)
    distribution = trial(j, magnitude, s)
    index = 0 if regime == "D" else 1 + j
    result = list(current)
    for i in range(5):
        nxt = 5 if i == 4 else i
        result[i] += DISCOUNT * sum(prob * FINAL[h, fixed_neutral][index][nxt] for h, prob in distribution.items())
    return tuple(result)


@cache
def contest(left, right, s, fixed_neutral=False):
    own = installed(RIGHT, right, "D", s, fixed_neutral)
    lawful = installed(LEFT, left, "D", s, fixed_neutral)
    entrenched = installed(RIGHT, right, "A", s, fixed_neutral)
    branches = own, lawful, entrenched
    # Enforcement over the whole joint menu also follows from the flow bound in the note.
    assert lawful[2] - entrenched[2] > HALF and lawful[3] - entrenched[3] < HALF
    gain = (entrenched[RIGHT] - lawful[RIGHT]) / 2 - COST
    outcomes = {}
    for refuse in (False, True):
        lcert, rcert = play_after_certificate(RIGHT, branches, ENFORCERS, refuse, COST)
        ballot_margins = tuple(rcert[i] - lcert[i] for i in (5, 4))
        # Conditional certificate probability .55 is checked globally below.
        party_payoffs, _ = election_payoffs(lcert, rcert, (False, True, True))
        outcomes[refuse] = ballot_margins, party_payoffs
    return gain, outcomes


def main():
    cuts = breakpoints(LEFT)
    assert cuts == [F(0), F(1, 4), F(91, 96), F(19, 20), F(99, 104), F(1)]
    assert breakpoints(RIGHT) == [F(0), F(1)]
    intervals = list(zip(cuts, cuts[1:]))
    # Extreme-valence certificates, enforcement, and dominance at the final election
    # are also shown by payoff bounds in the note.
    for s in (F(0), F(1)):
        for lo, hi in intervals:
            for refusal, age in product((False, True), (0, 1)):
                # The ballot margin is weakly increasing in the right mandate.
                lower, _, _ = minmax_quadratic(
                    lambda l: contest(l, F(0), s)[1][refusal][0][age], lo, hi
                )
                assert lower >= F(".01")
    intercept = contest(F(0), F(1), F(0))[0]
    assert intercept == -F(".150993")
    for lo, hi in intervals:
        next_gain = lambda l: contest(l, F(1), F(1))[0] - intercept
        u, mid, w = (next_gain(x) for x in (lo, (lo + hi) / 2, hi))
        quad = 2 * (u + w - 2 * mid) / (hi - lo) ** 2
        lin = (4 * mid - w - 3 * u) / (hi - lo) - 2 * quad * lo
        constant = u - lin * lo - quad * lo ** 2
        assert quad == F(1, 8) and minmax_quadratic(next_gain, lo, hi)[2] >= HALF
        print("G interval and coefficients", lo, hi, quad, lin, constant)
        for refusal, lower_derivative in ((False, F(".14")), (True, F(".07"))):
            conditional_value = lambda l: contest(l, F(1), F(1))[1][refusal][1][LEFT]
            assert minmax_quadratic(conditional_value, lo, hi)[2] > lower_derivative
    worst_concession = min(contest(F(0), F(1), s)[1][False][1][LEFT] for s in map(F, (0, 1)))
    best_refusal = max(contest(F(1), F(1), s)[1][True][1][LEFT] for s in map(F, (0, 1)))
    assert worst_concession == -F("3.4249742") > best_refusal == -F("3.75810885")
    right_slope = lambda l: contest(l, F(1), F(1))[0] - intercept
    cutoffs = tuple(-intercept / right_slope(l) for l in (F(1), F(0)))
    closed_form_start = -intercept / right_slope(F(1, 4))
    assert cutoffs == (F(31788, 209683), F(".402648"))
    assert closed_form_start == F(301986, 1015625)
    demonstration = F(150993, 426250)
    assert contest(F(1, 10), F(1), demonstration)[0] == 0
    print("Initial monitoring cutoffs", *cutoffs)
    print("Simple square-root formula from", closed_form_start)
    print("At S =", demonstration, "the equilibrium weights are -1/10 and +1.")


if __name__ == "__main__":
    main()

