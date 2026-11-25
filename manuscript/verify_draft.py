"""Additional exact checks for the one-election benchmark and the two-election electoral comparison."""
from fractions import Fraction as F
from itertools import product

from continuous_mandates import (
    COST, FINAL_S, LEFT, PRIOR, RIGHT, breakpoints, certificates,
    contest, flow, minmax_quadratic, trial,
)
from verify_bayesian_mandates import DISCOUNT, VALENCE_MASS


def current_only_gain(r, left, s):
    return 2 - COST + s * (4 * r - r * r + 4 * left + left * left) / 8


def check_terminal_benchmark():
    for s, r, left in product(map(F, (0, 1)), (F(0), F(1, 4), F(1)), (F(0), F(2, 3), F(1))):
        own = flow(RIGHT, r, PRIOR, s)
        challenger = flow(LEFT, -left, PRIOR, s)
        assert (own[RIGHT] - challenger[RIGHT]) / 2 - COST == current_only_gain(r, left, s)
    assert current_only_gain(F(1), F(1), F(".40")) == 0
    assert current_only_gain(F(1), F(0), F(1)) == -F(".025")
    for s in map(F, (0, 1)):
        rival = flow(RIGHT, F(1), PRIOR, s)[LEFT]
        flat = flow(LEFT, F(0), PRIOR, s)[LEFT]
        full = flow(LEFT, -F(1), PRIOR, s)[LEFT]
        gap = F(".45") * (flat - (rival + full) / 2)
        assert gap == F(".225") * (4 + s / 2) > 0


def check_electoral_comparison():
    s = F(".41")
    bounds = breakpoints(LEFT)
    intervals = list(zip(bounds, bounds[1:]))
    gains = []
    expected_odds = [F(".53"), F(".55")]
    for fixed in (False, True):
        def refusal_gain(offer):
            return contest(offer, F(1), s, fixed)[0]

        gains.append(refusal_gain(F(0)))
        for lo, hi in intervals:
            assert minmax_quadratic(refusal_gain, lo, hi)[2] >= s / 2
            for refused, age in product((False, True), (0, 1)):
                def ballot(offer):
                    return contest(offer, F(0), s, fixed)[1][refused][0][age]
                assert minmax_quadratic(ballot, lo, hi)[0] >= F(".01")
            for refused in (False, True):
                def left_value(offer):
                    return contest(offer, F(1), s, fixed)[1][refused][1][LEFT]
                assert minmax_quadratic(left_value, lo, hi)[2] > 0
        assert contest(F(0), F(1), s, fixed)[1][False][1][LEFT] > contest(F(1), F(1), s, fixed)[1][True][1][LEFT]
        probability = F(0)
        for h, mass in trial(LEFT, F(0), s).items():
            policies = [flow(LEFT, -F(1), h, FINAL_S), flow(RIGHT, F(1), h, FINAL_S)]
            ballots = (False, True, True) if fixed else certificates(*policies)
            elected = sum(p for p, right in zip(VALENCE_MASS, ballots) if right)
            probability += mass * elected
        assert probability == expected_odds[fixed]
    income = flow(RIGHT, 1, (0,), FINAL_S)[RIGHT] - flow(LEFT, -1, (0,), FINAL_S)[RIGHT]
    assert income == F("3.7632")
    assert gains == [F(".002757"), -F(".006651")]
    assert gains[0] - gains[1] == F(1, 2) * DISCOUNT * F(".20") * F(".10") * income
    # Both cutoffs for a flat opponent compare the same posterior laws.
    for s0 in (F(0), F(1)):
        assert contest(F(0), F(1), s0)[0] - current_only_gain(F(1), F(0), s0) == F(".249007")
    print("At S=.41, flat-opponent gains and right return probabilities", gains, expected_odds)


if __name__ == "__main__":
    check_terminal_benchmark()
    check_electoral_comparison()
    print("One-election and two-election counterfactual checks passed.")

