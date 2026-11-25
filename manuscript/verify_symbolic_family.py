"""Exact checks for the nonzero-intercept illustration of Corollary 1."""
from fractions import Fraction as F
from itertools import product

ZERO, ONE = F(0), F(1)
LEFT, RIGHT, LEFT_BLOC, RIGHT_BLOC, VOTER = range(5)
positions = (F("-.5"), F("1.5"), F("-.5"), F("1.5"), F(".505"))
weights = (F(-2), F(2), ZERO, ZERO, ONE)
left_cap, right_cap = F(".15"), F(".30")
alpha_mass = {F("-.02"): F(".55"), F(".03"): F(".45")}
beta_mass = {F(".10"): F(".60"), F("1.10"): F(".40")}
a = sum(x * p for x, p in alpha_mass.items())
b = sum(x * p for x, p in beta_mass.items())
noise_radius, final_monitoring = F(".016"), F(".01")
discount, refusal_cost, endorsement_cost = F(".25"), F("2.35"), F(".5")
q = F(".50")
valence = {F(-100): F(".45"), ZERO: F(".10"), F(100): F(".45")}


def extrema(fn, cap):
    """Return exact min and max of a quadratic on [0, cap]."""
    lo, middle, hi = fn(ZERO), fn(cap / 2), fn(cap)
    quadratic = 2 * (lo + hi - 2 * middle)
    linear = hi - lo - quadratic
    values = [lo, hi]
    if quadratic:
        vertex = -linear / (2 * quadratic)
        if 0 < vertex < 1:
            values.append(fn(vertex * cap))
    return min(values), max(values)


def difference_bounds(first, first_cap, second, second_cap):
    lo1, hi1 = extrema(first, first_cap)
    lo2, hi2 = extrema(second, second_cap)
    return lo1 - hi2, hi1 - lo2


def flow(i, j, weight, mean_alpha, mean_beta, monitoring):
    level = positions[j] + weight * mean_alpha
    return (-(level - positions[i]) ** 2 + 2 * weights[i] * level * mean_alpha
            + monitoring * mean_beta ** 2 * (2 * weights[i] * weight - weight ** 2))


def final_values(record):
    ma, mb = record

    def range_for(i):
        return difference_bounds(lambda r: flow(i, RIGHT, r, ma, mb, final_monitoring),
                                 right_cap,
                                 lambda ell: flow(i, LEFT, -ell, ma, mb, final_monitoring),
                                 left_cap)

    assert range_for(LEFT_BLOC)[1] < -endorsement_cost
    assert range_for(RIGHT_BLOC)[0] > endorsement_cost
    assert q * max(range_for(RIGHT)[1], -range_for(LEFT)[0]) < refusal_cost
    low_vote, high_vote = range_for(VOTER)
    # A definite ballot at every valence for all continuous final offers.
    assert all(low_vote + v > 0 or high_vote + v < 0 for v in valence)
    win_right = sum(p for v, p in valence.items() if low_vote + v > 0)
    assert 0 < win_right < 1 and ma ** 2 + final_monitoring * mb ** 2 > 0
    democratic, autocratic = [], []
    for i in range(5):
        u_left = flow(i, LEFT, -left_cap, ma, mb, final_monitoring)
        u_right = flow(i, RIGHT, right_cap, ma, mb, final_monitoring)
        expected_valence = sum(v * p for v, p in valence.items() if low_vote + v > 0)
        democratic.append(win_right * u_right + (1 - win_right) * u_left
                           + (expected_valence if i == VOTER else 0))
        autocratic.append(flow(i, RIGHT, weights[RIGHT], ma, mb, final_monitoring))
    return democratic, autocratic, win_right


def mean_terminal(values, field, i, resolved):
    alpha_records = alpha_mass if resolved else {a: ONE}
    return sum(pa * pb * values[ma, mb][field][i]
               for ma, pa in alpha_records.items() for mb, pb in beta_mass.items())


def verify():
    assert a == F(".0025") and b == F(".50")
    spread = max(alpha_mass) - min(alpha_mass)
    beta_spread = max(beta_mass) - min(beta_mass)
    xmin_l = -positions[LEFT] - left_cap * (abs(a) + abs(b))
    xmax_l = -positions[LEFT] + left_cap * (abs(a) + abs(b))
    xmin_r = positions[RIGHT] - right_cap * (abs(a) + abs(b))
    assert xmin_l > 0 and xmin_r > 0
    assert spread * xmax_l < 2 * noise_radius < spread * xmin_r
    assert 2 * noise_radius < min(xmin_l, xmin_r) * (beta_spread - spread)
    z0 = -spread * positions[LEFT] / (2 * noise_radius)
    z1 = spread * a / (2 * noise_radius)
    values = {record: final_values(record) for record in product((*alpha_mass, a), beta_mass)}
    assert {value[2] for value in values.values()} == {F(".45"), F(".55")}
    resolved = [mean_terminal(values, 0, i, True) for i in range(5)]
    partial = [mean_terminal(values, 0, i, False) for i in range(5)]
    automatic = [mean_terminal(values, 1, i, True) for i in range(5)]
    J = [full - incomplete for full, incomplete in zip(resolved, partial)]

    def W(i, r, s):
        return flow(i, RIGHT, r, a, b, s) + discount * resolved[i]

    def A(i, r, s):
        return flow(i, RIGHT, r, a, b, s) + discount * automatic[i]

    def C(i, ell, s):
        return flow(i, LEFT, -ell, a, b, s) + discount * (partial[i] + (z0 + z1 * ell) * J[i])

    # These are quadratic in each side's own offer and affine in monitoring.
    for s in (ZERO, ONE):
        for i in (LEFT_BLOC, RIGHT_BLOC):
            lo, hi = difference_bounds(lambda ell: C(i, ell, s), left_cap,
                                       lambda r: A(i, r, s), right_cap)
            assert lo > endorsement_cost if i == LEFT_BLOC else hi < endorsement_cost
        old_lo, old_hi = difference_bounds(
            lambda r: flow(VOTER, RIGHT, r, a, b, s), right_cap,
            lambda ell: flow(VOTER, LEFT, -ell, a, b, s), left_cap)
        assert 0 < old_lo <= old_hi < 100
        for success_after_loss in (ZERO, q):
            # The old voter's material margin scales by (1-success_after_loss).
            lo, hi = difference_bounds(
                lambda r: W(VOTER, r, s) - success_after_loss * A(VOTER, r, s), right_cap,
                lambda ell: (1 - success_after_loss) * C(VOTER, ell, s), left_cap)
            assert 0 < lo <= hi < 100
        flat, full = C(LEFT, ZERO, s), C(LEFT, left_cap, s)
        assert q * (flat - A(LEFT, right_cap, s)) > (1 - q) * (full - flat)

    gap = positions[RIGHT] - positions[LEFT]
    assert 2 * gap * a + 2 * a ** 2 * weights[RIGHT] > discount * z1 * J[RIGHT]
    assert 2 * a ** 2 * (-weights[LEFT] - left_cap) + discount * z1 * J[LEFT] > 0

    def gross(ell, s):
        return A(RIGHT, right_cap, s) - C(RIGHT, ell, s)

    def cutoff(ell):
        m = (weights[RIGHT] + ell) ** 2 - (weights[RIGHT] - right_cap) ** 2
        assert gross(ell, ONE) - gross(ell, ZERO) == b ** 2 * m
        return (refusal_cost / q - gross(ell, ZERO)) / (b ** 2 * m)

    assert gross(left_cap, ZERO) < refusal_cost / q < gross(ZERO, ONE)
    s_low, s_high = cutoff(left_cap), cutoff(ZERO)
    assert 0 < s_low < s_high < 1
    for ell in (ZERO, left_cap / 3, left_cap):
        s = cutoff(ell)
        assert q * gross(ell, s) == refusal_cost
        v = a ** 2 + s * b ** 2
        linear = 2 * gap * a + 2 * v * weights[RIGHT] - discount * z1 * J[RIGHT]
        assert 2 * v * ell + linear > 0
        assert (2 * v * ell + linear) ** 2 == linear ** 2 + 4 * v * (refusal_cost / q - gross(ZERO, s))
    print("Conditional probability of resolving the intercept:", z0, "+", z1, "* ell")
    print("Cutoffs:", s_low, s_high, "or", float(s_low), float(s_high))
    print("Strict endpoint, voting, enforcement and continuation checks passed.")


if __name__ == "__main__":
    verify()
