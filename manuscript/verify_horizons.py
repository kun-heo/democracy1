"""Exact certificates for Bayesian mandates with three elections and a stationary horizon.

Run python verify_horizons.py in the source directory. Python 3.10+; no packages.
"""
from fractions import Fraction as F

from bayesian_horizons import (
    AUT, DEM, LEFT, RIGHT, PLAYERS, OFFERS, PRIOR, RECORDS, STATES, VALUE_ZERO,
    FINAL_MONITORING, ZERO, ONE, BLOC_COST, Stage, Decision, pure_pair, kernel,
    kernel_and_reward, evaluate, linear_solve, prior_cells, initial, ordered,
    intervals, zero_of_affine,
)

S0 = FINAL_MONITORING
K0, K1 = F('2.40'), F('2.55')


def stationary_certificate(cost, s):
    """Specify only on-path offers and transfers. Derive and check every off-path choice."""
    # Record order is (-.02,0), (-.02,1), (.02,0), (.02,1), beta=0,
    # beta=1, prior. The first four are point posteriors.
    if (cost, s) == (K0, S0):
        refusing = set(RECORDS[1:])
    elif (cost, s) == (K1, S0):
        refusing = {(3,), (1, 3)}
    elif (cost, s) == (K1, F('.14')):
        refusing = {(1,), (3,), (1, 3), PRIOR}
    else:
        raise ValueError('There is no proposed stationary certificate for these parameters.')

    profile = {}
    for inc in PLAYERS:
        for h in RECORDS:
            pair = (0, 0) if h == (0, 2) else (1, 1)
            refusal = h in refusing
            middle_right = (inc == RIGHT) if refusal else (h != (0,))
            ballots = (False, middle_right, True)
            on = Decision(refusal, ballots, ((1-inc,), ()), F(1, 2),
                          F(9, 20) + F(middle_right, 10), ZERO)
            profile[inc, h] = tuple(F(p == pair) for p in OFFERS), {pair: on}

    values = evaluate(profile, s, cost)
    stage = Stage(values, attempt_cost=cost)
    for h in RECORDS:
        for inc in PLAYERS:
            mix, assumed = profile[inc, h]
            pair = pure_pair(mix)
            game = {p: stage.contest(inc, h, p, s) for p in OFFERS}
            payoff, actual = game[pair]
            assert payoff == values[DEM, inc, h]
            assert (actual.refuse, actual.ballots, actual.supporters) == (
                assumed[pair].refuse, assumed[pair].ballots, assumed[pair].supporters)
            for j in PLAYERS:
                alt = list(pair)
                alt[j] = 1 - alt[j]
                assert payoff[j] >= game[tuple(alt)][0][j]
            for p in OFFERS:
                _, lawful, entrench = stage.branches(inc, h, p, s)
                assert lawful[2+1-inc]-entrench[2+1-inc] > BLOC_COST
                assert lawful[2+inc]-entrench[2+inc] < BLOC_COST
            profile[inc, h] = mix, {p: game[p][1] for p in OFFERS}
            assert stage.installed(AUT, inc, h, 2*(2*inc-1), s) == values[AUT, inc, h]
    assert values == evaluate(profile, s, cost)
    return values, profile


def cutoff(stage, inc, rival):
    pair = ordered(inc, 1, rival)
    gains = [stage.contest(inc, PRIOR, pair, s)[1].gain for s in (ZERO, ONE)]
    return zero_of_affine(*gains)


def verify_initial(stage, tag):
    """Verify intervals and their boundaries, including all campaign and ballot indifferences."""
    result = []
    for inc in PLAYERS:
        lower, upper = (cutoff(stage, inc, rival) for rival in (1, 0))
        cuts = prior_cells(stage, inc)
        points = cuts | {(l+r)/2 for l, r in intervals(cuts)}
        for s in sorted(points):
            mix, decisions, unique = initial(stage, inc, s)
            assert unique  # the campaign outcome/payoffs, up to equivalent labels
            assert all(d.success == F(1, 2) and d.supporters == ((1-inc,), ())
                       for d in decisions.values())
            for pair, mass in zip(OFFERS, mix):
                if mass:
                    assert decisions[pair].refuse == (s > upper)
                    if s > ZERO:
                        assert pair == ordered(inc, 1, int(s <= lower or s > upper))
            if upper < ZERO:
                assert all(d.refuse for d in decisions.values())
            if s <= lower:
                assert all(not d.refuse for d in decisions.values())
            if lower < s <= upper:
                assert decisions[ordered(inc, 1, 1)].refuse
        print(f'{tag:31s} {"LR"[inc]}   {float(lower): .9f}   {float(upper): .9f}')
        result.append((lower, upper))
    return result


def finite_tail(cost):
    v1, p1, u1 = Stage(VALUE_ZERO, attempt_cost=cost).solve()
    v2, p2, u2 = Stage(v1, attempt_cost=cost).solve()
    assert all(u1.values()) and all(u2.values())
    for profile in (p1, p2):
        for (inc, _), (_, decisions) in profile.items():
            assert all(d.supporters == ((1-inc,), ()) for d in decisions.values())
    return v1, v2, p2


def return_odds(profile):
    """Next election, after R concedes to a flat L. Distinguish certification from installation."""
    cert, office = ZERO, ZERO
    for h, mass in kernel(PRIOR, LEFT, 0, S0).items():
        mix, decisions = profile[LEFT, h]
        for pair, m in zip(OFFERS, mix):
            d = decisions[pair]
            p = mass * m * d.right
            cert += p
            office += p * (1 - d.success * d.refuse)
    return cert, office


def electoral_feedback(cost, tail_beyond_next, actual_profile, current_s, tag):
    # Remove only the record-dependence of the zero-valence certificate at the next election.
    # The tail from the election after that is identical in both games.
    stages = []
    for fixed in (False, True):
        vals, prof, unique = Stage(tail_beyond_next, fixed, cost).solve()
        assert all(unique.values())
        stage = Stage(vals, attempt_cost=cost)
        mix, ds, _ = initial(stage, RIGHT, current_s)
        on = pure_pair(mix)
        gain = ds[0, 1].gain
        cert, office = return_odds(prof)
        assert ds[on].refuse == (not fixed) and (gain > 0) == (not fixed)
        if not fixed:
            assert return_odds(actual_profile) == (cert, office)
        assert on == ((0, 1) if fixed else (1, 1))
        print(f'{tag} {"fixed" if fixed else "responsive vote"}: '
              f'gain {float(gain):+.10f}, next certificate {float(cert):.4f}, '
              f'next office {float(office):.4f}, upper cutoff {float(cutoff(stage, RIGHT, 0)):.9f}')
        stages.append((cert, office, gain))
    return stages


def eventually_autocratic(profile, s, cost):
    # Closed democratic classes receive absorption probability zero.
    rows, _ = kernel_and_reward(profile, s, cost)
    reachable = {z for z in STATES if z[0] == AUT}
    while True:
        larger = reachable | {z for z in STATES if set(rows[z]) & reachable}
        if larger == reachable:
            break
        reachable = larger
    dem = [z for z in STATES if z[0] == DEM and z in reachable]
    A = [[F(z == t)-rows[z].get(t, ZERO) for t in dem] for z in dem]
    b = [[sum(p for t, p in rows[z].items() if t[0] == AUT)] for z in dem]
    ans = {z: F(z[0] == AUT) for z in STATES}
    for z, (w,) in zip(dem, linear_solve(A, b)):
        ans[z] = w
    assert all(ans[z] == sum(p*ans[t] for t, p in rows[z].items()) for z in STATES)
    return ans


def main():
    stat = {}
    for cost in (K0, K1):
        v1, v2, p2 = finite_tail(cost)
        for n, future in [(2, v1), (3, v2)]:
            verify_initial(Stage(future, attempt_cost=cost), f'{n} elections, k={float(cost)}')
        if cost == K0:
            stats = electoral_feedback(cost, v1, p2, F('.14'), 'three elections')
            assert [x[:2] for x in stats] == [(F('.51'), F('.465')), (F('.55'), F('.495'))]
        v, p = stationary_certificate(cost, S0)
        verify_initial(Stage(v, attempt_cost=cost), f'infinite horizon, k={float(cost)}')
        stat[cost] = v, p
    v, p = stat[K1]
    stats = electoral_feedback(K1, v, p, F('.41'), 'infinite horizon')
    assert [x[:2] for x in stats] == [(F('.50'), F('.4325')), (F('.55'), F('.4675'))]
    for s in (S0, F('.14')):
        _, profile = stationary_certificate(K1, s)
        risk = eventually_autocratic(profile, s, K1)
        for inc in PLAYERS:
            on = pure_pair(profile[inc, PRIOR][0])
            refuse = profile[inc, PRIOR][1][on].refuse
            assert refuse == (s == F('.14'))
            print(f'permanent s={float(s):.2f}, {"LR"[inc]} incumbent: refuses={refuse}, '
                  f'eventual autocracy {risk[DEM, inc, PRIOR]} ({float(risk[DEM, inc, PRIOR]):.9f})')
        assert all(risk[DEM, i, (0,)] == 0 and risk[DEM, i, (3,)] == 1 for i in PLAYERS)
    print('All equilibrium and probability checks passed with rational arithmetic.')


if __name__ == '__main__':
    main()
