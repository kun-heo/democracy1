"""Symbolic equilibrium branches for the original finite-menu OLG game.

Run: python full_game.py
Exact rational interpolation is used only for functions with a proved degree
bound. Bernstein coefficients certify the signs throughout each interval.
"""
import json
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import sympy as sp

from solve_equilibrium import (AUT, ENVS, OLD, PARTIES, RANKING, RECORDS, SIGNALS,
                               WORLDS, YOUNG, Election, Game, available, exact_values)


STATES = tuple(product(PARTIES, ENVS, RECORDS))
S_BAND = (F(".880"), F(".882"))
K_BAND = (F(".565"), F(".572"))
BANDS = {"high": S_BAND, "attempt": K_BAND}


def load_paths():
    specs = json.loads(Path(__file__).with_name("stationary_paths.json").read_text())
    paths = []
    for spec in specs:
        p = {}
        for inc, env, rec, l, r, cutoff, refuse, groups in spec["elections"]:
            p["programs", inc, env, rec] = (l, r)
            p["election", inc, env, rec, l, r] = Election(
                bool(refuse), tuple(map(tuple, groups)), tuple(j >= cutoff for j in range(11)), ())
        for regime, party, env, rec, package, *rule in spec["policies"]:
            p["policy", regime, party, env, rec, package] = tuple(rule)
        paths.append(p)
    return paths


def pure_eliminations(options, scores):
    left, right, witnesses = list(options), list(options), []
    while True:
        rows, cols = list(left), list(right)
        for l in left:
            better = next((y for y in left if all(scores[y, r][0] > scores[l, r][0] for r in right)), None)
            if better is not None:
                rows.remove(l)
                witnesses.extend((0, (better, r), (l, r)) for r in right)
        for r in right:
            better = next((y for y in right if all(scores[l, y][1] > scores[l, r][1] for l in left)), None)
            if better is not None:
                cols.remove(r)
                witnesses.extend((1, (l, better), (l, r)) for l in left)
        if (rows, cols) == (left, right):
            return witnesses, left, right
        left, right = rows, cols


def campaign_checks(date, state, scores, paths, templates, margins, targets):
    options = available(state[2])
    if date == 4:  # stationary branch
        for p in paths:
            l, r = p["programs", *state]
            for y in options:
                if y != l:
                    margins.append(("weak", scores[l, r][0]-scores[y, r][0]))
                if y != r:
                    margins.append(("weak", scores[l, r][1]-scores[l, y][1]))
        return paths[0]["programs", *state]

    key = (date, state)
    if key not in templates:
        templates[key] = pure_eliminations(options, scores)
    witnesses, left, right = templates[key]
    margins.extend(("strict", scores[a][j]-scores[b][j]) for j, a, b in witnesses)
    if len(right) == 1:
        r = right[0]
        margins.extend(("equal", scores[l, r][0]-scores[left[0], r][0]) for l in left[1:])
        return (left[0], r)
    assert date == 3 and state == (1, 0, 0) and left == [0, 1, 2] and right == [0, 1, 3]
    for r in right:
        margins.append(("strict", 3*scores[0, r][0]+scores[2, r][0]-4*scores[1, r][0]))
    for l in (0, 2):
        margins.append(("strict", scores[l, 3][1]-scores[l, 0][1]))
    a = scores[0, 1][1]-scores[0, 3][1]
    b = scores[2, 3][1]-scores[2, 1][1]
    c = scores[2, 1][0]-scores[0, 1][0]
    d = scores[0, 3][0]-scores[2, 3][0]
    margins.extend(("strict", value) for value in (a, b, c, d))
    targets.update(zip(("a", "b", "c", "d"), (a, b, c, d)))
    return None  # this campaign has the interior two-by-two support


def stage(game, future, date, paths, templates):
    """Rational values of every inequality defining the displayed political profile."""
    margins, targets, installed, signature = [], {}, {}, []
    updated = game.zeros()
    for regime, party, env, rec in product((0, 1), PARTIES, ENVS, RECORDS):
        state = (regime, party, env, rec)
        payoffs, rules, _ = game.actions(future, *state)
        if regime:
            updated[state] = payoffs[AUT]
        nxt = sum(p*future[regime, party, e] for e, p in enumerate(game.ep[env]))
        nxt[..., YOUNG] = nxt[..., OLD].copy()
        nxt[..., OLD] = F(0)
        same = nxt[rec]
        learned = np.array([nxt[a+1, a] for a in WORLDS])
        increments = game.delta*game.publication*(learned-same) if rec == 0 else 0
        continuation = game.delta*same + game.nonzero[:, None, None]*increments
        for pkg, value in payoffs.items():
            installed[*state, pkg] = value
            signature.append(rules[pkg])
            if pkg == AUT:
                baseline, weight = game.beta[party], game.gamma[party]
                account = rec-1 if rec else F(1, 2)
                credit = game.expectation(continuation.transpose(1, 0, 2), rec)[:, party]
            else:
                baseline, weight = game.mandates[party][pkg//2]
                account = pkg%2
                credit = continuation[:, account, party]
            for m, choice in zip(SIGNALS, rules[pkg]):
                score = -(game.xs-baseline)**2 + 2*weight*account*m*game.xs + credit
                for y in range(len(score)):
                    if y != choice:
                        margins.append(("strict" if y < choice else "weak", score[choice]-score[y]))
            if date == 4:
                for p in paths:
                    key = ("policy", *state, pkg)
                    if key in p:
                        assert rules[pkg] == p[key]

    for inc, env, rec in STATES:
        other, state = 1-inc, (inc, env, rec)
        scores, values = {}, {}
        for pair in product(available(rec), repeat=2):
            victory = installed[0, inc, env, rec, pair[inc]]
            concede = installed[0, other, env, rec, pair[other]]
            entrench = installed[1, inc, env, rec, pair[inc]]
            val, election, stats = game.contest(inc, env, rec, pair, victory, concede, entrench)
            signature.append((election, stats.willing))
            account = ((rec-1,)*len(RANKING) if rec else
                       (pair[0]%2,)*len(RANKING) if pair[0]%2 == pair[1]%2 else RANKING)
            for j, willing in enumerate(stats.willing):
                a = account[j]
                gap = concede[a, j+2]-entrench[a, j+2]-game.defense
                margins.append(("weak", gap) if willing else ("strict", -gap))
            margins.append(("strict", stats.gain) if election.refuse else ("weak", -stats.gain))
            # A common valence enters each ballot with slope one. Check both
            # sides of its single cutoff, including the left-at-equality rule.
            for j, gap in enumerate(stats.gaps):
                first = next((h for h, ballots in enumerate(election.ballots) if ballots[j]), len(game.eta))
                if first:
                    margins.append(("weak", -(gap+game.eta[first-1])))
                if first < len(game.eta):
                    margins.append(("strict", gap+game.eta[first]))
            values[pair] = val
            scores[pair] = game.expectation(val, rec)[:2]
            if rec == 0 and env == 1 and pair in ((0, 3), (2, 3)):
                targets[f"T{inc}:{pair[0]}{pair[1]}"] = stats.gain
            if date == 4:
                for p in paths:
                    key = ("election", *state, *pair)
                    if key in p:
                        assert election[:3] == p[key][:3]
        chosen = campaign_checks(date, state, scores, paths, templates, margins, targets)
        if chosen is None:
            left_zero = targets["b"] / (targets["a"] + targets["b"])
            right_one = targets["d"] / (targets["c"] + targets["d"])
            left_mix = ((0, left_zero), (2, 1-left_zero))
            right_mix = ((1, right_one), (3, 1-right_one))
            updated[0, *state] = sum(
                pl*pr*values[l, r] for l, pl in left_mix for r, pr in right_mix)
        else:
            updated[0, *state] = values[chosen]
    return updated, margins, targets, signature


def kernels(paths):
    matrices = []
    for s in (0, 1):
        game = Game(exact=True, params={"high": str(s)})
        values, chains = exact_values(game, paths[0])
        alt, other = exact_values(game, paths[1])
        assert (values == alt).all() and chains == other
        matrices.append([m for _, m in chains])
        # Same transition and reward functions for the two stationary paths.
        # Fixed-plan rewards are affine in k and affine in s on each k edge.
        for k in K_BAND:
            edge = Game(exact=True, params={"high": str(s), "attempt": str(k)})
            values, chains = exact_values(edge, paths[0])
            alt, other = exact_values(edge, paths[1])
            assert (values == alt).all() and chains == other
    x = sp.symbols("s")
    denominator = (451894-13399*x)**2
    for m0, m1 in zip(*matrices):
        assert sum(m0.row(i) != m1.row(i) for i in range(16)) == 2
        kernel = m0 + x*(m1-m0)
        det = sp.factor((sp.eye(16)-sp.Rational(1, 4)*kernel).det(method="domain-ge"))
        ratio = sp.cancel(det/denominator)
        assert ratio.is_Rational and ratio > 0
    return denominator


@lru_cache(None)
def bernstein_inverse(n):
    entries = [[sp.binomial(n, j)*sp.Rational(i, n)**j*(1-sp.Rational(i, n))**(n-j)
                for j in range(n+1)] for i in range(n+1)]
    return tuple(tuple(F(x) for x in row) for row in sp.Matrix(entries).inv().tolist())


def bernstein_values(samples):
    inverse = bernstein_inverse(len(samples)-1)
    return tuple(sum(w*y for w, y in zip(row, samples)) for row in inverse)


def polynomial_from_samples(nodes, values, symbol):
    x = sp.symbols(symbol)
    return sp.factor(sp.interpolate([(sp.Rational(t), sp.Rational(y)) for t, y in zip(nodes, values)], x))


def certify_band(parameter, paths, templates, anchors, symbolic_det, fixed=None):
    lo, hi = BANDS[parameter]
    cache = {}
    def evaluate(x):
        if x not in cache:
            game = Game(exact=True, params={**(fixed or {}), parameter: str(x)})
            future = game.zeros()
            result = {}
            for n in (1, 2, 3, 4):
                if n == 4:
                    future, _ = exact_values(game, paths[0])
                updated, margins, targets, signature = stage(game, future, n, paths, templates)
                assert signature == anchors[n], (parameter, x, n, "different action profile")
                result[n] = margins, targets
                future = updated
            cache[x] = result
        return cache[x]

    all_formulas = {}
    for n in (1, 2, 3, 4):
        degree = n if parameter == "high" else 1
        nodes = [lo+(hi-lo)*F(i, degree) for i in range(degree+1)]
        scale = [(451894-13399*x)**2 if n == 4 and parameter == "high" else F(1) for x in nodes]
        runs = [evaluate(x)[n] for x in nodes]
        # Distinct incentive margins can be identical polynomials.
        polynomials = set()
        for column in zip(*(m for m, _ in runs), strict=True):
            discipline = column[0][0]
            assert all(mode == discipline for mode, _ in column)
            polynomials.add((discipline, tuple(x*w for (_, x), w in zip(column, scale))))
        for mode, samples in polynomials:
            coefficients = bernstein_values(samples)
            if mode == "equal":
                assert not any(coefficients), (parameter, n, mode, coefficients)
            elif mode == "strict":
                assert min(coefficients) > 0, (parameter, n, mode, coefficients)
            else:
                assert min(coefficients) >= 0, (parameter, n, mode, coefficients)
        print(f"{parameter}: horizon {n if n<4 else 'infinite'}, "
              f"{len(runs[0][0])} inequalities/identities, {len(polynomials)} distinct margin tests")
        results = {}
        for k in runs[0][1]:
            vals = [r[1][k]*w for r, w in zip(runs, scale)]
            num = polynomial_from_samples(nodes, vals, "s" if parameter == "high" else "k")
            den = symbolic_det if parameter == "high" and n == 4 else 1
            results[k] = sp.factor(num/den)
        all_formulas[n] = results.copy()
        if fixed is not None:
            continue
        if n == 3:
            results['left_mix'] = sp.factor(results['b']/(results['a']+results['b']))
            results['right_mix'] = sp.factor(results['d']/(results['c']+results['d']))
        for name, formula in results.items():
            print(f"  {name} = {formula}")
    return all_formulas


def main():
    paths = load_paths()
    denominator = kernels(paths)
    templates, anchors = {}, {}
    base = Game(exact=True)
    future = base.zeros()
    for n in (1, 2, 3, 4):
        if n == 4:
            future, _ = exact_values(base, paths[0])
        updated, margins, _, anchors[n] = stage(base, future, n, paths, templates)
        assert all(v>0 if mode=="strict" else v==0 if mode=="equal" else v>=0
                   for mode, v in margins)
        if n == 4:
            for rec, a in ((0, 0), (0, 1), (1, 0), (2, 1)):
                assert (updated[:, :, :, rec, a] == future[:, :, :, rec, a]).all()
        future = updated
    for parameter in BANDS:
        certify_band(parameter, paths, templates, anchors, denominator)
    edges = []
    for cost in K_BAND:
        print(f"Rectangle edge k={cost}")
        edges.append(certify_band("high", paths, templates, anchors, denominator, {"attempt": str(cost)}))
    print("Each fixed-profile margin numerator is affine in k. Both edges imply the whole rectangle.")
    s, k, h, d = sp.symbols("s k h d")
    central_s, central_k = sp.Rational(22, 25), sp.Rational(57, 100)
    a, b = map(sp.Rational, K_BAND)
    full = {}
    for n in (1, 2, 3, 4):
        full[n] = {}
        for name, low in edges[0][n].items():
            high = edges[1][n][name]
            full[n][name] = sp.factor(((b-k)*low+(k-a)*high)/(b-a))
        print(f"Rectangle formulas for horizon {n if n<4 else 'infinite'}:")
        for name, formula in full[n].items():
            print(f"  {name} = {formula}")
    published = {
        (2, "T1:23"): ("-.035258816", ".5832768", "0"),
        (3, "T1:23"): (".03716781804544", ".614626818384", "-.00264"),
        (3, "T1:03"): (".53258958478336", ".629251907072", "-.00264"),
    }
    for (n, name), terms in published.items():
        constant, linear, quadratic = map(sp.Rational, terms)
        compact = constant + linear*h + quadratic*h*h - d
        assert sp.expand(full[n][name].subs({s: h+central_s, k: d+central_k})-compact) == 0
    for n, sign, pair in [(2, -1, "T1:23"), (3, 1, "T1:23"), (3, 1, "T1:03"), (4, 1, "T1:03")]:
        assert sign*full[n][pair].subs({s: central_s, k: central_k}) > 0


if __name__ == "__main__":
    main()
