"""Exhaust the stationary independent-strategy equilibria of the binary-mandate game.

Run python stationary_equilibria.py. Every bound and comparison uses rational
arithmetic. In the single payoff-equivalent record we keep one label per party.
"""

from collections import Counter
from fractions import Fraction as F
from itertools import product
from typing import NamedTuple

import bayesian_horizons as m
from verify_bayesian_mandates import SIGN
from verify_horizons import eventually_autocratic, verify_initial

P = (0, 1)
ACTORS = (0, 1, 5)  # left party, right party, old voter



class Aff(NamedTuple):
    """An affine function of one actor's democratic values under L and R incumbency at this record."""
    c: F
    l: F
    r: F

    def plus(self, obj):
        if isinstance(obj, tuple):
            return Aff(*(x + y for x, y in zip(self, obj)))
        return Aff(self.c + obj, self.l, self.r)

    def minus(self, other):
        return Aff(*(x - y for x, y in zip(self, other)))

    def times(self, v):
        return Aff(*(x * v for x in self))

    def range(self, bounds):
        lo, hi = (self.c, self.c)
        for a, (low, high) in zip(self[1:], bounds):
            x, y = (a * low, a * high)
            lo += min(x, y)
            hi += max(x, y)
        return (lo, hi)


def const(x):
    return Aff(x, F(0), F(0))


def mix(x, y, p):
    return x.plus(y.minus(x).times(p))


def block_data(h, s, k, exact):
    """Construct every certificate branch for given values at more informative records. An old voter's value is a flow, and a young voter's continuation uses that old value."""
    inst = {}
    for j, r in product(m.PLAYERS, range(2)):
        w = SIGN[j] * r
        flow = m.flow(j, w, h, s)
        probs = m.kernel(h, j, w, s)
        f = {}
        for reg in (m.DEM, m.AUT):
            f[reg] = {}
            for i in (0, 1, 4, 5):
                if i == 5:
                    val = const(flow[5])
                else:
                    actor = 5 if i == 4 else i
                    same = probs.get(h, F(0)) if reg == m.DEM else F(0)
                    extra = sum((p * exact[reg, j, nxt][actor] for nxt, p in probs.items() if reg == m.AUT or nxt != h))
                    c = flow[i] + m.DISCOUNT * extra
                    val = Aff(c, m.DISCOUNT * same if j == 0 else 0, m.DISCOUNT * same if j == 1 else 0)
                f[reg][i] = val
        inst[j, r] = f
    blocks = {}
    for inc in m.PLAYERS:
        chal = 1 - inc
        for pair in m.OFFERS:
            iw = inst[inc, pair[inc]]
            cw = inst[chal, pair[chal]]
            T = iw[m.AUT][inc].minus(cw[m.DEM][inc]).times(F(1, 2)).plus(-k)
            os = {}
            for refused, neutral in product((0, 1), repeat=2):
                loss = {}
                for i in (0, 1, 4, 5):
                    loss[i] = mix(cw[m.DEM][i], iw[m.AUT][i], F(refused, 2)).plus(-k * refused if i == inc else 0)
                left, right = (iw[m.DEM], loss) if inc == 0 else (loss, iw[m.DEM])
                old = right[5].minus(left[5]).c
                young = right[4].minus(left[4])
                if neutral and old <= 0:
                    continue
                if not neutral and old <= 0:
                    young = None
                p_right = F(9, 20) + F(neutral, 10)
                vals = {i: mix(left[i], right[i], p_right).plus(45 if i == 5 else 0) for i in (0, 1, 5)}
                os[refused, neutral] = (young, vals)
            blocks[inc, pair] = (T, os)
    return blocks


def choices(blocks, bounds, allowed, force):
    """Keep actions compatible with the value bounds, deterministic tie rules, and any imposed branch. Remove an offer only if its alternative pays strictly more against each remaining rival offer."""
    options = {}
    for node, (t, os) in blocks.items():
        inc, pair = node
        tlo, thi = t.range(bounds[inc])
        possible = {}
        for (ref, neu), (young, values) in os.items():
            if node in force and (ref, neu) not in force[node]:
                continue
            if ref and thi <= 0 or (not ref and tlo > 0):
                continue
            if young is not None:
                lo, hi = young.range(bounds[5])
                if neu and hi <= 0 or (not neu and lo > 0):
                    continue
            possible[ref, neu] = values
        if not possible:
            return None
        options[node] = possible
    possible_actions = {key: set(vals) for key, vals in allowed.items()}
    for inc in P:
        for i in P:
            for r in tuple(allowed[inc, i]):
                opp = 1 - i
                losses = []
                for other in allowed[inc, opp]:
                    pair = tuple((r if j == i else other for j in P))
                    alt = tuple((1 - r if j == i else other for j in P))
                    advantages = [a[i].minus(b[i]).range(bounds[i])[0] for a in options[inc, alt].values() for b in options[inc, pair].values()]
                    losses.append(min(advantages) > 0)
                if all(losses):
                    possible_actions[inc, i].discard(r)
                    if not possible_actions[inc, i]:
                        return None
    return (options, possible_actions)


def tighten(blocks, options, initial):
    """Project necessary linear restrictions from refusal and voting onto each actor's box. Using weak inequalities contains all values satisfying a strict restriction."""
    bounds = {i: list(box) for i, box in initial.items()}

    def impose(i, a):
        if a.range(bounds[i])[1] < 0:
            return False
        for coord, coef in enumerate(a[1:]):
            if not coef:
                continue
            box = bounds[i]
            other = 1 - coord
            good = max((a[1 + other] * x for x in box[other]))
            limit = -(a[0] + good) / coef
            low, high = box[coord]
            low, high = (max(limit, low), high) if coef > 0 else (low, min(limit, high))
            if low > high:
                return False
            box[coord] = (low, high)
        return True
    for key, opts in options.items():
        inc, pair = key
        t, os = blocks[key]
        if len({ref for ref, neu in opts}) == 1:
            ref = next(iter(opts))[0]
            if not impose(inc, t if ref else t.times(-1)):
                return None
        if len(opts) == 1:
            ref, neu = next(iter(opts))
            young = os[ref, neu][0]
            if young is not None and (not impose(5, young if neu else young.times(-1))):
                return None
    return bounds


def bounds_update(options, actions, bounds):
    """An equilibrium value is an expectation over its supported campaign outcomes. Intersect with their convex-hull bounds and round outwards on a rational grid."""
    new = {}
    for i in ACTORS:
        new[i] = []
        for inc in P:
            ranges = [v[i].range(bounds[i]) for pair in product(actions[inc, 0], actions[inc, 1]) for v in options[inc, pair].values()]
            lo = max(bounds[i][inc][0], min((x for x, y in ranges)))
            hi = min(bounds[i][inc][1], max((y for x, y in ranges)))
            if lo > hi:
                return None
            grid = 10 ** 14
            lo = F(lo * grid // 1, grid)
            hi = -F(-hi * grid // 1, grid)
            new[i].append((lo, hi))
    return new


def default_bounds():
    """Initial valid bounds for party values and the expected democratic old-voter flow including valence. The accompanying note derives them."""
    return {0: [(F(-30), F(30))] * 2, 1: [(F(-30), F(30))] * 2, 5: [(F(35), F(55))] * 2}


def prune(block, force=None, bounds=None, allowed=None):
    """Repeatedly intersect necessary conditions. Stopping early only enlarges the enclosure."""
    force = force or {}
    bounds = bounds or default_bounds()
    allowed = allowed or {(i, j): set(P) for i in P for j in P}
    for n in range(60):
        eligible = choices(block, bounds, allowed, force)
        if eligible is None:
            return None
        options, acts = eligible
        constrained = tighten(block, options, bounds)
        if constrained is None:
            return None
        new = bounds_update(options, acts, constrained)
        if new is None:
            return None
        if new == bounds and acts == allowed:
            break
        bounds, allowed = (new, acts)
    return (options, acts, bounds)



def explore(block, h):
    """Enumerate both incumbent games at this record. Split at every unresolved deterministic refusal or neutral certificate. Collapse the equivalent campaign labels when both posterior means vanish."""
    leaves = []
    unresolved = []
    stats = Counter()
    initial = {(j, i): {0} if h == (0, 2) else set(P) for j in P for i in P}

    def visit(force, box, allowed):
        stats['calls'] += 1
        rs = prune(block, force, box, allowed)
        if rs is None:
            stats['rejected'] += 1
            return
        opts, acts, box = rs
        acts_on = [len(acts[i, j]) for i in P for j in P]
        if max(acts_on) == 1:
            pairs = [tuple((next(iter(acts[j, i])) for i in P)) for j in P]
            if all((len(opts[j, pairs[j]]) == 1 for j in P)):
                desc = tuple(((p, next(iter(opts[j, p]))) for j, p in enumerate(pairs)))
                leaves.append((desc, force))
                return
        split = [(p not in product(acts[j, 0], acts[j, 1]), j, p) for (j, p), options in opts.items() if len(options) > 1]
        if not split:
            unresolved.append((opts, acts, box, force))
            return
        _, j, p = min(split)
        key = (j, p)
        for d in opts[key]:
            visit({**force, key: {d}}, box, acts)
    visit({}, None, initial)
    return (leaves, unresolved, stats)



def bound_mixed(opts, acts, box):
    """In a two-by-two game with a strict best-reply cycle, bound the mixing probabilities by each party's indifference equation. Intersect continuation boxes with the implied payoff bounds."""
    mixtures = {}
    for inc in P:
        for i in P:
            if len(acts[inc, i]) == 1:
                e = next(iter(acts[inc, i]))
                mixtures[inc, i] = (e, e)
            else:
                mixtures[inc, i] = (F(0), F(1))
        if any((len(acts[inc, i]) == 1 for i in P)):
            continue
        diffs = {}
        for i in P:
            diffs[i] = []
            for other in P:
                pos = m.ordered(i, 1, other)
                neg = m.ordered(i, 0, other)
                d = [v[i].minus(w[i]).range(box[i]) for v in opts[inc, pos].values() for w in opts[inc, neg].values()]
                diffs[i].append((min((l for l, u in d)), max((u for l, u in d))))
        signs = {i: tuple((1 if l > 0 else -1 if u < 0 else 0 for l, u in pair)) for i, pair in diffs.items()}
        if any((0 in s for s in signs.values())) or signs[0] not in [(1, -1), (-1, 1)] or signs[1] != tuple((-x for x in signs[0])):
            continue
        for i, arr in diffs.items():
            (lo, hi), (l2, h2) = [(l, u) if l > 0 else (-u, -l) for l, u in arr]
            mixtures[inc, 1 - i] = (lo / (lo + h2), hi / (hi + l2))
    new = {i: [] for i in ACTORS}
    for i in ACTORS:
        for inc in P:
            can = {pair: [v[i].range(box[i]) for v in opts[inc, pair].values()] for pair in m.OFFERS}
            low = {pair: min((v[0] for v in d)) for pair, d in can.items()}
            high = {pair: max((v[1] for v in d)) for pair, d in can.items()}

            def ev(pl, pr, b):
                return sum((x * (pl if p[0] else 1 - pl) * (pr if p[1] else 1 - pr) for p, x in b.items()))
            candidates = list(product(mixtures[inc, 0], mixtures[inc, 1]))
            lo = min((ev(pl, pr, low) for pl, pr in candidates))
            hi = max((ev(pl, pr, high) for pl, pr in candidates))
            if i != 5 and all((0 < l <= u < 1 for l, u in [mixtures[inc, 0], mixtures[inc, 1]])):
                for r in P:
                    cs = [tuple((r if j == i else x for j in P)) for x in mixtures[inc, 1 - i]]
                    lo = max(lo, min((ev(pl, pr, low) for pl, pr in cs)))
                    hi = min(hi, max((ev(pl, pr, high) for pl, pr in cs)))
            lo = max(box[i][inc][0], lo)
            hi = min(box[i][inc][1], hi)
            if lo > hi:
                return None
            grid = 10 ** 12
            new[i].append((F(lo * grid // 1, grid), -F(-hi * grid // 1, grid)))
    return (new, mixtures)



def initial_values(s, k):
    """Fill unsolved records with a dummy profile. Only its autocracy values are used."""
    profile = {}
    for h in m.RECORDS:
        for incumbent in P:
            decision = m.Decision(False, (False, True, True), ((1-incumbent,), ()),
                                  F(1, 2), F(11, 20), F(0))
            profile[incumbent, h] = (tuple(F(p == (0, 0)) for p in m.OFFERS),
                                     {(0, 0): decision})
    return m.evaluate(profile, s, k), profile


def maybe_copy(profile, h, desc, s, k):
    """Solve candidate values exactly and verify the mandates, both age groups' vote, enforcement and all unilateral deviations."""
    new = profile.copy()
    for inc, (pair, (r, n)) in enumerate(desc):
        spec = m.Decision(bool(r), (False, bool(n), True), ((1 - inc,), ()), F(1, 2), F(9, 20) + F(n, 10), F(0))
        probs = tuple((F(x == pair) for x in m.OFFERS))
        new[inc, h] = (probs, {pair: spec})
    val = m.evaluate(new, s, k)
    st = m.Stage(val, attempt_cost=k)
    for inc, (pair, (r, n)) in enumerate(desc):
        game = {pr: st.contest(inc, h, pr, s) for pr in m.OFFERS}
        v, d = game[pair]
        if v != val[m.DEM, inc, h] or d.refuse != bool(r) or d.ballots != (False, bool(n), True) or (d.success != F(1, 2)):
            return None
        for i in P:
            alt = tuple((1 - y if i == j else y for j, y in enumerate(pair)))
            if game[alt][0][i] > v[i]:
                return None
        new[inc, h] = (new[inc, h][0], {p: el[1] for p, el in game.items()})
    return (val, new)


def describe(profile):
    return [[(m.pure_pair(profile[j, h][0]), profile[j, h][1][m.pure_pair(profile[j, h][0])].refuse) for j in P] for h in m.RECORDS]


def classify(s, k, verbose=True):
    """Process posterior supports in increasing size. Preserve each economically distinct equilibrium continuation for less informative records."""
    cases = [initial_values(s, k)]
    leftovers = 0
    for h in m.RECORDS:
        next_cases = []
        for val, profile in cases:
            block = block_data(h, s, k, val)
            candidates, remainders, stats = explore(block, h)
            for options, actions, box, force in remainders:
                tightened = bound_mixed(options, actions, box)
                if tightened is not None:
                    new_box, _ = tightened
                    if prune(block, force, new_box, actions) is not None:
                        leftovers += 1
            for desc in sorted(set((x for x, force in candidates))):
                sol = maybe_copy(profile, h, desc, s, k)
                if sol is not None:
                    next_cases.append(sol)
        cases = next_cases
        if verbose:
            print('records', h, 'profiles', len(cases), 'leftovers', leftovers, flush=True)
    return (cases, leftovers)



def main():
    expected = {
        (F(".05"), F("2.55")): [
            (F(6578735941,25753024084), F(6578735941,25753024084)),
            (F(6508495981,12876512042), F(6508495981,12876512042)),
        ],
        (F(".14"), F("2.55")): [
            (F(791,1282), F(63007,102560)),
            (F(2073,2564), F(165567,205120)),
        ],
        (F(".05"), F("2.40")): [
            (F(19047,23407), F(6061929,7490240)),
        ],
    }
    for (s,k), expected_risks in expected.items():
        solutions, unresolved = classify(s,k,verbose=False)
        assert unresolved == 0
        risks = []
        for values, profile in solutions:
            risk = eventually_autocratic(profile,s,k)
            risk_pair = tuple(risk[m.DEM,j,m.PRIOR] for j in P)
            risks.append(risk_pair)
            for (inc,h), (campaign,choices) in profile.items():
                mandated = m.pure_pair(campaign)
                assert mandated == ((0,0) if h == (0,2) else (1,1))
                assert all(d.success == F(1,2) for d in choices.values())
            print("S, k, eventual takeover probabilities (L,R)", s,k,risk_pair)
            if s == F(".05"):
                verify_initial(m.Stage(values,attempt_cost=k), "current monitoring, fixed tail")
        assert sorted(risks) == sorted(expected_risks)
        print("Exhaustive outcome count", len(risks))


if __name__ == "__main__":
    main()
