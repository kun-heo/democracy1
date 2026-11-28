"""Stationary elections with citizen-backed refusal. See the manuscript's Appendix D.

Both parties choose flat/full mandates. Elections continue at every information
state, including a known technology. Markov strategies are verified at all eight
incumbent/mandate combinations, with individual votes and endorsements.
"""
import argparse
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.polynomial.hermite import hermgauss

from verify_citizen_support import (BASE, SHOCKS, SHOCK_PROB,
                                    DISCOUNT, Institutions, flow, L, R)

OFFERS = np.array(list(product(range(2), repeat=2)))  # columns ell and r
# Columns index the offer pair, refusal after a loss, and an R neutral certificate.
CAND = np.array(list(product(range(4), range(2), range(2))))
CAND_L, CAND_R = (x.ravel() for x in np.meshgrid(np.arange(len(CAND)),
                                                  np.arange(len(CAND)), indexing='ij'))
PARTIES = np.arange(2)
VALENCE = SHOCK_PROB[-1] * SHOCKS[-1]
TIE = 5e-12


def electoral_count(margins, weights):
    """Equal cohorts. A citizen tie or a tie in the aggregate count goes to L."""
    return (((margins > TIE) @ weights).mean(axis=-1) > .5 + TIE)


@dataclass
class Branches:
    # Array axes are ruling party, mandate (also party's ideal at index 2),
    # and actor (two parties followed by four citizen types).
    f: np.ndarray
    d: np.ndarray  # democratic branch, including all continuations
    a: np.ndarray  # entrenchment now; party uses its mandate for this term


@dataclass
class Plan:
    # Arrays prior to campaign are for both incumbent identities.
    q: np.ndarray
    refuse: np.ndarray
    pr: np.ndarray  # probability R wins the count
    value: np.ndarray  # party payoff or expected old citizen flow
    incentives: np.ndarray
    young_support: np.ndarray


def play(b, inst, protect=False, all_shocks=False, *, fixed_neutral=False):
    """Backward induction within each election. Optional leading axis: profiles."""
    q_list, ref_list, prob_list, val_list, gain_list, sy_list = ([] for _ in range(6))
    ell, r = OFFERS.T
    flows = [b.f[L, ell], b.f[R, r]]
    dem = [b.d[..., L, ell, :], b.d[..., R, r, :]]
    aut = [b.a[..., L, ell, :], b.a[..., R, r, :]]
    for inc in PARTIES:
        other = 1 - inc
        old_gain = flows[inc] - flows[other]
        gain = aut[inc] - dem[other]
        old_votes = (old_gain[..., 2:] > TIE) @ inst.weights
        young_votes = (gain[..., 2:] > TIE) @ inst.weights
        q = inst.success(.5 * (old_votes + young_votes))
        incentive = q * gain[..., inc] - inst.k
        refuse = incentive > TIE
        if protect:
            refuse = refuse & (OFFERS[:, other] != 0)
        tau = q * refuse
        sign = 2 * inc - 1
        old_margin = sign * (1 - tau[..., None]) * old_gain[..., 2:]
        young_margin = sign * (dem[inc] - dem[other] - tau[..., None] * gain)[..., 2:]
        margins = np.stack([old_margin, young_margin], axis=-2)
        if all_shocks:
            votes = np.stack([electoral_count(margins + h, inst.weights) for h in SHOCKS], axis=-1)
            if fixed_neutral:
                votes[..., SHOCKS == 0] = True
            pr = votes @ SHOCK_PROB
            valence = (votes * SHOCKS) @ SHOCK_PROB
        else:
            neutral = np.ones_like(refuse) if fixed_neutral else electoral_count(margins, inst.weights)
            pr = .45 + .10 * neutral
            valence = VALENCE
        lose = (1 - pr) if inc == R else pr
        seated = lose * (1 - tau)
        value = np.empty((*pr.shape, 6))
        value[..., :2] = ((1 - lose[..., None]) * dem[inc][..., :2]
                            + lose[..., None] * (dem[other] + tau[..., None] * gain)[..., :2])
        value[..., inc] -= lose * refuse * inst.k
        value[..., 2:] = ((1 - seated[..., None]) * flows[inc][..., 2:]
                          + seated[..., None] * flows[other][..., 2:]
                          + np.asarray(valence)[..., None])
        for target, item in zip((q_list, ref_list, prob_list, val_list, gain_list, sy_list),
                               (q, refuse, pr, value, incentive, young_votes)):
            target.append(item)
    return Plan(q=np.stack(q_list, axis=-2), refuse=np.stack(ref_list, axis=-2),
                pr=np.stack(prob_list, axis=-2), value=np.stack(val_list, axis=-3),
                incentives=np.stack(gain_list, axis=-2),
                young_support=np.stack(sy_list, axis=-2))


def campaign_nash(values):
    """For each incumbent check both unilateral campaign deviations."""
    idx = np.arange(4)
    return ((values[..., idx, L] >= values[..., idx ^ 2, L] - TIE)
            & (values[..., idx, R] >= values[..., idx ^ 1, R] - TIE))


def solve_two(matrix, rhs):
    """Solve any number of two-incumbent Bellman systems, across actor columns."""
    a, b = matrix[..., 0, 0], matrix[..., 0, 1]
    c, d = matrix[..., 1, 0], matrix[..., 1, 1]
    det = (a * d - b * c)[..., None]
    if np.any(np.abs(det) < 1e-14):
        raise ArithmeticError('singular Bellman matrix')
    return np.stack([(d[..., None] * rhs[..., 0, :] - b[..., None] * rhs[..., 1, :]) / det,
                     (a[..., None] * rhs[..., 1, :] - c[..., None] * rhs[..., 0, :]) / det], axis=-2)


def stationary_candidates(f, aut, future, stay, inst, fixed_neutral=False):
    """Endogenous q from the old. Verify young support again in play().
    future = current flow + discounted democratic values at strictly later
    information states; stay is the probability the public record does not move.
    """
    candidates, matrices = [], []
    for inc in PARTIES:
        other = 1 - inc
        pair, ref, neutral = CAND.T
        iw, jw = OFFERS[pair, inc], OFFERS[pair, other]
        q = inst.success(.5 * ((f[inc, iw, 2:] - f[other, jw, 2:] > TIE) @ inst.weights))
        lose = .45 + .10 * (neutral if inc == L else 1 - neutral)
        tau = ref * q
        seated = lose * (1 - tau)
        entrench = lose * tau
        val = np.empty((len(CAND), 6))
        val[:, :2] = ((1 - lose[:, None]) * future[inc, iw, :2]
                     + seated[:, None] * future[other, jw, :2] + entrench[:, None] * aut[inc, iw, :2])
        val[:, inc] -= lose * ref * inst.k
        val[:, 2:] = (1 - seated[:, None]) * f[inc, iw, 2:] + seated[:, None] * f[other, jw, 2:] + VALENCE
        coeff = np.zeros((len(CAND), 2))
        coeff[:, inc] = 1 - DISCOUNT * (1 - lose) * stay[inc, iw]
        coeff[:, other] = -DISCOUNT * seated * stay[other, jw]
        candidates.append(val)
        matrices.append(coeff)
    rhs = np.stack([candidates[L][CAND_L], candidates[R][CAND_R]], axis=1)
    matrices = np.stack([matrices[L][CAND_L], matrices[R][CAND_R]], axis=1)
    vals = rhs.copy()
    vals[..., :2] = solve_two(matrices, rhs[..., :2])
    dem = future + DISCOUNT * stay[..., None] * vals[:, :, None, :]
    strategies = play(Branches(f, dem, aut), inst, fixed_neutral=fixed_neutral)
    nash = campaign_nash(strategies.value)
    ids = np.arange(len(vals))
    valid = np.ones(len(vals), dtype=bool)
    for inc, columns in ((L, CAND_L), (R, CAND_R)):
        pair, ref, neutral = CAND[columns].T
        key = (ids, inc, pair)
        valid &= nash[key] & (strategies.refuse[key] == ref)
        valid &= np.abs(strategies.pr[key] - (.45 + .10 * neutral)) < TIE
        valid &= (strategies.young_support[key] == 0)
    return vals, np.stack([CAND[CAND_L], CAND[CAND_R]], axis=1), np.flatnonzero(valid)


def offer_probabilities(full):
    """Independent mixing by the two parties, by incumbent and offer pair."""
    return np.prod(np.where(OFFERS, full[:, None, :], 1 - full[:, None, :]), axis=-1)


def choice_law(f, aut, future, stay, inst, response):
    const = np.empty((2, 4, 6))
    law = np.zeros((2, 4, 2))
    for inc in PARTIES:
        other = 1 - inc
        iw, jw = OFFERS[:, inc], OFFERS[:, other]
        lose = response.pr[inc] if inc == L else 1 - response.pr[inc]
        tau = response.q[inc] * response.refuse[inc]
        seat = lose * (1 - tau)
        success = lose * tau
        const[inc, :, :2] = ((1 - lose[:, None]) * future[inc, iw, :2]
                         + seat[:, None] * future[other, jw, :2] + success[:, None] * aut[inc, iw, :2])
        const[inc, :, inc] -= lose * response.refuse[inc] * inst.k
        const[inc, :, 2:] = (1 - seat[:, None]) * f[inc, iw, 2:] + seat[:, None] * f[other, jw, 2:] + VALENCE
        law[inc, :, inc] = (1 - lose) * stay[inc, iw]
        law[inc, :, other] = seat * stay[other, jw]
    return const, law


def price(full, const, law):
    prob = offer_probabilities(full)
    result = np.einsum('ip,ipj->ij', prob, const)
    transition = np.einsum('ip,ipj->ij', prob, law)
    result[:, :2] = solve_two(np.eye(2) - DISCOUNT * transition, result[:, :2])
    return result


def regret(pay, full):
    prob = offer_probabilities(full)
    actual = np.einsum('ip,ipj->ij', prob, pay[..., :2])
    gains = []
    for who in PARTIES:
        for action in (0., 1.):
            dev = full.copy()
            dev[:, who] = action
            alt = np.sum(offer_probabilities(dev) * pay[:, :, who], axis=1)
            gains.extend(alt - actual[:, who])
    return max(0., max(gains))


def stage_nash(pay):
    pure = np.flatnonzero(campaign_nash(pay))
    if len(pure):
        return OFFERS[pure[-1]].astype(float)
    mat = pay[:, :2].reshape(2, 2, 2)
    incentive_l = mat[1, :, L] - mat[0, :, L]
    incentive_r = mat[:, 1, R] - mat[:, 0, R]
    return np.array([incentive_r[0] / (incentive_r[0] - incentive_r[1]),
                     incentive_l[0] / (incentive_l[0] - incentive_l[1])])


def mixed_candidates(f, a, future, stay, inst, pure_values, fixed_neutral=False):
    """Search support patterns, then certify the returned equilibrium independently.

    Within a fixed transfer/ballot pattern each actor's indifferences depend
    only on the opponent's mixing once one supported own action is fixed.
    """
    def assess(v):
        return play(Branches(f, future + DISCOUNT * stay[..., None] * v[:, None, :], a),
                    inst, fixed_neutral=fixed_neutral)

    def certify(full, v):
        if np.any(full < -1e-9) or np.any(full > 1 + 1e-9):
            return False
        r = assess(v)
        x = offer_probabilities(full)
        gap = np.max(abs(np.einsum('ip,ipj->ij', x, r.value) - v))
        return gap < 5e-11 and regret(r.value, full) < 5e-11

    patterns = {}
    def keep(r):
        key = (tuple(r.refuse.ravel()), tuple((r.pr > .5).ravel()))
        if key not in patterns:
            patterns[key] = choice_law(f, a, future, stay, inst, r)
    # First follow the ordinary stage Nash correspondence. These iterations
    # generate candidates. Only the residual and deviation checks certify them.
    v = pure_values[-1].copy()
    for _ in range(55):
        rp = assess(v)
        keep(rp)
        full = np.array([stage_nash(row) for row in rp.value])
        const, law = choice_law(f, a, future, stay, inst, rp)
        nxt = price(full, const, law)
        if np.max(abs(v - nxt)) < 3e-10 and certify(full, nxt):
            return full, nxt
        v = .4 * v + .6 * nxt
    # Additional patterns implied by deterministic campaign continuation values.
    dem = future + DISCOUNT * stay[..., None] * pure_values[:, :, None, :]
    stage = play(Branches(f, dem, a), inst)
    flags = np.c_[stage.refuse.reshape(-1, 8), (stage.pr > .5).reshape(-1, 8)]
    for j in np.unique(flags, axis=0, return_index=True)[1]:
        keep(assess(pure_values[j]))
    # The generic one-dimensional support cases include mixing in different
    # incumbent states. Test all four choices of the two remaining pure actions.
    for const, law in patterns.values():
        for state_l, state_r in product(PARTIES, repeat=2):
            for pure_l, pure_r in product((0., 1.), repeat=2):
                base = np.array([[pure_l, pure_r], [pure_l, pure_r]])
                base[state_l, L] = base[state_r, R] = 0.
                def residual(who, other_mix):
                    own_state, other_state = (state_l, state_r) if who == L else (state_r, state_l)
                    full = base.copy()
                    full[other_state, 1 - who] = other_mix
                    prob = offer_probabilities(full)
                    trans = np.einsum('ip,ipj->ij', prob, law)
                    b = np.sum(prob * const[:, :, who], axis=1)
                    matrix = np.eye(2) - DISCOUNT * trans
                    val = solve_two(matrix, b[:, None])[:, 0]
                    q = const[own_state, :, who] + DISCOUNT * (law[own_state] @ val)
                    opponent = full[own_state, 1 - who]
                    delta = (q[2:] - q[:2]) if who == L else (q[1::2] - q[::2])
                    return np.linalg.det(matrix) * ((1 - opponent) * delta[0] + opponent * delta[1])
                roots = []
                for who in PARTIES:
                    v0, vm, v1 = (residual(who, u) for u in (0., .5, 1.))
                    c2 = 2 * (v1 + v0 - 2 * vm)
                    c1 = v1 - v0 - c2
                    if abs(c2) < 1e-10:
                        candidates = [-v0 / c1] if abs(c1) > 1e-11 else []
                    else:
                        candidates = [z.real for z in np.roots([c2, c1, v0]) if abs(z.imag) < 1e-8]
                    roots.append([z for z in candidates if 1e-9 < z < 1 - 1e-9])
                for pr, pl in product(*roots):
                    full = base.copy()
                    full[state_l, L] = pl
                    full[state_r, R] = pr
                    val = price(full, const, law)
                    if certify(full, val):
                        return full, val
    return None


class GaussianGrid:
    """Finite Markov approximation in a,b and information about theta_+, theta_-."""
    def __init__(self, a, b, precision, order=15):
        self.a, self.b, self.precision = np.asarray(a), np.asarray(b), np.asarray(precision)
        self.shape = (len(precision), len(precision), len(a), len(b))
        self.states = list(np.ndindex(self.shape))
        self.nodes, self.mass = hermgauss(order)
        self.nodes *= np.sqrt(2)
        self.mass /= np.sqrt(np.pi)
        base_precision = np.linalg.inv(np.array([[.26, -.24], [-.24, .26]]))
        self.cov = {}
        for i, j in product(range(len(precision)), repeat=2):
            u = self.precision[[i, j]]
            finite = np.flatnonzero(np.isfinite(u))
            cov = np.zeros((2, 2))
            if len(finite):
                part = base_precision[np.ix_(finite, finite)] + np.diag(u[finite])
                cov[np.ix_(finite, finite)] = np.linalg.inv(part)
            self.cov[i, j] = cov
        self.order = sorted(range(len(self.states)), key=lambda n:sum(self.states[n][:2]), reverse=True)
        self._kernels = {}

    def prior_state(self):
        return self.index(0, 0, np.argmin(abs(self.a)), np.argmin(abs(self.b - .5)))

    def index(self, i, j, k, l):
        return np.ravel_multi_index((i, j, k, l), self.shape)

    @staticmethod
    def interpolate(grid, x):
        if len(grid) == 1:
            return [(np.zeros_like(x, dtype=int), np.ones_like(x))]
        right = np.clip(np.searchsorted(grid, x), 1, len(grid) - 1)
        frac = np.clip((x - grid[right - 1]) / (grid[right] - grid[right - 1]), 0., 1.)
        return [(right - 1, 1 - frac), (right, frac)]

    def precision_steps(self, index, direction, x):
        """Lottery over two adjacent precisions matches Gaussian posterior variance."""
        pos = index[direction]
        cov = self.cov[index]
        var = cov[direction, direction]
        if var == 0 or x == 0:
            return [(index, 1.)]
        target = self.precision[pos] + x * x / .0625
        hi = int(np.searchsorted(self.precision, target, side='left'))
        lo = max(pos, hi - 1)
        idxs = []
        for destination in (lo, hi):
            key = list(index)
            key[direction] = destination
            idxs.append(tuple(key))
        lower, upper = [self.cov[k][direction, direction] for k in idxs]
        exact = var / (1 + var * x * x / .0625)
        prob = np.clip((lower - exact) / (lower - upper), 0., 1.)
        return [(idxs[0], 1 - prob), (idxs[1], prob)]

    def kernel_endpoint(self, state, party, offer, report):
        key = state, party, offer, report
        if key in self._kernels:
            return self._kernels[key]
        i, j, k, l = self.states[state]
        a, b = self.a[k], self.b[l]
        w = (2 * party - 1) * offer
        cov = self.cov[i, j]
        result = {}
        for direction, omega in enumerate((1, -1)):
            x = BASE[party] + w * (a + report * b * omega)
            for nxt, prob in self.precision_steps((i, j), direction, x):
                if prob <= 1e-16:
                    continue
                if nxt == (i, j):
                    result[state] = result.get(state, 0.) + .5 * prob
                    continue
                var = cov[direction, direction]
                newvar = self.cov[nxt][direction, direction]
                step = cov[:, direction] / var * np.sqrt(max(var - newvar, 0.))
                new_a = a + .5 * (step[0] + step[1]) * self.nodes
                new_b = b + .5 * (step[0] - step[1]) * self.nodes
                for ak, aw in self.interpolate(self.a, new_a):
                    for bl, bw in self.interpolate(self.b, new_b):
                        dest = self.index(*nxt, ak, bl)
                        for h, weight in zip(dest, .5 * prob * self.mass * aw * bw):
                            if weight > 0:
                                result[h] = result.get(h, 0.) + weight
        assert abs(sum(result.values()) - 1) < 2e-12
        ids = np.array(sorted(result), dtype=int)
        weights = np.array([result[n] for n in ids])
        self._kernels[key] = ids, weights
        return ids, weights


def material(grid, state, s, left_flat=0.):
    _, _, k, l = grid.states[state]
    a, b = grid.a[k], grid.b[l]
    weights = (-np.array([left_flat, 1., 2.]), np.arange(3))
    return np.stack([flow(BASE[j], weights[j], a, a * a + s * b * b) for j in PARTIES])


def transitions(grid, state, s, left_flat=0.):
    out = []
    for party in PARTIES:
        row = []
        for offer in (left_flat if party == L else 0., 1., 2.):
            mix = {}
            for report, prob in [(0, 1 - s), (1, s)]:
                if prob == 0:
                    continue
                ids, weights = grid.kernel_endpoint(state, party, offer, report)
                for h, p in zip(ids, weights):
                    mix[h] = mix.get(h, 0.) + prob * p
            row.append(mix)
        out.append(row)
    return out


def split_expectation(state, kernel, values):
    n_actors = values.shape[-1]
    stay = np.zeros((2, 3))
    higher = np.zeros((2, 3, n_actors))
    for j, w in product(range(2), range(3)):
        move = kernel[j][w]
        stay[j, w] = move.get(state, 0.)
        dest = np.array([n for n in move if n != state], dtype=int)
        if len(dest):
            higher[j, w] = np.array([move[n] for n in dest]) @ values[dest, j]
    return stay, higher


def absorbing_risk(local, higher):
    """Only states that can reach success have positive eventual takeover risk."""
    active = higher > 0
    active |= (local @ active.astype(float)) > 0
    risk = np.zeros(2)
    if any(active):
        sub = local[np.ix_(active, active)]
        risk[active] = np.linalg.solve(np.eye(sum(active)) - sub, higher[active])
    return risk


@dataclass
class Solution:
    grid: GaussianGrid
    s: float
    inst: Institutions
    values: np.ndarray
    auto: np.ndarray
    policy: np.ndarray
    risk: np.ndarray
    multiplicity: np.ndarray
    residual: float
    fixed_neutral: bool = False


def solve(grid, s, inst, select='concede', *, fixed_neutral=False):
    n = len(grid.states)
    vals = np.zeros((n, 2, 6))
    auto = np.zeros_like(vals)
    policy = np.zeros((n, 2, 2))
    risk = np.zeros((n, 2))
    multi = np.zeros(n, int)
    residual = 0.
    for state in grid.order:
        kernel = transitions(grid, state, s)
        f = material(grid, state, s)
        stay, ad = split_expectation(state, kernel, auto)
        auto[state, :, :2] = (f[:, 2, :2] + DISCOUNT * ad[:, 2, :2]) / (1 - DISCOUNT * stay[:, 2, None])
        auto[state, :, 2:] = f[:, 2, 2:]
        a = f + DISCOUNT * (ad + stay[..., None] * auto[state, :, None, :])
        _, dd = split_expectation(state, kernel, vals)
        future = f + DISCOUNT * dd
        v, profiles, ok = stationary_candidates(f, a, future, stay, inst, fixed_neutral)
        if len(ok):
            sign = 1 if select == 'concede' else -1
            def rank(i):
                p = profiles[i]
                return (sign * sum(p[:, 1]), -int(OFFERS[p[:, 0]].sum()), tuple(p.ravel()))
            pick = min(ok, key=rank)
            vals[state] = v[pick]
            policy[state] = OFFERS[profiles[pick, :, 0]]
        else:
            mixed = mixed_candidates(f, a, future, stay, inst, v, fixed_neutral)
            if mixed is None:
                raise RuntimeError(f'No certified profile at {grid.states[state]}, S={s}, select={select}')
            policy[state], vals[state] = mixed
        multi[state] = len(ok)
        d = future + DISCOUNT * stay[..., None] * vals[state, :, None, :]
        resp = play(Branches(f, d, a), inst, all_shocks=True, fixed_neutral=fixed_neutral)
        prob = offer_probabilities(policy[state])
        expected = np.einsum('ip,ipj->ij', prob, resp.value)
        err = max(np.max(abs(expected - vals[state])), regret(resp.value, policy[state]))
        residual = max(err, residual)
        assert err < 5e-10 and np.max(resp.young_support) == 0
        stable = play(Branches(f, d, a), inst, fixed_neutral=fixed_neutral)
        assert np.max(abs(resp.pr - stable.pr)) < TIE and np.max(abs(resp.value - stable.value)) < 5e-11
        higher = np.zeros(2)
        local = np.zeros((2, 2))
        for inc, pair in product(PARTIES, range(4)):
            chance_pair = prob[inc, pair]
            if chance_pair == 0:
                continue
            opp = 1 - inc
            pr = resp.pr[inc, pair]
            lose = pr if inc == L else 1 - pr
            z = lose * resp.q[inc, pair] * resp.refuse[inc, pair]
            higher[inc] += chance_pair * z
            for j, chance in ((inc, 1 - lose), (opp, lose - z)):
                for h, p in kernel[j][OFFERS[pair, j]].items():
                    t = chance_pair * chance * p
                    if h == state:
                        local[inc, j] += t
                    else:
                        higher[inc] += t * risk[h, j]
        risk[state] = absorbing_risk(local, higher)
    assert np.min(risk) > -1e-10 and np.max(risk) < 1. + 1e-10
    return Solution(grid, s, inst, vals, auto, policy, risk, multi, residual, fixed_neutral)


def current(sol, state, s, *, left_flat=0.):
    """Current branches, optionally pricing a one-time L deviation in the flat slot."""
    grid = sol.grid
    f = material(grid, state, s, left_flat)
    kernel = transitions(grid, state, s, left_flat)
    stay, next_d = split_expectation(state, kernel, sol.values)
    _, next_a = split_expectation(state, kernel, sol.auto)
    d = f + DISCOUNT * (next_d + stay[..., None] * sol.values[state, :, None, :])
    a = f + DISCOUNT * (next_a + stay[..., None] * sol.auto[state, :, None, :])
    return Branches(f, d, a)


def campaign(response, inc):
    full = stage_nash(response.value[inc])
    prob = np.prod(np.where(OFFERS, full, 1 - full), axis=-1)
    lose = response.pr[inc] if inc == L else 1 - response.pr[inc]
    attempt = response.refuse[inc]
    return dict(full=full, payoff=prob @ response.value[inc], loss=prob @ lose,
                attempt=prob @ (lose * attempt), installed=prob @ (lose * (1 - attempt * response.q[inc])),
                q=prob @ response.q[inc])


def mix_branches(low, high, s):
    return Branches(*(x + s * (y - x) for x, y in zip((low.f, low.d, low.a), (high.f, high.d, high.a))))


def forecast_cuts(low, high, inst):
    """Partition S at every citizen, refusal, ballot and campaign indifference."""
    def divide(cuts, values):
        new = set(cuts)
        for left, right in zip(cuts, cuts[1:]):
            a = (2 * left + right) / 3
            b = (left + 2 * right) / 3
            va, vb = np.asarray(values(a)).ravel(), np.asarray(values(b)).ravel()
            nonzero = abs(vb - va) > 1.e-12
            for x in a - va[nonzero] * (b - a) / (vb[nonzero] - va[nonzero]):
                if left + 1e-10 < x < right - 1e-10:
                    new.add(round(float(x), 12))
        return sorted(new)
    def profile(s):
        return mix_branches(low, high, s)
    def willingness(s):
        b = profile(s)
        all_margins = []
        for i in PARTIES:
            j = 1 - i
            iw, jw = OFFERS[:, i], OFFERS[:, j]
            all_margins.extend((b.f[i, iw, 2:] - b.f[j, jw, 2:], b.a[i, iw, 2:] - b.d[j, jw, 2:]))
        return all_margins
    cuts = divide([0., 1.], willingness)
    cuts = divide(cuts, lambda s:play(profile(s), inst).incentives)
    def ballots(s):
        b = profile(s)
        all_margins = []
        for prot in (False, True):
            p = play(b, inst, prot)
            for i in PARTIES:
                j = 1 - i
                iw, jw = OFFERS[:, i], OFFERS[:, j]
                tau = (p.refuse[i] * p.q[i])[:, None]
                old = (2 * i - 1) * (1 - tau) * (b.f[i, iw, 2:] - b.f[j, jw, 2:])
                young = (2 * i - 1) * (b.d[i, iw, 2:] - b.d[j, jw, 2:] - tau * (b.a[i, iw, 2:] - b.d[j, jw, 2:]))
                for shock in SHOCKS:
                    all_margins.extend((old + shock, young + shock))
        return all_margins
    cuts = divide(cuts, ballots)
    def indifference(s):
        gaps = []
        for prot in (False, True):
            p = play(profile(s), inst, prot, all_shocks=True)
            for who in PARTIES:
                gaps.append(p.value[:, :, who] - p.value[:, np.arange(4) ^ (2 - who), who])
        return gaps
    return divide(cuts, indifference)


def cutoff_table(sol):
    """Certified intervals for the initial R incumbent. In the reported rows
    campaigns are pure in every open current-S cell, so repeal gains are affine.
    At a boundary supports and optimal actions are evaluated again from scratch.
    """
    state = sol.grid.prior_state()
    inst = sol.inst
    low, high = [current(sol, state, s) for s in (0., 1.)]
    def outcomes(s):
        b = mix_branches(low, high, s)
        unprotected = play(b, inst, all_shocks=True)
        protected = play(b, inst, True, all_shocks=True)
        free, kept = (campaign(response, R) for response in (unprotected, protected))
        gain = free['payoff'][R] - kept['payoff'][R]
        return unprotected, free, kept, gain
    cuts = forecast_cuts(low, high, inst)
    additions = []
    for a, b in zip(cuts, cuts[1:]):
        left, right = (2 * a + b) / 3, (a + 2 * b) / 3
        _, free, kept, y = outcomes(left)
        assert all(np.all((o['full'] == 0) | (o['full'] == 1)) for o in (free, kept))
        _, _, _, z = outcomes(right)
        if abs(z - y) > 1e-12:
            for cost in (0., .01):
                root = left + (cost - y) * (right - left) / (z - y)
                if a + 1e-9 < root < b - 1e-9:
                    additions.append(root)
    cuts = sorted(set(cuts + additions))
    def flags(s):
        response, free, kept, gain = outcomes(s)
        # A counterfactual flat offer is evaluated against R's equilibrium offer.
        assert free['full'][R] == 1.
        return (not response.refuse[R, 1], gain > 2e-11, gain > .01 + 2e-11)
    samples = sorted(set(cuts + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])]))
    rows = [flags(s) for s in samples]
    for left, right in zip(rows, rows[1:]):
        assert int(left[0]) >= int(right[0]) and all(int(left[j]) <= int(right[j]) for j in (1, 2))
    mids = [flags((a + b) / 2) for a, b in zip(cuts, cuts[1:])]
    voluntary = [b for (a, b), flag in zip(zip(cuts, cuts[1:]), mids) if flag[0]]
    result = [max(voluntary) if voluntary else None]
    for j in (1, 2):
        starts = [a for a, flag in zip(cuts, mids) if flag[j]]
        result.append(min(starts) if starts else None)
    return result, outcomes


def next_return(sol, s, offer=0):
    """R's next certificate and installation conditional on seating L now."""
    ks = transitions(sol.grid, sol.grid.prior_state(), s)[L][offer]
    certificate = installed = 0.
    for nxt, prob in ks.items():
        b = current(sol, nxt, sol.s)
        r = play(b, sol.inst, all_shocks=True, fixed_neutral=sol.fixed_neutral)
        weights = offer_probabilities(sol.policy[nxt])[L]
        certificate += prob * (weights @ r.pr[L])
        installed += prob * (weights @ (r.pr[L] * (1 - r.q[L] * r.refuse[L])))
    return certificate, installed


def audit_chain(sol):
    """Independent global revaluation of the complete stationary Markov chain."""
    from scipy.sparse import coo_matrix, eye
    from scipy.sparse.linalg import spsolve
    n = len(sol.grid.states)
    rows = []
    cols = []
    data = []
    rewards = np.zeros((4 * n, 6))
    def add(row, regime, who, kernel, probability):
        if probability == 0:
            return
        for h, p in kernel.items():
            rows.append(row)
            cols.append(regime * 2 * n + 2 * h + who)
            data.append(probability * p)
    for state in range(n):
        b = current(sol, state, sol.s)
        r = play(b, sol.inst, all_shocks=True, fixed_neutral=sol.fixed_neutral)
        kr = transitions(sol.grid, state, sol.s)
        mix = offer_probabilities(sol.policy[state])
        for inc in PARTIES:
            j = 1 - inc
            row = 2 * state + inc
            ar = 2 * n + row
            add(ar, 1, inc, kr[inc][2], 1.)
            rewards[ar] = b.f[inc, 2]
            rewards[row, 2:] = VALENCE
            for pair, weight in enumerate(mix[inc]):
                if not weight:
                    continue
                pr = r.pr[inc, pair]
                loss = pr if inc == L else 1 - pr
                ent = loss * r.q[inc, pair] * r.refuse[inc, pair]
                iw, jw = OFFERS[pair, inc], OFFERS[pair, j]
                add(row, 0, inc, kr[inc][iw], weight * (1 - loss))
                add(row, 0, j, kr[j][jw], weight * (loss - ent))
                add(row, 1, inc, kr[inc][iw], weight * ent)
                rewards[row] += weight * ((1 - loss + ent) * b.f[inc, iw] + (loss - ent) * b.f[j, jw])
                rewards[row, inc] -= weight * loss * r.refuse[inc, pair] * sol.inst.k
    chain = coo_matrix((data, (rows, cols)), shape=(4 * n, 4 * n)).tocsr()
    assert np.max(abs(np.asarray(chain.sum(axis=1)).ravel() - 1)) < 2e-12
    parties = spsolve(eye(4 * n, format='csr') - DISCOUNT * chain, rewards[:, :2])
    target = np.concatenate((sol.values.reshape(-1, 6), sol.auto.reshape(-1, 6)))
    assert np.max(abs(parties - target[:, :2])) < 2e-10
    assert np.max(abs(rewards[:, 2:] - target[:, 2:])) < 2e-10
    dem = chain[:2 * n, :2 * n]
    hazard = np.asarray(chain[:2 * n, 2 * n:].sum(axis=1)).ravel()
    active = hazard > 0
    while True:
        new = active | ((dem @ active.astype(float)) > 0)
        if np.array_equal(new, active):
            break
        active = new
    hit = np.zeros(2 * n)
    if any(active):
        hit[active] = spsolve(eye(sum(active), format='csr') - dem[active, :][:, active], hazard[active])
    assert np.max(abs(hit - sol.risk.ravel())) < 5e-9


def known_technology():
    """Exact rational certificate for the voluntary pure stationary branch
    when a=0, b=1/2. Partition every on- and off-path incentive at its roots.
    """
    from fractions import Fraction as F
    delta = F(1, 4)
    k = F(12, 5)
    p = F(11, 20)
    weights = [F(0), F(2, 5), F(7, 20), F(1, 4)]
    base = [F(-1, 2), F(3, 2), F(-1, 2), F(9, 20), F(101, 200), F(3, 2)]
    gamma = [-2, 2, 1, 1, 1, 1]
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    def profile(s, flat, intercept=F(29, 100)):
        acts = [3, 1 if flat else 3]
        f = np.array([[[-(base[j] - di)**2 + (2 * gi * w - w * w) * s / 4 for di, gi in zip(base, gamma)]
                    for w in [(2 * j - 1) * n for n in range(3)]] for j in (0, 1)], dtype=object)
        term = np.array([(1 - p) * f[L, pairs[u][L]] + p * f[R, pairs[u][R]] for u in acts])
        val = term.copy()
        val[:, :2] += delta / (1 - delta) * ((1 - p) * term[L, :2] + p * term[R, :2])
        val[:, 2:] += 45
        aut = f[:, 2].copy()
        aut[:, :2] /= (1 - delta)
        a = f + delta * aut[:, None, :]
        d = f + delta * val[:, None, :]
        tests = [[], [], [], []]
        valid = True
        for inc in (0, 1):
            j = 1 - inc
            pay = []
            on = acts[inc]
            for pair, (ell, r) in enumerate(pairs):
                iw, jw = pairs[pair][inc], pairs[pair][j]
                old = f[inc, iw] - f[j, jw]
                gain = a[inc, iw] - d[j, jw]
                tests[0].extend(old[2:])
                tests[0].extend(gain[2:])
                share = sum(w * int(v > 0) for w, v in zip(weights, old[2:]))
                share += sum(w * int(v > 0) for w, v in zip(weights, gain[2:]))
                q = intercept + F(3, 10) * share
                t = q * gain[inc] - k
                tests[1].append(t)
                refuse = int(t > 0)
                tau = q * refuse
                om = (2 * inc - 1) * (1 - tau) * old[2:]
                ym = (2 * inc - 1) * (d[inc, iw] - d[j, jw] - tau * gain)[2:]
                pr = F(0)
                for eta, prob in [(-100, F(9, 20)), (0, F(1, 10)), (100, F(9, 20))]:
                    votes = F(0)
                    for age in (om, ym):
                        tests[2].extend(age + eta)
                        votes += sum(w * int(x + eta > 0) for x, w in zip(age, weights)) / 2
                    if votes > F(1, 2):
                        pr += prob
                if pair == on:
                    valid &= not refuse and pr == p
                lose = pr if inc == L else 1 - pr
                u = (1 - lose) * d[inc, iw, :2] + lose * (d[j, jw, :2] + tau * gain[:2])
                u[inc] -= lose * refuse * k
                pay.append(u)
            for who in (0, 1):
                margin = pay[on][who] - pay[on ^ (2 - who)][who]
                tests[3].append(margin)
                valid &= margin >= 0
        return tests, valid

    # Under the flat-challenger/mutual-concession profile R's continuation
    # if L is incumbent is -12/5-(31/320)S, and autocracy pays S/(1-delta).
    slope = F(3, 4) + delta * (1 / (1 - delta) + F(31, 320))
    permanent = (k / F(47, 100) - F(23, 5)) / slope
    for left, right, flat in [(F(0), F(1, 5), False), (F(1, 5), permanent, True)]:
        knots = [left, right]
        for stage in range(4):
            added = set(knots)
            for u, v in zip(knots, knots[1:]):
                x = (2 * u + v) / 3
                y = (u + 2 * v) / 3
                f0 = profile(x, flat)[0][stage]
                f1 = profile(y, flat)[0][stage]
                for a, b in zip(f0, f1):
                    if b == a:
                        continue
                    root = x - a * (y - x) / (b - a)
                    if u < root < v:
                        added.add(root)
            knots = sorted(added)
        checks = knots + [(u + v) / 2 for u, v in zip(knots, knots[1:])]
        assert all(profile(x, flat)[1] for x in checks if not(flat and x == F(1, 5)))
    assert not profile(permanent + F(1, 1000000), True)[1]
    shock = (k / F(47, 100) - F(23, 5) - F(23, 60) * F(1, 20)) / F(3, 4)
    paid = shock + F(1, 100) / (F(9, 20) * F(47, 100) * F(3, 4))
    assert profile(F(1, 20), False, F(13, 40))[1]
    flat_low = F(101, 200)
    flat_stakes0 = F(5543, 1200)
    voluntary = (k / flat_low - flat_stakes0) / F(3, 4)
    costly = voluntary + F(1, 100) / (F(9, 20) * flat_low * F(3, 4))
    assert voluntary < F(1, 5) < costly < F(4, 5)
    return permanent, shock, paid, voluntary, F(1, 5), costly


def draw_figure(lines):
    """Show the same flat first-term mandate at two stationary continuations."""
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11.5, 'pdf.fonttype':42, 'font.family':'DejaVu Sans',
                         'axes.spines.right':False, 'axes.spines.top':False})
    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    fig.subplots_adjust(left=.155, right=.97, bottom=.24, top=.90)
    ax.axhline(0, color='#555555', lw=1)
    ax.axvline(.60, color='#808588', lw=1.1, ls=':')
    for (s, pts, cut), color, style in zip(lines, ('#204e6c', '#a34e22'), ('-', '--')):
        ax.plot([.4, .75], pts, color=color, ls=style, lw=2.3,
                label=f'Future elections: $S^\\infty={s:.2f}$')
        ax.scatter([cut], [0], s=30, color=color, zorder=4)
        ax.annotate(f'{cut:.3f}', (cut, 0), xytext=((9, -18) if s == .05 else (-35, 9)),
                    textcoords='offset points', color=color, fontsize=10.5)
        ax.scatter([.60], [np.interp(.6, [.4, .75], pts)], color=color, s=30, zorder=4)
    ax.legend(loc='upper left', frameon=False, fontsize=10.5)
    ax.set_xlim(.395, .755)
    ax.set_xticks([.4, .45, .5, .55, .60, .65, .7, .75])
    ax.set_xlabel('Forecast variance in the current term, $S$')
    ax.set_ylabel('Incumbent gain from refusal')
    ax.text(.748, .080, 'Refuses', ha='right', color='#404648', fontsize=10)
    ax.text(.748, -.067, 'Concedes', ha='right', color='#404648', fontsize=10)
    ax.set_ylim(-.082, .092)
    ax.grid(axis='y', color='#e2e6e8', lw=.55)
    ax.set_axisbelow(True)
    fig.savefig(Path(__file__).with_name('citizen_stationary_transfer.pdf'),
                metadata={'Creator':'verify_citizen_mpe.py', 'CreationDate':None, 'ModDate':None})
    plt.close(fig)


def compare_future_ballots(grid):
    """Figure 5 and the fixed-neutral electoral counterfactual in Section 6.5."""
    inst = Institutions(intercept=.29, k=2.4)
    root = grid.prior_state()
    # OFFERS stores (ell,r) lexicographically. Price the current pair (0,1).
    flat = 1
    stakes = []
    ballot_values = []
    cases = ((.05, False, (.60, .62)), (.14, False, (.60,)), (.05, True, (.62,)))
    for future_s, fixed, current_levels in cases:
        sol = solve(grid, future_s, inst, fixed_neutral=fixed)
        audit_chain(sol)
        cuts, outcomes = cutoff_table(sol)  # Today's ballot is ordinary in all cases.
        print(f'Future S={future_s:.2f}, neutral ballot={"fixed to R" if fixed else "responsive"}')
        for s in current_levels:
            b = current(sol, root, s)
            p, chosen, _, _ = outcomes(s)
            np.testing.assert_allclose(chosen['full'], [0, 1], atol=TIE)
            np.testing.assert_allclose([chosen['loss'], p.q[R, flat]], [.45, .47], atol=TIE)
            a, c, gain = b.a[R, 1, R], b.d[L, 0, R], p.incentives[R, flat]
            certified, installed = next_return(sol, s)
            print(f'  S={s:.2f}, A_R={a:.6f}, C_R={c:.6f}, T_R={gain:+.6f}, '
                  f'L installed={chosen["installed"]:.4f}, S+={cuts[0]:.6f}; '
                  f'next R certified={certified:.6f}, installed={installed:.6f}')
            if s == .60:
                stakes.append((a, c))
                deviation = play(current(sol, root, s, left_flat=.01), inst, all_shocks=True)
                extra = deviation.value[R, flat, L] - p.value[R, flat, L]
                print(f'  ell=.01: L campaign gain={extra:+.6f}, T_R={deviation.incentives[R, flat]:+.6f}')
            else:
                ballot_values.append((a, c, gain))
    ordinary, fixed = ballot_values
    np.testing.assert_allclose(ordinary[0], fixed[0], rtol=0., atol=TIE)
    np.testing.assert_allclose(ordinary[2] - fixed[2], .47 * (fixed[1] - ordinary[1]), atol=TIE)
    delta_a, delta_c = np.subtract(stakes[1], stakes[0])
    print(f'Figure 5: delta A_R={delta_a:+.6f}, delta C_R={delta_c:+.6f}, '
          f'autocracy share={delta_a / (delta_a - delta_c):.1%}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refinement', action='store_true',
                        help='solve the larger mean grid reported in the appendix')
    parser.add_argument('--figure', action='store_true', help='regenerate the stationary transfer figure')
    parser.add_argument('--future-ballot', action='store_true',
                        help='compare future electoral rules and decompose the Figure 5 change')
    args = parser.parse_args()
    if args.future_ballot and args.figure:
        parser.error('--future-ballot reports the comparisons without redrawing the figure')
    lines = []
    exact = known_technology()
    print('Known technology: permanent branch, then flat/repeal cutoffs for two institutions')
    print(' '.join(f'{x} ({float(x):.6f})' for x in exact))
    fixed = GaussianGrid([0.], [.5], [np.inf])
    for q, targets in [(.29, (exact[1], exact[1], exact[2])), (.325, exact[3:])]:
        sol = solve(fixed, .05, Institutions(intercept=q))
        cuts, _ = cutoff_table(sol)
        np.testing.assert_allclose(cuts, [float(x) for x in targets], atol=5e-10, rtol=0.)
        audit_chain(sol)
    if args.refinement:
        grid = GaussianGrid(np.linspace(-.08, .08, 9), np.linspace(-.5, 1.5, 9),
                          [0., 4., 16., 64., np.inf], 21)
        cases = [(.29, 2.4, .05), (.29, 2.4, .14)]
    else:
        grid = GaussianGrid(np.linspace(-.04, .04, 5), np.linspace(0., 1., 5),
                          [0., 4., 16., 64., np.inf], 9)
        cases = [(.29, 2.4, .05), (.29, 2.4, .14), (.29, 2.55, .05), (.35, 2.4, .05)]
    if args.future_ballot:
        compare_future_ballots(grid)
        return
    for q, k, s in cases:
        inst = Institutions(k=k, intercept=q)
        sol = solve(grid, s, inst)
        audit_chain(sol)
        stationary = play(current(sol, grid.prior_state(), s), inst, all_shocks=True)
        for inc in PARTIES:
            projected = campaign(stationary, inc)['full']
            np.testing.assert_allclose(projected, sol.policy[grid.prior_state(), inc], atol=TIE)
        cuts, get_outcome = cutoff_table(sol)
        unprotected, chosen, protected, gross = get_outcome(.60)
        returns = next_return(sol, .60)
        def form(x):
            return 'none' if x is None else f'{x:.6f}'
        print(f'q0={q:.2f}, k={k:.2f}, future S={s:.2f}: '
              f'S+, free, paid = {", ".join(map(form, cuts))}; '
              f'eventual takeover={sol.risk[grid.prior_state(), R]:.6f}', flush=True)
        print(f'  current .60: full probabilities={chosen["full"]}, '
              f'T(flat)={unprotected.incentives[R, 1]:+.6f}, '
              f'installed={chosen["installed"]:.6f}, gross repeal gain={gross:.6f}')
        print(f'  after seating flat L: R next certified={returns[0]:.6f}, '
              f'installed={returns[1]:.6f}; '
              f'mixed at {sum(sol.multiplicity == 0)} information states; '
              f'max error={sol.residual:.2e}', flush=True)
        if q == .29 and k == 2.4:
            pts = [get_outcome(t)[0].incentives[R, 1] for t in (.4, .75)]
            lines.append((s, pts, cuts[0]))
            assert cuts[0] == cuts[1] and cuts[0] < cuts[2] < .8
            assert (unprotected.incentives[R, 1] > 0) == (s == .14)
            np.testing.assert_allclose(chosen['full'], [0., 1.], atol=TIE)
            np.testing.assert_allclose(unprotected.q[R, 1], .47, atol=TIE)
        if q == .29 and k == 2.55:
            np.testing.assert_allclose(cuts, [.8, .8, .8], atol=1e-10)
        if q == .35:
            assert cuts == [None, 0., 0.] and get_outcome(0.)[-1] > .01
    if args.figure:
        if args.refinement:
            parser.error('--figure uses the baseline integration grid')
        draw_figure(lines)


if __name__ == '__main__':
    main()
