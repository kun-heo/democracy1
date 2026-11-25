"""Reproduce the two-election Bayesian mandate calculation with exact rational arithmetic.

Run with Python 3.10+; only the standard library is used. Both electoral regimes
share the same priors, policies, information and payoff functions. The control
assigns the final zero-valence certificate to R independently of the record.
"""
from collections import defaultdict
from fractions import Fraction as F
from functools import cache
from itertools import product

LEFT, RIGHT = 0, 1
BASE = (F(-1, 2), F(3, 2))
SIGN = (-1, 1)
VOTER_BASE = F(101, 200)
# Parties, enforcement blocs, young voter, old voter.
TYPES = ((BASE[0], -2), (BASE[1], 2), (BASE[0], 0), (BASE[1], 0),
         (VOTER_BASE, 1), (VOTER_BASE, 1))
FINAL_MONITORING, DISCOUNT = F(1, 20), F(1, 4)
ATTEMPT_COST, SUCCESS = F(12, 5), F(1, 2)
BLOC_COST, ERROR_RADIUS = F(1, 2), F(1, 80)
VALENCE = (-100, 0, 100)
VALENCE_MASS = (F(9, 20), F(1, 10), F(9, 20))
WORLDS = tuple(product((-F(1, 50), F(1, 50)), (0, 1)))
PRIOR = tuple(range(4))
OFFERS = tuple(product((0, 1), repeat=2))
ZERO, ONE = F(0), F(1)


@cache
def means(record):
    return tuple(sum(F(WORLDS[i][k]) for i in record)/len(record) for k in (0, 1))


@cache
def posteriors(policy, state):
    """Integrate the uniform measurement error over intervals of constant likelihood."""
    centers = [policy*(a+b*state) for a, b in WORLDS]
    edges = sorted({c+sign*ERROR_RADIUS for c in centers for sign in (-1, 1)})
    result = defaultdict(F)
    for start, end in zip(edges, edges[1:]):
        mid = (start+end)/2
        record = tuple(i for i, c in enumerate(centers) if abs(mid-c) < ERROR_RADIUS)
        if record:
            result[record] += (end-start)*len(record)/(len(WORLDS)*2*ERROR_RADIUS)
    assert sum(result.values()) == 1
    return dict(result)


def flow(government, weight, record, monitoring):
    """Current payoff, up to a policy-independent term for each player and true state."""
    a, b = means(record)
    policy_mean = BASE[government]+weight*a
    return tuple(-(policy_mean-d)**2-weight**2*b**2*monitoring
                 +2*gamma*(policy_mean*a+weight*b**2*monitoring)
                 for d, gamma in TYPES)


def interpolate(low, high, weight):
    return tuple(x+weight*(y-x) for x, y in zip(low, high))


def certificates(left, right):
    # Equal sized cohorts, sincere ballots. Tied ballots and certificates go to L.
    return tuple(all(right[i]-left[i]+v > 0 for i in (4, 5)) for v in VALENCE)


def election_payoffs(left, right, outcomes):
    right_prob = sum(p for p, elected in zip(VALENCE_MASS, outcomes) if elected)
    valence = sum(p*v for p, v, elected in zip(VALENCE_MASS, VALENCE, outcomes) if elected)
    payoffs = interpolate(left, right, right_prob)
    return tuple(x+valence if i >= 4 else x for i, x in enumerate(payoffs)), right_prob


def verify_quorum(entrenched, lawful, incumbent):
    challenger = 1-incumbent
    # The challenger bloc alone is willing. Only the one-endorsement quorum can form.
    assert lawful[2+challenger]-entrenched[2+challenger] > BLOC_COST
    assert entrenched[2+incumbent]-lawful[2+incumbent] > BLOC_COST


def ordered(incumbent, own, rival):
    return (own, rival) if incumbent == LEFT else (rival, own)


def zero_of_affine(at_zero, at_one):
    return None if at_one == at_zero else -at_zero/(at_one-at_zero)


def insert_root(roots, root, left=ZERO, right=ONE):
    if root is not None and left < root < right:
        roots.add(root)


def intervals(points):
    ordered_points = sorted(points)
    return list(zip(ordered_points, ordered_points[1:]))


class TwoElections:
    def __init__(self, fixed_final_vote=False):
        self.fixed_final_vote = fixed_final_vote

    @cache
    def final(self, record):
        values, wins = {}, {}
        for pair in OFFERS:
            left, right = (flow(j, SIGN[j]*pair[j], record, FINAL_MONITORING) for j in (LEFT, RIGHT))
            for incumbent, (own, rival) in enumerate(((left, right), (right, left))):
                verify_quorum(own, rival, incumbent)
                assert SUCCESS*(own[incumbent]-rival[incumbent]) < ATTEMPT_COST
            outcomes = (False, True, True) if self.fixed_final_vote else certificates(left, right)
            values[pair], wins[pair] = election_payoffs(left, right, outcomes)
        a, b = means(record)
        if a or b:
            # Elimination by strict best responses covers mixed campaigns.
            assert all(values[l, 1][RIGHT] > values[l, 0][RIGHT] for l in (0, 1))
            assert values[1, 1][LEFT] > values[0, 1][LEFT]
            pair = (1, 1)
        else:
            assert all(value == values[0, 0] for value in values.values())
            pair = (0, 0)
        return values[pair], wins[pair]

    @cache
    def branch_endpoint(self, government, responsive, entrenched, report):
        """Branch conditional on no state report (0) or a revealing state report (1)."""
        weight = SIGN[government]*responsive
        current = flow(government, weight, PRIOR, report)
        future = [ZERO]*len(TYPES)
        a, b = means(PRIOR)
        for state in (-1, 1):
            policy = BASE[government]+weight*(a+b*report*state)
            for record, probability in posteriors(policy, state).items():
                if entrenched:
                    following = flow(government, 2*SIGN[government], record, FINAL_MONITORING)
                else:
                    following, _ = self.final(record)
                for i in range(4):
                    future[i] += probability*following[i]/2
                future[4] += probability*following[5]/2  # Young becomes old. Old exits.
        return tuple(v+DISCOUNT*y for v, y in zip(current, future))

    def branches(self, incumbent, pair, s):
        return tuple(interpolate(self.branch_endpoint(j, pair[j], autocratic, 0),
                                 self.branch_endpoint(j, pair[j], autocratic, 1), s)
                     for j, autocratic in ((incumbent, False), (1-incumbent, False), (incumbent, True)))

    def gain(self, incumbent, pair, s):
        _, lawful, entrenched = self.branches(incumbent, pair, s)
        return SUCCESS*(entrenched[incumbent]-lawful[incumbent])-ATTEMPT_COST

    def certificate_values(self, incumbent, pair, s, refuse):
        victory, lawful, entrenched = self.branches(incumbent, pair, s)
        loss = list(lawful)
        if refuse:
            loss = list(interpolate(lawful, entrenched, SUCCESS))
            loss[incumbent] -= ATTEMPT_COST
            loss[2+1-incumbent] -= (1-SUCCESS)*BLOC_COST
        return (victory, loss) if incumbent == LEFT else (loss, victory)

    def campaign_value(self, incumbent, pair, s, refuse=None, outcomes=None):
        if refuse is None:
            refuse = self.gain(incumbent, pair, s) > 0
        left, right = self.certificate_values(incumbent, pair, s, refuse)
        if outcomes is None:
            outcomes = certificates(left, right)
        return election_payoffs(left, right, outcomes)[0]

    def verify(self, incumbent):
        challenger = 1-incumbent
        cuts = {ZERO, ONE}
        for pair in OFFERS:
            for s in (ZERO, ONE):
                _, lawful, entrenched = self.branches(incumbent, pair, s)
                verify_quorum(entrenched, lawful, incumbent)
            insert_root(cuts, zero_of_affine(self.gain(incumbent, pair, ZERO),
                                             self.gain(incumbent, pair, ONE)))

        # Conditional on a refusal profile, all ballot comparisons are affine in s.
        for start, end in intervals(cuts):
            for pair in OFFERS:
                refuse = self.gain(incumbent, pair, (start+end)/2) > 0
                l0, r0 = self.certificate_values(incumbent, pair, ZERO, refuse)
                l1, r1 = self.certificate_values(incumbent, pair, ONE, refuse)
                for age in (4, 5):
                    for v in VALENCE:
                        insert_root(cuts, zero_of_affine(r0[age]-l0[age]+v, r1[age]-l1[age]+v), start, end)

        # Conditional on refusals and certificates, each campaign deviation is affine.
        for start, end in intervals(cuts):
            mid = (start+end)/2
            v0, v1 = {}, {}
            for pair in OFFERS:
                refuse = self.gain(incumbent, pair, mid) > 0
                outcomes = certificates(*self.certificate_values(incumbent, pair, mid, refuse))
                v0[pair] = self.campaign_value(incumbent, pair, ZERO, refuse, outcomes)
                v1[pair] = self.campaign_value(incumbent, pair, ONE, refuse, outcomes)
            for party in (LEFT, RIGHT):
                for other in (0, 1):
                    responsive, flat = ordered(party, 1, other), ordered(party, 0, other)
                    insert_root(cuts, zero_of_affine(v0[responsive][party]-v0[flat][party],
                                                     v1[responsive][party]-v1[flat][party]), start, end)

        lower, upper = (zero_of_affine(self.gain(incumbent, ordered(incumbent, 1, rival), ZERO),
                                       self.gain(incumbent, ordered(incumbent, 1, rival), ONE))
                        for rival in (1, 0))
        assert 0 < lower < upper < 1
        # At every root, apply the original tie rules. On each open cell every comparison's sign is constant.
        for s in cuts | {(l+r)/2 for l, r in intervals(cuts)}:
            values = {pair: self.campaign_value(incumbent, pair, s) for pair in OFFERS}
            if s <= lower:
                assert all(self.gain(incumbent, pair, s) <= 0 for pair in OFFERS)
                continue
            for rival in (0, 1):
                assert values[ordered(incumbent, 1, rival)][incumbent] > values[ordered(incumbent, 0, rival)][incumbent]
            both, accommodating = ordered(incumbent, 1, 1), ordered(incumbent, 1, 0)
            assert values[both][challenger] != values[accommodating][challenger]
            assert (values[both][challenger] > values[accommodating][challenger]) == (s > upper)
            assert self.gain(incumbent, both, s) > 0
            assert (self.gain(incumbent, accommodating, s) > 0) == (s > upper)
        return lower, upper, cuts


def main():
    for fixed in (False, True):
        model = TwoElections(fixed)
        print("Final votes fixed" if fixed else "Final votes respond to public evidence")
        for incumbent in (LEFT, RIGHT):
            lower, upper, cuts = model.verify(incumbent)
            print(f"  {'LR'[incumbent]} incumbent: {lower} ({float(lower):.6f}), "
                  f"{upper} ({float(upper):.6f}); {len(cuts)-1} open cells plus all boundaries.")
        flat_vote = sum(mass*model.final(record)[1]/2 for state in (-1, 1)
                        for record, mass in posteriors(BASE[LEFT], state).items())
        gain = model.gain(RIGHT, (0, 1), F(41, 100))
        assert flat_vote == (F(11, 20) if fixed else F(53, 100))
        assert gain == (F(-6651, 1000000) if fixed else F(2757, 1000000))
        print(f"  R return after conceding to flat L: {flat_vote}; R refusal gain at S=.41: {gain}.")


if __name__ == "__main__":
    main()
