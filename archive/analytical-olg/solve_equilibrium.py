"""A finite party game with mandates, public learning and constitutional enforcement.

Run with Python 3.11+, NumPy and SymPy. An initial floating-point search proposes
a full contingent plan. Independent rational evaluation checks all deviations.
"""
from fractions import Fraction
from itertools import product
from typing import NamedTuple

import numpy as np
import sympy as sp
from sympy.polys.matrices import DomainMatrix


PARTIES = (0, 1)
ENVS = (0, 1)
RECORDS = (0, 1, 2)  # U, K0, K1
WORLDS = (0, 1)
SIGNALS = (-1, 0, 1)
AUT = -1  # continuing, uncampaigned autocracy
YOUNG = slice(4, 8)
OLD = slice(8, 12)
NAMES = ("L", "R", "left bloc", "right bloc",
         "young 1", "young 2", "young 3", "young 4", "old 1", "old 2", "old 3", "old 4")
TYPES = (
    ("-.45", "-.5", ".15"), ("1", "2.1", ".15"),
    ("-.2", "-.5", ".43"), ("1", "1.8", "1.08"),
    ("0", ".15", ".65"), ("0", "1.9", ".65"),
    (".1", "2.5", ".65"), ("1", "2", ".65"),
)
MANDATES = ((("-.1", "-.1"), (".3", "1.2")), (("1", "2.4"), (".5", ".7")))
VOTER_RANKING = (0, 1, 0, 1)
RANKING = (0, 1) + VOTER_RANKING * 2  # blocs, young voters, old voters
TYPE_MASSES = (15, 30, 20, 35)
MASSES = TYPE_MASSES * 2  # equal-sized cohorts, each with the same composition
VALENCE = ("-4", "-3", "-1.5", "-.6", "-.2", "0", ".2", ".6", "1.5", "3", "4")
WEIGHTS = (".10", ".01", ".01", ".10", ".10", ".36", ".10", ".10", ".01", ".01", ".10")
DEFAULTS = dict(delta=".25", publication=".25", attempt=".57", sanction=".30",
                defense=".12", two=".64", rent=".15", low=".12", high=".88",
                up=".12", down=".20")


class Election(NamedTuple):
    refuse: bool
    defenders: tuple
    certificates: tuple
    ballots: tuple


class Detail(NamedTuple):
    q: object
    gain: object
    gaps: tuple
    right: object
    willing: tuple


def available(record):
    """List the mandate/account pairs admissible after the public record."""
    return tuple(k for k in range(4) if record == 0 or k % 2 == record - 1)


class Game:
    """Values are indexed by regime, party in office, environment, public record,
    true technology, and actor, in that order. At the unresolved record the
    actor uses its own model when it compares the two world-specific values.
    """
    def __init__(self, exact=False, params=None, types=TYPES, mandates=MANDATES,
                 inactive_quorums=()):
        self.inactive_quorums = frozenset(inactive_quorums)
        self.exact = exact
        cast = self.cast = Fraction if exact else float
        self.dtype = object if exact else float
        self.tol = 0 if exact else 1e-9
        for k, v in (DEFAULTS | (params or {})).items():
            setattr(self, k, cast(v))
        actor_types = tuple(types) + tuple(types[4:])
        self.beta, self.gamma, self.chi = np.array(
            [[cast(x) for x in row] for row in actor_types], dtype=self.dtype).T
        self.mandates = tuple(tuple(tuple(cast(x) for x in m) for m in p) for p in mandates)
        self.xs = np.array([cast(i)/2 for i in range(-4, 7)], dtype=self.dtype)
        self.nonzero = self.xs != 0
        self.eta = tuple(map(cast, VALENCE))
        self.weights = tuple(map(cast, WEIGHTS))
        self.ep = ((1-self.up, self.up), (self.down, 1-self.down))
        self.qp = (1-self.two, self.two)
        self.sprob = tuple((p/2, 1-p, p/2) for p in (self.low, self.high))
        # Flow to each actual actor, conditional on the true technology. The
        # mandate's implementation score is separate from these material payoffs.
        self.flow = {}
        for d, party, m in product((0, 1), PARTIES, SIGNALS):
            rent = np.array([self.rent if i == party else cast(0) for i in range(len(NAMES))])
            x = self.xs[:, None]
            self.flow[d, party, m] = np.stack([
                -(x-self.beta)**2 + 2*self.gamma*a*x*m + rent - d*self.chi for a in WORLDS
            ], axis=1)

    def zeros(self):
        return np.full((2, 2, 2, 3, 2, len(NAMES)), self.cast(0), dtype=self.dtype)

    @staticmethod
    def expectation(array, record):
        return array[record-1] if record else (array[0]+array[1])/2

    def actions(self, values, d, party, e, rec):
        """Compute all within-term choices, including those after rejected platforms."""
        future = sum(p*values[d, party, en] for en, p in enumerate(self.ep[e]))
        future[..., YOUNG] = future[..., OLD].copy()
        future[..., OLD] = self.cast(0)
        stay = future[rec]
        informative = np.array([future[a+1, a] for a in WORLDS])
        increments = self.delta*self.publication*(informative-stay) if rec == 0 else 0
        continuation = self.delta*stay + self.nonzero[:, None, None]*increments
        offers = available(rec) + ((AUT,) if d else ())
        return_values, rules, details = {}, {}, {}
        for package in offers:
            b, g = (self.beta[party], self.gamma[party]) if package == AUT else self.mandates[party][package//2]
            if package == AUT:
                a = rec-1 if rec else self.cast(1)/2
                vnext = self.expectation(continuation.transpose(1, 0, 2), rec)[:, party]
            else:
                a = package % 2
                vnext = continuation[:, a, party]
            choices, payoffs, margins = [], [], []
            for m in SIGNALS:
                score = -(self.xs-b)**2 + 2*g*a*self.xs*m + vnext
                best = max(score)
                ix = next(i for i, q in enumerate(score) if q >= best-self.tol)
                choices.append(ix)
                margins.append(score[ix]-max(q for i, q in enumerate(score) if i != ix))
                payoffs.append(self.flow[d, party, m][ix]+continuation[ix])
            return_values[package] = sum(p*v for p, v in zip(self.sprob[e], payoffs))
            rules[package] = tuple(choices)
            details[package] = tuple(margins)
        return return_values, rules, details

    def contest(self, inc, environment, rec, offers, victory, concession, entrenchment):
        """Solve pivotal enforcement, party refusal and each constituency's ballot."""
        if rec:
            views = (rec-1,)*len(RANKING)
        elif offers[0]%2 == offers[1]%2:
            views = (offers[0]%2,)*len(RANKING)
        else:
            views = RANKING
        willing = tuple(concession[a, i+2]-entrenchment[a, i+2] >= self.defense-self.tol
                        for i, a in enumerate(views[:2]))
        ordered = tuple(i for i in (1-inc, inc) if willing[i])
        defenders = tuple(ordered[:q] if len(ordered) >= q else () for q in (1, 2))
        if (inc, environment, rec, *offers) in self.inactive_quorums:
            defenders = (defenders[0], ())
        if self.exact:
            # Each bloc's own participation must survive a unilateral switch.
            for quorum, participating in enumerate(defenders, start=1):
                for bloc, account in enumerate(views[:2]):
                    before = len(participating) >= quorum
                    after = (len(participating) + (1 if bloc not in participating else -1)) >= quorum
                    benefit = (int(before)-int(after))*(concession[account,2+bloc]-entrenchment[account,2+bloc])
                    benefit += self.defense*(1 if bloc not in participating else -1)
                    assert benefit >= 0, "Profitable enforcement deviation."
        success = sum(p for p, group in zip(self.qp, defenders) if not group)
        gain = success*self.expectation(entrenchment-concession, rec)[inc] - self.attempt-(1-success)*self.sanction
        refuse = bool(gain > self.tol)
        after = concession
        if refuse:
            after = concession + success*(entrenchment-concession)
            after[:, inc] -= self.attempt + (1-success)*self.sanction
            for group, p in zip(defenders, self.qp):
                for bloc in group:
                    after[:, 2+bloc] -= p*self.defense
        left, right = (victory, after) if inc == 0 else (after, victory)
        gaps = tuple(right[a,4+i]-left[a,4+i] for i,a in enumerate(views[2:]))
        ballots = tuple(tuple(bool(v+eta > self.tol) for v in gaps) for eta in self.eta)
        certs = tuple(2 * sum(n for n, vote in zip(MASSES, row) if vote) > sum(MASSES)
                      for row in ballots)
        win = sum(p for p,c in zip(self.weights, certs) if c)
        payoff = (1-win)*left + win*right
        payoff[:, 4:] += sum(p*eta for p,eta,c in zip(self.weights, self.eta, certs) if c)
        return payoff, Election(refuse, defenders, certs, ballots), Detail(success, gain, gaps, win, willing)

    def step(self, values, selection=None, proposed=None):
        """Solve the within-term game given a continuation value for each actor."""
        update = self.zeros()
        plan, detail, installations = {}, {}, {}
        for state in product((0, 1), PARTIES, ENVS, RECORDS):
            package_values, rules, margins = self.actions(values, *state)
            for package, value in package_values.items():
                key = (*state, package)
                installations[key] = value
                plan["policy", *key] = rules[package]
                detail["policy", *key] = margins[package]
            if state[0] == 1:
                update[state] = package_values[AUT]

        for incumbent, env, record in product(PARTIES, ENVS, RECORDS):
            challenger = 1 - incumbent
            options = available(record)
            payoffs = {}
            for offers in product(options, repeat=2):
                victory = installations[0, incumbent, env, record, offers[incumbent]]
                concession = installations[0, challenger, env, record, offers[challenger]]
                entrenchment = installations[1, incumbent, env, record, offers[incumbent]]
                value, election, stats = self.contest(
                    incumbent, env, record, offers, victory, concession, entrenchment
                )
                key = ("election", incumbent, env, record, *offers)
                plan[key], detail[key] = election, stats
                payoffs[offers] = value

            # Parties evaluate an entire program deviation under their public posterior.
            scores = {pair: self.expectation(value, record)[:2] for pair, value in payoffs.items()}
            margins = {}
            for left, right in payoffs:
                best_left = max(scores[other, right][0] for other in options if other != left)
                best_right = max(scores[left, other][1] for other in options if other != right)
                left_value, right_value = scores[left, right]
                margins[left, right] = (left_value - best_left, right_value - best_right)

            key = ("programs", incumbent, env, record)
            pure = [pair for pair, gain in margins.items() if min(gain) >= -self.tol]
            if proposed is not None:
                chosen = proposed[key]
                if chosen not in pure:
                    raise AssertionError(("Profitable campaign deviation", key, chosen))
            elif not pure:
                if self.exact:
                    raise ArithmeticError(("No pure program equilibrium", key))
                # A temporary iterate may fail to admit a pure campaign equilibrium.
                chosen = max(payoffs, key=lambda pair: min(margins[pair]))
            elif selection is not None:
                chosen = min(pure, key=selection[incumbent, env, record].index)
            else:
                preferred = [pair for pair in pure if (pair[0] % 2, pair[1] % 2) == (0, 1)]
                chosen = min(preferred if record == 0 and preferred else pure)
            plan[key] = chosen
            detail[key] = margins[chosen]
            update[0, incumbent, env, record] = payoffs[chosen]
        return update, plan, detail

    def propose(self, initial=None, damping=.7, iterations=260, selection=None):
        """Search for a complete candidate profile. Exact verification follows separately."""
        values = self.zeros() if initial is None else initial.copy()
        previous = None
        repetitions = 0
        for _ in range(iterations):
            following, plan, detail = self.step(values, selection=selection)
            if plan == previous:
                repetitions += 1
                campaign_margins = [min(v) for k, v in detail.items() if k[0] == "programs"]
                if repetitions >= 5 and min(campaign_margins) > -1e-7:
                    return plan
            else:
                repetitions = 0
            previous = plan
            values = (1 - damping) * values + damping * following
        raise ArithmeticError("The strategy search did not settle.")


def event_branches(game, state, plan, reward):
    """Enumerate government installations and their political rewards independently."""
    regime, incumbent, env, record = state
    if regime == 1:
        return {(1, incumbent, AUT): game.cast(1)}
    challenger = 1 - incumbent
    offers = plan["programs", incumbent, env, record]
    election = plan["election", incumbent, env, record, *offers]
    branches = {}

    def add(regime, governing_party, probability):
        branch = (regime, governing_party, offers[governing_party])
        branches[branch] = branches.get(branch, game.cast(0)) + probability

    for winner, probability, valence in zip(election.certificates, game.weights, game.eta):
        if winner:
            reward[4:] += probability * valence
        if int(winner) == incumbent:
            add(0, incumbent, probability)
        elif not election.refuse:
            add(0, challenger, probability)
        else:
            reward[incumbent] -= probability * game.attempt
            for defenders, quorum_probability in zip(election.defenders, game.qp):
                mass = probability * quorum_probability
                if defenders:
                    reward[incumbent] -= mass * game.sanction
                    for bloc in defenders:
                        reward[2 + bloc] -= mass * game.defense
                    add(0, challenger, mass)
                else:
                    add(1, incumbent, mass)
    return branches


def exact_values(game, plan):
    """Solve each true world's sixteen feasible public states in rational arithmetic."""
    values, chains = game.zeros(), []
    for world in WORLDS:
        states = list(product((0, 1), PARTIES, ENVS, (0, world + 1)))
        index = {state: row for row, state in enumerate(states)}
        matrix = sp.zeros(len(states))
        rewards = sp.zeros(len(states), len(NAMES))
        for row, state in enumerate(states):
            _, _, env, record = state
            reward = np.full(len(NAMES), Fraction(0), dtype=object)
            branches = event_branches(game, state, plan, reward)
            for (regime, party, program), probability in branches.items():
                actions = plan["policy", regime, party, env, record, program]
                for report, mass, action in zip(SIGNALS, game.sprob[env], actions):
                    weight = probability * mass
                    reward += weight * game.flow[regime, party, report][action, world]
                    informative = record == 0 and game.xs[action] != 0
                    release = game.publication if informative else Fraction(0)
                    for next_env, env_mass in enumerate(game.ep[env]):
                        same = index[regime, party, next_env, record]
                        learned = index[regime, party, next_env, world + 1]
                        matrix[row, same] += weight * env_mass * (1 - release)
                        matrix[row, learned] += weight * env_mass * release
            for actor, payoff in enumerate(reward):
                rewards[row, actor] = payoff
        assert all(sum(matrix.row(row)) == 1 for row in range(len(states)))
        system = sp.eye(len(states)) - sp.Rational(game.delta) * matrix
        lhs = DomainMatrix.from_Matrix(system).to_field()
        rhs = DomainMatrix.from_Matrix(rewards[:, :4]).convert_to(lhs.domain)
        organizations = lhs.lu_solve(rhs).to_Matrix()
        old = rewards[:, OLD]
        assert rewards[:, YOUNG] == old
        young = rewards[:, YOUNG] + sp.Rational(game.delta) * matrix * old
        solution = organizations.row_join(young).row_join(old)
        for row, state in enumerate(states):
            values[(*state, world)] = [Fraction(int(v.p), int(v.q)) for v in solution.row(row)]
        chains.append((states, matrix))
    return values, chains


def certify(params=None, proposal=None, types=TYPES, mandates=MANDATES, inactive_quorums=()):
    """Check a proposed equilibrium at every feasible state and within-term node."""
    settings = dict(params=params, types=types, mandates=mandates, inactive_quorums=inactive_quorums)
    if proposal is None:
        proposal = Game(**settings).propose()
    game = Game(exact=True, **settings)
    values, chains = exact_values(game, proposal)
    updated, plan, detail = game.step(values, proposed=proposal)
    assert plan == proposal, "An exact deviation or tie decision differs from the proposal."
    for record, world in ((0, 0), (0, 1), (1, 0), (2, 1)):
        assert (values[:, :, :, record, world] == updated[:, :, :, record, world]).all()
    return game, plan, detail, chains


def at(result, incumbent, env, record=0):
    _, plan, detail, _ = result
    programs = plan["programs", incumbent, env, record]
    node = ("election", incumbent, env, record, *programs)
    return programs, plan[node], detail[node]


def report(result):
    _, _, detail, _ = result
    for record, env, incumbent in product(RECORDS, ENVS, PARTIES):
        offers, choice, stats = at(result, incumbent, env, record)
        loss = stats.right if incumbent == 0 else 1 - stats.right
        risk = loss * stats.q * choice.refuse
        margins = detail["programs", incumbent, env, record]
        print(
            f"party={NAMES[incumbent]}, environment={env}, record={record}, programs={offers}, "
            f"Pr(R certificate)={float(stats.right):.2f}, conditional success={float(stats.q):.2f}, "
            f"gain={float(stats.gain):.5f}, refuse={choice.refuse}, takeover={float(risk):.4f}, "
            f"willing blocs={stats.willing}, voter gaps={np.round(np.asarray(stats.gaps, dtype=float), 3)}, "
            f"campaign margins={np.round(np.asarray(margins, dtype=float), 3)}"
        )
    print("All institutional choices, ballots, party deviations and Bellman equations verified.")


if __name__ == "__main__":
    report(certify())
