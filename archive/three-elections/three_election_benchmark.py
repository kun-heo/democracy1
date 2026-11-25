"""Exact backward induction for the three-election OLG benchmark.

Run with Python 3.11+, NumPy, and SymPy: python three_election_benchmark.py
The other two Python files in this directory are unchanged from the two-election note.
"""
from fractions import Fraction as F
from itertools import product
from typing import NamedTuple

import numpy as np

from solve_equilibrium import (
    AUT, ENVS, NAMES, PARTIES, RECORDS, WORLDS, Detail, Election, Game, available, event_branches,
)
from two_election_benchmark import (
    Term, check_campaign_dominance, check_values_from_events, contest_result, surviving_programs,
)


class Contest(NamedTuple):
    value: np.ndarray
    election: Election
    stats: Detail


class Campaign(NamedTuple):
    contests: dict
    distribution: dict
    tied_left: tuple


def campaign_equilibrium(state, options, payoffs):
    """Reduce each simultaneous game and solve its remaining indifferences exactly."""
    left, right = surviving_programs(options, payoffs)
    if len(right) == 1:
        fixed = right[0]
        assert all(payoffs[row, fixed][0] == payoffs[left[0], fixed][0] for row in left)
        return {(left[0], fixed): F(1)}, tuple(left)

    assert state == (1, 0, 0) and (left, right) == ([0, 1, 2], [0, 1, 3])
    # The combination 3/4 L0 + 1/4 L2 strictly dominates L1.
    assert all(3 * payoffs[0, s][0] + payoffs[2, s][0] > 4 * payoffs[1, s][0]
               for s in right)
    assert all(payoffs[l, 3][1] > payoffs[l, 0][1] for l in (0, 2))
    # R1 versus R3, first against L0 and then against L2.
    right_gain_at_0 = payoffs[0, 1][1] - payoffs[0, 3][1]
    right_gain_at_2 = payoffs[2, 1][1] - payoffs[2, 3][1]
    # L0 versus L2, first against R1 and then against R3.
    left_gain_at_1 = payoffs[0, 1][0] - payoffs[2, 1][0]
    left_gain_at_3 = payoffs[0, 3][0] - payoffs[2, 3][0]
    assert right_gain_at_0 > 0 > right_gain_at_2 and left_gain_at_3 > 0 > left_gain_at_1
    left_0 = -right_gain_at_2 / (right_gain_at_0 - right_gain_at_2)
    right_1 = left_gain_at_3 / (left_gain_at_3 - left_gain_at_1)
    distribution = {(l, s): p*q for l, p in ((0, left_0), (2, 1-left_0))
                    for s, q in ((1, right_1), (3, 1-right_1))}
    return distribution, ()


def check_mixed_deviations(options, payoffs, distribution):
    assert sum(distribution.values()) == 1 and all(p > 0 for p in distribution.values())
    left = {l: sum(p for (x, _), p in distribution.items() if x == l) for l in options}
    right = {s: sum(p for (_, y), p in distribution.items() if y == s) for s in options}
    assert all(distribution.get((l, s), 0) == left[l]*right[s] for l, s in product(options, repeat=2))
    left_scores = {l: sum(q*payoffs[l, s][0] for s, q in right.items()) for l in options}
    right_scores = {s: sum(p*payoffs[l, s][1] for l, p in left.items()) for s in options}
    assert all(left_scores[l] == max(left_scores.values()) for l in options if left[l])
    assert all(right_scores[s] == max(right_scores.values()) for s in options if right[s])


def first_date(game, future):
    """Find contingent policies, transfer decisions and equilibria at the first date."""
    plans, installations, campaigns = {}, {}, {}
    for state in product((0, 1), PARTIES, ENVS, RECORDS):
        values, rules, _ = game.actions(future, *state)
        for program, value in values.items():
            installations[*state, program] = value
            plans["policy", *state, program] = rules[program]
    for inc, env, record in product(PARTIES, ENVS, RECORDS):
        opposite, options = 1-inc, available(record)
        contests, payoffs = {}, {}
        for pair in product(options, repeat=2):
            victory = installations[0, inc, env, record, pair[inc]]
            concession = installations[0, opposite, env, record, pair[opposite]]
            entrenchment = installations[1, inc, env, record, pair[inc]]
            contest = Contest(*game.contest(inc, env, record, pair, victory, concession, entrenchment))
            contests[pair] = contest
            payoffs[pair] = game.expectation(contest.value, record)[:2]
            plans["election", inc, env, record, *pair] = contest.election
        distribution, ties = campaign_equilibrium((inc, env, record), options, payoffs)
        check_mixed_deviations(options, payoffs, distribution)
        if len(ties) > 1:
            # Substitution leaves the certificate, compliance decision, enforcement,
            # material payoffs in both worlds and the installed policies unchanged.
            (l, s), = distribution
            for other in ties:
                pair = (other, s)
                base, alt = contests[l, s], contests[pair]
                assert base.election[:3] == alt.election[:3] and base.stats.gain == alt.stats.gain
                assert (base.value == alt.value).all()
                for regime, party in product((0, 1), PARTIES):
                    assert (plans["policy", regime, party, env, record, (l, s)[party]] ==
                            plans["policy", regime, party, env, record, pair[party]])
        campaigns[inc, env, record] = Campaign(contests, distribution, ties)

    assert sum(len(c.contests) for c in campaigns.values()) == 96
    assert {st: c.tied_left for st, c in campaigns.items() if len(c.tied_left) > 1} == {
        (0, 0, 0): (0, 1), (1, 1, 0): (0, 1)}
    # The event tree for every component of the first-date mixture and the
    # indifferent offers can be checked using deterministic contingent plans.
    choices = {}
    for st, c in campaigns.items():
        choices[st] = list(c.distribution)
        if len(c.tied_left) > 1:
            (_, s), = c.distribution
            choices[st] = [(l, s) for l in c.tied_left]
    states = tuple(campaigns)
    for combination in product(*(choices[st] for st in states)):
        plan = dict(plans)
        values = game.zeros()
        for st, pair in zip(states, combination):
            inc, env, rec = st
            plan["programs", *st] = pair
            values[0, *st] = campaigns[st].contests[pair].value
            values[1, *st] = installations[1, inc, env, rec, AUT]
        check_values_from_events(game, Term(values, plan, {}), future)
    return plans, campaigns, installations


def next_public_states(game, plan, world, env, record, regime, party, program):
    """Public states at the next decision after a specified government is installed."""
    actions = plan["policy", regime, party, env, record, program]
    transitions = {}
    for p_report, action in zip(game.sprob[env], actions):
        informs = record == 0 and game.xs[action] != 0
        release = game.publication if informs else F(0)
        for next_env, p_env in enumerate(game.ep[env]):
            for next_record, p_rec in ((record, 1-release), (world+1, release)):
                if p_rec:
                    state = (regime, party, next_env, next_record)
                    transitions[state] = transitions.get(state, 0) + p_report*p_env*p_rec
    assert sum(transitions.values()) == 1
    return transitions


def installations_in_term(game, state, term):
    rewards = np.full(len(NAMES), F(0), dtype=object)
    branches = event_branches(game, state, term.plan, rewards)
    assert sum(branches.values()) == 1
    return branches


class Returns(NamedTuple):
    certificate2: F
    office2: F
    first_return3: F
    office3: F
    by3: F


def returns_after_offer(game, plans, second, last, incumbent, env, record, offers, world):
    other = 1 - incumbent
    out = np.full(5, F(0), dtype=object)
    first_transition = next_public_states(game, plans, world, env, record, 0, other, offers[other])
    for state2, p in first_transition.items():
        _, _, details = contest_result(second, state2[1], state2[2], state2[3])
        out[0] += p * (details.right if incumbent == 1 else 1-details.right)
        for (regime2, party2, program), q in installations_in_term(game, state2, second).items():
            returns2 = party2 == incumbent
            out[1] += p * q * returns2
            final_transition = next_public_states(game, second.plan, world, state2[2], state2[3],
                                                   regime2, party2, program)
            for state3, s in final_transition.items():
                for (_, party3, _), v in installations_in_term(game, state3, last).items():
                    returns3 = party3 == incumbent
                    mass = p*q*s*v
                    out[2] += mass * (returns3 and not returns2)
                    out[3] += mass * returns3
                    out[4] += mass * (returns2 or returns3)
    assert out[4] == out[1]+out[2] and out[2] <= 1-out[0]
    return out


def aggregate_first(game, plans, campaigns, second, last, inc, env):
    campaign = campaigns[inc, env, 0]
    certificate = loss = refusal = takeover = F(0)
    conditional_returns = np.full(5, F(0), dtype=object)
    for pair, p in campaign.distribution.items():
        _, election, stats = campaign.contests[pair]
        loss_given_pair = stats.right if inc == 0 else 1-stats.right
        loss_mass = p*loss_given_pair
        certificate += p*stats.right
        loss += loss_mass
        refusal += loss_mass*int(election.refuse)
        takeover += loss_mass*int(election.refuse)*stats.q
        for world in WORLDS:
            outcome = returns_after_offer(game, plans, second, last, inc, env, 0, pair, world)
            conditional_returns += (loss_mass/2)*outcome
    assert loss == (certificate if inc == 0 else 1-certificate)
    return certificate, refusal/loss, takeover, Returns(*(conditional_returns/loss))


def main():
    game = Game(exact=True)
    zero = game.zeros()
    last = Term(*game.step(zero))
    second = Term(*game.step(last.values))
    for term, continuation in ((last, zero), (second, last.values)):
        check_values_from_events(game, term, continuation)
        check_campaign_dominance(game, term, continuation)
    plans, campaigns, installed = first_date(game, second.values)
    mixed = campaigns[1, 0, 0].distribution
    p_left = sum(p for (l, _), p in mixed.items() if l == 0)
    p_right = sum(p for (_, r), p in mixed.items() if r == 1)
    assert p_left == F(4680712875010686, 23660196986955187)
    assert p_right == F(36424654052437133, 132010666342686133)
    print(f"Low monitoring, right incumbent: P(L0)={float(p_left):.10f}, P(R1)={float(p_right):.10f}")
    for (l, s), weight in mixed.items():
        c = campaigns[1, 0, 0].contests[l, s]
        util = game.expectation(c.value, 0)[:2]
        print(f"  pair {(l,s)}: party payoffs {float(util[0]):.6f}, {float(util[1]):.6f}; "
              f"mass {float(weight):.6f}; R certificate {float(c.stats.right):.2f}; "
              f"T_R {float(c.stats.gain):+.6f}; refuse {c.election.refuse}")
    print("Term, T_L and T_R under high monitoring, unresolved record:")
    for date, term in ((2, second), (3, last)):
        print(date, *(float(contest_result(term, inc, 1)[2].gain) for inc in PARTIES))
    print(1, *(float(campaigns[inc, 1, 0].contests[0, 3].stats.gain) for inc in PARTIES))

    # Compare the right incumbent at identical announced programs and current
    # policy rules after a concession or a successful refusal.
    pair, inc, env, rec = (2, 3), 1, 1, 0
    key = ("election", inc, env, rec, *pair)
    assert second.detail[key].q == campaigns[inc, env, rec].contests[pair].stats.q == F(".64")
    assert second.detail[key].gain == F("-.035258816")
    assert campaigns[inc, env, rec].contests[pair].stats.gain == F(".03716781804544")
    assert (game.attempt + (1-F(".64")) * game.sanction)/F(".64") == F("1.059375")
    second_installed = []
    for regime, party, program in ((1, 1, 3), (0, 0, 2)):
        assert (plans["policy", regime, party, env, rec, program] ==
                second.plan["policy", regime, party, env, rec, program])
        payoffs, _, _ = game.actions(last.values, regime, party, env, rec)
        second_installed.append(game.expectation(payoffs[program], rec)[inc])
    difference = installed[1, 1, env, rec, 3] - installed[0, 0, env, rec, 2]
    assert second_installed[0]-second_installed[1] == F("1.0042831")
    assert game.expectation(difference, rec)[inc] == F("1.117449715696")

    print("First election: starting holder / monitoring / R certificate / refusal conditional on loss / takeover.")
    print("After conceding a certified loss: next certificate / next control / first return in term 3 / "
          "term 3 control / any return by term 3.")
    for inc, env in product(PARTIES, ENVS):
        cert, refuse, takeover, ret = aggregate_first(game, plans, campaigns, second, last, inc, env)
        print(NAMES[inc], "high" if env else "low", *(f"{float(x):.9f}" for x in (cert, refuse, takeover)))
        print("   returns:", *(f"{float(x):.9f}" for x in ret))
        if (inc, env) == (1, 0):
            assert takeover == p_left*p_right*F(".22")*F(".64")
            assert refuse == p_left*p_right*F(".22")/(1-cert)
            assert round(ret.certificate2, 8) == F(".60062493")
            assert round(ret.office2, 8) == F(".54545885")
            assert round(ret.by3, 8) == F(".69289959")
        if (inc, env) == (1, 1):
            assert (cert, refuse, takeover) == (F(".88"), 1, F(".0768"))
            assert (ret.certificate2, ret.office2, ret.by3) == (F(".68"), F(".33184"), F(".418261504"))
    print("All campaign, institutional, continuation and path calculations verified.")


if __name__ == "__main__":
    main()
