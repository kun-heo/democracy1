"""Reproduce the two-election benchmark with Python 3.11+, NumPy, and SymPy.

Run: python two_election_benchmark.py
The accompanying solve_equilibrium.py is unchanged from the OLG manuscript.
All equilibrium and dominance comparisons below use rational arithmetic.
"""
from fractions import Fraction
from itertools import product
from typing import NamedTuple

import numpy as np
import sympy as sp

from solve_equilibrium import (
    ENVS, NAMES, OLD, PARTIES, RECORDS, SIGNALS, WORLDS, YOUNG,
    Game, available, certify, event_branches,
)


class Term(NamedTuple):
    values: np.ndarray
    plan: dict
    detail: dict


def contest_result(term, incumbent, environment, record=0):
    offers = term.plan["programs", incumbent, environment, record]
    key = ("election", incumbent, environment, record, *offers)
    return offers, term.plan[key], term.detail[key]


def check_values_from_events(game, term, future):
    """Independently reconstruct all values on histories feasible in each world."""
    for world in WORLDS:
        for state in product((0, 1), PARTIES, ENVS, (0, world + 1)):
            _, _, environment, record = state
            flow = np.full(len(NAMES), Fraction(0), dtype=object)
            continuation = np.full(len(NAMES), Fraction(0), dtype=object)
            branches = event_branches(game, state, term.plan, flow)
            assert sum(branches.values()) == 1
            for (regime, government, program), branch_mass in branches.items():
                actions = term.plan["policy", regime, government, environment, record, program]
                for report, p_report, action in zip(SIGNALS, game.sprob[environment], actions):
                    mass = branch_mass * p_report
                    flow += mass * game.flow[regime, government, report][action, world]
                    informative = record == 0 and game.xs[action] != 0
                    release = game.publication if informative else Fraction(0)
                    for next_env, p_env in enumerate(game.ep[environment]):
                        next_values = future[regime, government, next_env]
                        continuation += mass * p_env * (
                            (1 - release) * next_values[record, world]
                            + release * next_values[world + 1, world]
                        )
            actual = term.values[(*state, world)]
            assert (actual[:4] == flow[:4] + game.delta * continuation[:4]).all()
            assert (actual[OLD] == flow[OLD]).all()
            assert (flow[YOUNG] == flow[OLD]).all()
            assert (actual[YOUNG] == flow[YOUNG] + game.delta * continuation[OLD]).all()


def surviving_programs(options, table):
    """Iterate strict pure-strategy dominance, allowing mixed campaigns."""
    left, right = list(options), list(options)
    while True:
        surviving_left = [
            x for x in left
            if not any(all(table[y, z][0] > table[x, z][0] for z in right) for y in left)
        ]
        surviving_right = [
            x for x in right
            if not any(all(table[z, y][1] > table[z, x][1] for z in left) for y in right)
        ]
        if surviving_left == left and surviving_right == right:
            return left, right
        left, right = surviving_left, surviving_right


def check_campaign_dominance(game, term, future):
    """Verify the unique campaign pair under the manuscript's continuation rules."""
    installations = {}
    for state in product((0, 1), PARTIES, ENVS, RECORDS):
        payoffs, _, _ = game.actions(future, *state)
        for program, payoff in payoffs.items():
            installations[*state, program] = payoff

    comparisons = 0
    for inc, env, rec in product(PARTIES, ENVS, RECORDS):
        opponent, options = 1 - inc, available(rec)
        scores = {}
        for pair in product(options, repeat=2):
            victory = installations[0, inc, env, rec, pair[inc]]
            concession = installations[0, opponent, env, rec, pair[opponent]]
            entrenchment = installations[1, inc, env, rec, pair[inc]]
            value, _, _ = game.contest(inc, env, rec, pair, victory, concession, entrenchment)
            scores[pair] = game.expectation(value, rec)[:2]
            comparisons += 1
        left, right = surviving_programs(options, scores)
        chosen = term.plan["programs", inc, env, rec]
        assert (left, right) == ([chosen[0]], [chosen[1]])
    assert comparisons == 96


def after_first_concession(game, first, last, incumbent, environment):
    """Average the final certificate and actual installation after an imposed concession."""
    challenger = 1 - incumbent
    offers, _, _ = contest_result(first, incumbent, environment)
    actions = first.plan["policy", 0, challenger, environment, 0, offers[challenger]]
    release = game.publication * sum(p for p, act in zip(game.sprob[environment], actions)
                                     if game.xs[act] != 0)
    # Each technology has equal prior probability at the unresolved record.
    records = ((0, 1 - release), (1, release / 2), (2, release / 2))
    certificate = office = Fraction(0)
    for next_env, p_env in enumerate(game.ep[environment]):
        for rec, p_rec in records:
            _, election, stats = contest_result(last, challenger, next_env, rec)
            p_win = stats.right if incumbent == 1 else 1 - stats.right
            certificate += p_env * p_rec * p_win
            office += p_env * p_rec * p_win * (1 - election.refuse * stats.q)
    return certificate, office


def verify_formula():
    s, b, d, phi, h = sp.symbols("S b d phi h")
    g1, base, slope, q, delta, k, sanction = sp.symbols("g1 base slope q delta k sanction")
    p = sp.Rational(1, 2) + phi * (1 - h) * (b + s*d)
    g2 = base + slope*s
    gain = q * (g1 + delta*g2*(1 - p*(1 - h))) - k - (1 - q)*sanction
    derivative = delta*q*((1 - p*(1 - h))*slope - g2*phi*(1 - h)**2*d)
    assert sp.expand(sp.diff(gain, s) - derivative) == 0


def main():
    verify_formula()
    game = Game(exact=True)
    future = game.zeros()
    terms = {}
    for date in (2, 1):
        term = Term(*game.step(future))
        check_values_from_events(game, term, future)
        check_campaign_dominance(game, term, future)
        terms[date] = term
        future = term.values

    # All entries start at an unresolved public record. Each number is exact.
    expected = {
        (2, 0, 0): (".32", "-.1916"), (2, 1, 0): (".32", "-.87"),
        (2, 0, 1): (".68", ".0516"), (2, 1, 1): (".88", "-.22744"),
        (1, 0, 0): (".68", "-.0849984"), (1, 1, 0): (".78", "-.87"),
        (1, 0, 1): (".68", ".328312"), (1, 1, 1): (".89", "-.035258816"),
    }
    print("Term    Incumbent    S       Offers     P(R certificate)     Refusal gain     Decision")
    for date, env, inc in product((2, 1), ENVS, PARTIES):
        offers, election, stats = contest_result(terms[date], inc, env)
        assert (stats.right, stats.gain) == tuple(map(Fraction, expected[date, inc, env]))
        left_code = 2 if inc == 1 and (date == 1 or env == 1) else 0
        assert offers == (left_code, 3)
        decision = "refuse" if election.refuse else "concede"
        s = game.high if env else game.low
        print(f"{date:>4}    {NAMES[inc]:>9}  {float(s):5.2f}      {offers}"
              f"           {float(stats.right):.2f}         {float(stats.gain):+.7f}     {decision}")

    targets = {
        (0, 0): (".6322", ".6322"), (0, 1): (".2905", ".2905"),
        (1, 0): (".4004", ".352784"), (1, 1): (".601", ".28356"),
    }
    print("\nAfter a first-term concession, probability of a final certificate and of actual control")
    for inc, env in product(PARTIES, ENVS):
        values = after_first_concession(game, terms[1], terms[2], inc, env)
        assert values == tuple(map(Fraction, targets[inc, env]))
        print(NAMES[inc], "high" if env else "low ", *(f"{float(x):.6f}" for x in values))

    # Same R mandate, unresolved record, lawful installation and high monitoring.
    for date, target in ((1, "-.5"), (2, "0")):
        instruments = terms[date].plan["policy", 0, 1, 1, 0, 3]
        assert game.xs[instruments[0]] == Fraction(target)
    ideal = Fraction(".50") - Fraction(".70")
    assert (Fraction("-.5") - ideal)**2 - ideal**2 == Fraction(".05")

    # Hold the pair of high-monitoring programs in the reported stationary
    # equilibrium fixed while changing the remaining number of elections.
    station_game, station_plan, station_detail, _ = certify()
    assert station_game.high == game.high and station_game.attempt == game.attempt
    key = ("election", 1, 1, 0, 0, 3)
    assert terms[2].detail[key].gain == Fraction(".25256")
    assert terms[1].detail[key].gain == Fraction(".4642964224")
    assert station_plan["programs", 1, 1, 0] == (0, 3)
    assert round(station_detail[key].gain, 4) == Fraction(".5721")
    assert all(station_plan["election", i, 1, 0, 0, 3].refuse for i in PARTIES)
    print("\nAll values, action comparisons, campaign dominance checks and reported quantities verified.")


if __name__ == "__main__":
    main()
