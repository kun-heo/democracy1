"""Verify the finite-menu Markov-perfect equilibrium using exact arithmetic.

Run with Python 3 and SymPy (python verify_equilibrium.py). All numbers in the
game are rational. The final display converts deviation gains to decimals.
"""

from itertools import product

from sympy import Matrix, Rational, eye, zeros


def frac(numerator, denominator=1):
    return Rational(numerator, denominator)


leaders = ("L0", "L1", "R0", "R1")
sides = ("L", "R")
states = ("low", "high", "known")
election_states = tuple(product(leaders, states))
party = {name: name[0] for name in leaders}
opponent = {"L": "R", "R": "L"}

# Type coordinates are (baseline, responsiveness).
points = {"L0": 0, "L1": frac(1, 10), "R0": 1, "R1": frac(9, 10)}
types = {name: (point, point) for name, point in points.items()}
types.update({"partyL": (0, 0), "partyR": (1, 1),
              "blocL": (0, 0), "blocR": (1, 1),
              "vL": (0, 0), "vR": (1, 1),
              "vLR": (frac(1, 5), frac(4, 5)),
              "vRL": (frac(4, 5), frac(1, 5))})
voters = {"vL": frac(1, 4), "vR": frac(1, 4),
          "vLR": frac(7, 20), "vRL": frac(3, 20)}

delta = frac(13, 20)
# Once a public study reveals the common causal slope, all slope distances vanish.
effective_spread = {"low": frac(1, 5), "high": 4, "known": 0}
unresolved = frac(19, 20)
transition = Matrix([[unresolved*frac(9, 10), unresolved*frac(1, 10), 1-unresolved],
                     [unresolved*frac(3, 10), unresolved*frac(7, 10), 1-unresolved],
                     [0, 0, 1]])
rents = frac(1, 5)
act_cost = frac(9, 50)
sanction = frac(7, 20)
defense_cost = frac(7, 50)
prob_both_needed = frac(1, 5)


def flow(actor, holder, state):
    baseline, slope = types[actor]
    holder_base, holder_slope = types[holder]
    policy = -(baseline - holder_base)**2 - effective_spread[state]*(slope - holder_slope)**2
    return policy + (rents if actor in (holder, "party" + party[holder]) else 0)


def regime_loss(actor):
    return frac(21, 10) if actor.startswith(("v", "bloc")) else frac(9, 20)


def autocratic_values():
    values = {}
    inverse = (eye(len(states)) - delta * transition).inv()
    for actor, holder in product(types, leaders):
        payoff = Matrix([flow(actor, holder, s) - regime_loss(actor) for s in states])
        result = inverse * payoff
        for ix, state in enumerate(states):
            values[actor, holder, state] = result[ix]
    return values


def nominee(inc, state):
    if state == "known":
        return "R0" if party[inc] == "L" else "L0"
    if party[inc] == "L":
        return "R1"
    return "L1" if (inc, state) == ("R1", "high") else "L0"


def legal_inc_win(inc, state):
    right_wins = frac(3, 5) if state == "high" else frac(2, 5)
    return right_wins if party[inc] == "R" else 1 - right_wins


def ordered_defenders(eligible, new_side, quorum):
    if len(eligible) < quorum:
        return ()
    order = (new_side, opponent[new_side])
    return tuple(side for side in order if side in eligible)[:quorum]


def enforcement(eligible, new_side):
    failure = 0
    expected_cost = {side: 0 for side in sides}
    for quorum, probability in ((1, 1 - prob_both_needed), (2, prob_both_needed)):
        group = ordered_defenders(eligible, new_side, quorum)
        if group:
            for side in group:
                expected_cost[side] += probability * defense_cost
        else:
            failure += probability
    return failure, expected_cost


def cost_for(actor, inc, failure, expected_cost):
    if actor == inc:
        return act_cost + (1 - failure)*sanction
    if actor in ("blocL", "blocR"):
        return expected_cost[actor[-1]]
    return 0


def democratic_values(autocracy):
    # Profile to verify. Solve its discounted transition system first.
    index = {key: j for j, key in enumerate(election_states)}
    kernel = zeros(len(index))
    rewards = zeros(len(index), len(types))
    for pos, (inc, state) in enumerate(election_states):
        challenger = nominee(inc, state)
        win = legal_inc_win(inc, state)
        attack = state == "high"
        failure, costs = enforcement((party[challenger],), party[challenger])
        if not attack:
            failure = 0
        for k, next_state in enumerate(states):
            prob = transition[states.index(state), k]
            kernel[pos, index[inc, next_state]] += win*prob
            kernel[pos, index[challenger, next_state]] += (1-win)*(1-failure)*prob
        for col, actor in enumerate(types):
            lose_flow = (1-failure)*flow(actor, challenger, state)
            if attack:
                lose_flow += failure*autocracy[actor, inc, state] - cost_for(actor, inc, failure, costs)
            rewards[pos, col] = win*flow(actor, inc, state) + (1-win)*lose_flow
    solution = (eye(len(index)) - delta*kernel).inv()*rewards
    assert solution == rewards + delta*kernel*solution
    return {(actor, inc, s): solution[pos, col]
            for pos, (inc, s) in enumerate(election_states)
            for col, actor in enumerate(types)}


def legal_values(democracy):
    values = {}
    for actor, holder, state in product(types, leaders, states):
        future = sum(transition[states.index(state), k]*democracy[actor, holder, new]
                     for k, new in enumerate(states))
        values[actor, holder, state] = flow(actor, holder, state) + delta*future
    return values


def contest(inc, challenger, state, autocracy, lawful):
    challenger_side = party[challenger]
    incumbency_gain = {actor: autocracy[actor, inc, state] - lawful[actor, challenger, state]
                       for actor in types}
    defense_gains = {side: -incumbency_gain["bloc" + side] - defense_cost for side in sides}
    assert all(gap != 0 for gap in defense_gains.values())
    eligible = tuple(side for side in sides if defense_gains[side] > 0)
    failure, costs = enforcement(eligible, challenger_side)
    net_gain = failure*incumbency_gain[inc] - cost_for(inc, inc, failure, costs)
    assert net_gain != 0
    attack = net_gain > 0
    after_loss = {}
    for actor in types:
        outcome = lawful[actor, challenger, state]
        if attack:
            outcome += failure*incumbency_gain[actor] - cost_for(actor, inc, failure, costs)
        after_loss[actor] = outcome
    vote_gaps = {v: lawful[v, inc, state] - after_loss[v] for v in voters}
    assert all(gap != 0 for gap in vote_gaps.values())
    win = sum(weight for v, weight in voters.items() if vote_gaps[v] > 0)
    values = {actor: win*lawful[actor, inc, state] + (1-win)*after_loss[actor] for actor in types}
    return values, (win, eligible, failure, net_gain, vote_gaps)


def breakdown_before_public_study():
    # Known-slope democratic states and autocracies are both terminal for this event.
    uncertain_states = [key for key in election_states if key[1] != "known"]
    index = {key: position for position, key in enumerate(uncertain_states)}
    kernel = zeros(len(index))
    hazard = zeros(len(index), 1)
    for current, row in index.items():
        inc, state = current
        challenger = nominee(inc, state)
        win = legal_inc_win(inc, state)
        failure = prob_both_needed if state == "high" else 0
        hazard[row] = (1-win)*failure
        for next_state in ("low", "high"):
            trans = transition[states.index(state), states.index(next_state)]
            kernel[row, index[inc, next_state]] += win*trans
            kernel[row, index[challenger, next_state]] += (1-win)*(1-failure)*trans
    probabilities = (eye(len(index)) - kernel).inv()*hazard
    assert probabilities == hazard + kernel*probabilities
    return {key: probabilities[row] for key, row in index.items()}


def main():
    autocracy = autocratic_values()
    democracy = democratic_values(autocracy)
    lawful = legal_values(democracy)
    print("Inc  S      nominee  Pr(R certified)  Pr(success|attempt)  gain to attempt  nomination margin")
    for inc, state in election_states:
        challenger = nominee(inc, state)
        other = next(name for name in leaders if party[name] == party[challenger] and name != challenger)
        chosen, details = contest(inc, challenger, state, autocracy, lawful)
        alternative, _ = contest(inc, other, state, autocracy, lawful)
        win, willing, failure, gain, vote_gaps = details
        decision_maker = "party" + party[challenger]
        nomination_margin = chosen[decision_maker] - alternative[decision_maker]
        assert nomination_margin > 0
        assert win == legal_inc_win(inc, state)
        assert (gain > 0) == (state == "high")
        assert set(willing) == (set(sides) if state != "high" else {party[challenger]})
        assert all(chosen[actor] == democracy[actor, inc, state] for actor in types)
        for voter, ordinary_choice in (("vL", "L"), ("vR", "R"), ("vLR", "L"), ("vRL", "R")):
            selected = party[inc] if vote_gaps[voter] > 0 else party[challenger]
            switches = state == "high" and voter in ("vLR", "vRL")
            assert selected == (opponent[ordinary_choice] if switches else ordinary_choice)
        assert all(autocracy[v, inc, state] < lawful[v, challenger, state] for v in ("vLR", "vRL"))
        right_wins = win if party[inc] == "R" else 1-win
        print(f"{inc:3}  {state:5}  {challenger:7}  {float(right_wins):15.2f}"
              f"  {float(failure):19.2f}  {float(gain):15.6f}  {float(nomination_margin):17.6f}")

    chances = breakdown_before_public_study()
    print("\nTakeover before the public study, by starting democratic state")
    for key, chance in chances.items():
        print(*key, f"{float(chance):.4f}")


if __name__ == "__main__":
    main()
