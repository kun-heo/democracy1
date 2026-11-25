"""Check the two-date transfer example and the policy-learning identities.

Run with Python 3 and SymPy: python verify_policy_learning.py.
All constitutional decisions below are calculated after a certified defeat.
"""

from sympy import Matrix, Rational, Symbol, diff, simplify, sqrt, symbols


def check_policy_records():
    # A binary state has squared realization V. A reveal-or-null signal
    # yields m = omega with probability r and m = 0 otherwise.
    base, anchor, slope = symbols("base anchor slope", real=True)
    size = Symbol("size", positive=True)
    prob = Symbol("r", nonnegative=True)
    second_moment = sum(
        Rational(1, 2) * wt * ((base + slope*mean - anchor)*omega)**2
        for omega in (-size, size)
        for mean, wt in ((omega, prob), (0, 1-prob))
    )
    claimed = size**2 * ((base-anchor)**2 + slope**2 * prob*size**2)
    assert simplify(second_moment-claimed) == 0

    # A Brownian map pinned at 0 and at b. The drift drops out of
    # the bridge's conditional mean and variance inside (0,b).
    t, b, d, process_var = symbols("t b d process_var", positive=True)
    drift, value_at_b = symbols("drift value_at_b", real=True)
    mean_inside = drift*t + t/b*(value_at_b - drift*b)
    variance_inside = process_var*t - (process_var*t)**2/(process_var*b)
    assert simplify(mean_inside - t/b*value_at_b) == 0
    assert simplify(variance_inside - process_var*t*(b-t)/b) == 0
    alt = Symbol("other_drift", real=True)
    extra_evidence = ((drift-alt)*d)**2/(2*process_var*d)
    assert simplify(extra_evidence - d*(drift-alt)**2/(2*process_var)) == 0



def check_drift_contest():
    # Observe several exact increments of a persistent Brownian map.
    length = symbols("l1:4", positive=True)
    observed = symbols("y1:4", real=True)
    drift = Symbol("a", real=True)
    roughness = Symbol("nu_squared", positive=True)
    log_score = -sum((dy-drift*dx)**2/(2*roughness*dx)
                     for dx, dy in zip(length, observed))
    assert simplify(diff(log_score, drift)
                    - (sum(observed)-drift*sum(length))/roughness) == 0
    assert simplify(diff(log_score, drift, 2) + sum(length)/roughness) == 0

    # Add the voter's private frontier observation. The center of the
    # resulting quadratic score is exactly the model-fit ideal X.
    width, distance, new_var = symbols("D d Vd", positive=True)
    endpoint_change, local_signal = symbols("endpoint_change Z", real=True)
    public_score = (drift*endpoint_change-drift**2*width/2)/roughness
    private_score = -(local_signal-drift*distance)**2/(2*new_var)
    precision = width/roughness+distance**2/new_var
    best_fit = (endpoint_change/roughness+distance*local_signal/new_var)/precision
    assert simplify(diff(public_score+private_score, drift).subs(drift, best_fit)) == 0
    assert simplify(diff(public_score+private_score, drift, 2)+precision) == 0

def check_experiment():
    variance, noise, discounted_stakes = symbols("v sigma_squared K", positive=True)
    size_squared = Symbol("y", nonnegative=True)
    surplus = (-size_squared + discounted_stakes*variance**2*size_squared
               / (noise + variance*size_squared))
    interior = sqrt(discounted_stakes*noise) - noise/variance
    assert simplify(diff(surplus, size_squared).subs(size_squared, interior)) == 0
    optimum_value = (sqrt(discounted_stakes*variance) - sqrt(noise/variance))**2
    assert simplify(surplus.subs(size_squared, interior) - optimum_value) == 0
    assert simplify(diff(surplus, size_squared).subs(size_squared, 0)
                    - (discounted_stakes*variance**2/noise-1)) == 0


def after_defeat(spread):
    delta = Rational(4, 5)
    regime_cost = Rational(1, 10)
    attempt_cost = sanction = Rational(1, 10)
    endorsement_cost = Rational(1, 20)
    pilot_squared = max(0, sqrt(delta*spread)-1)
    # The prior and measurement noise have unit variance. The first
    # state is +/-1, so the nonnegative optimal pilot is the same at both.
    learning_value = (-pilot_squared
                      + delta*spread*pilot_squared/(1+pilot_squared))
    ruler_gain = learning_value - (1+delta)*regime_cost
    gain_by_model = {}
    for belief in (0, 1):
        # Conditional on the bloc's adopted point model, the officeholder's
        # posterior mean after the pilot is normally distributed.
        post_mean = pilot_squared*belief/(1+pilot_squared)
        post_var = pilot_squared/(1+pilot_squared)**2
        second_moment = post_mean**2+post_var
        terminal_flow = spread*(2*belief*post_mean-second_moment)
        gain_by_model[belief] = (-pilot_squared + delta*terminal_flow
                                 - (1+delta)*regime_cost)
    willing = sum(1 for gain in gain_by_model.values() if -gain > endorsement_cost)
    success = sum(Rational(1, 2) for quorum in (1, 2) if quorum > willing)
    refusal = success*ruler_gain - (1-success)*sanction-attempt_cost
    return (pilot_squared, learning_value, ruler_gain, gain_by_model, willing,
            success, refusal)


def check_ideology_contest():
    # FOC linear system for F(c) and gap*f(c) at the two interior optima.
    a = Matrix([[1, -Rational(1, 2)], [-1, -Rational(1, 2)]])
    rhs = Matrix([0, -1])
    assert a.inv()*rhs == Matrix([Rational(1, 2), 1])


if __name__ == "__main__":
    check_policy_records()
    check_drift_contest()
    check_experiment()
    check_ideology_contest()
    low = after_defeat(1)
    high = after_defeat(5)
    assert low == (0, 0, -Rational(9, 50),
                   {0: -Rational(9, 50), 1: -Rational(9, 50)},
                   2, 0, -Rational(1, 5))
    assert high == (1, 1, Rational(41, 50),
                    {0: -Rational(109, 50), 1: Rational(41, 50)},
                    1, Rational(1, 2), Rational(13, 50))
    print("S2  pilot^2  value of learning  A_I-C_I  willing blocs  Pr(success)  refusal gain")
    for s, values in ((1, low), (5, high)):
        pilot, learning, ruler, _, willing, success, refusal = values
        print(s, pilot, learning, ruler, willing, success, refusal, sep="     ")
