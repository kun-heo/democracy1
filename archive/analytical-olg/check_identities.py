"""Independent algebra checks for the closed-form OLG political game."""
import sympy as sp


def equal(left, right):
    assert sp.factor(left-right) == 0


def stationary_parties():
    u, v, z, x, q, delta, g, chi, cost = sp.symbols("u v z x q delta g chi C")
    H, O, T = sp.symbols("H O T")
    t = 1-delta
    a = 1-delta*u
    d0 = t*(1+delta*(v-u))
    d1 = delta**2*v*(1-u)
    det = d0+d1*z
    O0 = v*((1-z)*g-a*z*chi/t)/det
    gamma = q*delta**2*v*(1-u)*(1-z)/det
    index = q*((g-chi)/t-delta*O0)-cost
    trial_T = index/(1+x*gamma)
    trial_O = O0+delta*v*(1-u)*(1-z)*x*trial_T/det
    trial_H = (u*g+(1-u)*(delta*trial_O+x*trial_T))/a
    equal(trial_T, q*((g-chi)/t-delta*trial_O)-cost)
    equal(trial_H, u*(g+delta*trial_H)+(1-u)*(delta*trial_O+x*trial_T))
    equal(trial_O, (1-v)*delta*trial_O+v*((1-z)*(g+delta*trial_H)-z*chi/t))
    ref = (g-chi)/t-cost/q
    intercept = ref*d0-delta*v*g
    slope = ref*d1+delta*v*(g+a*chi/t)
    equal(index, q*(intercept+slope*z)/det)
    equal(sp.diff(O0, z), -v*((g+a*chi/t)*d0+g*d1)/det**2)
    return index, (u, v, z, q, delta, g, chi, cost)


def symmetric_slice(index, variables):
    u, v, z, q, delta, g, chi, cost = variables
    p, k, ratio = sp.symbols("p k R")
    half = sp.Rational(1, 2)
    disc = sp.Rational(4, 5)
    subs = {u:half, v:half, z:p/2, q:half, delta:disc, chi:0, cost:k}
    # A fair certificate and common refusal success leave q and delta unrestricted.
    t = 1-delta
    b = 1-delta/2
    general = index.subs({u:half, v:half, z:q*p, chi:0, cost:k})
    equal(general, q*g*b*(t+delta*q*p/2)/(t*(t+delta**2*q*p/4))-k)
    equal(sp.solve(general.subs(p,0),g)[0]/k, t/(q*b))
    equal(sp.solve(general.subs(p,1),g)[0]/k,
          t*(t+delta**2*q/4)/(q*b*(t+delta*q/2)))
    equal(sp.solve(general.subs(g,ratio*k),p)[0],
          4*t*(t-q*b*ratio)/(q*delta*(2*q*b*ratio-t*delta)))
    for remaining in (1, 2, 3):
        continuation = sum(delta**r for r in range(remaining-1))*g/2
        gain = q*(g*sum(delta**r for r in range(remaining))-delta*continuation)-k
        equal(sp.solve(gain, g)[0]/k,
              1/(q*(1+sum(delta**r for r in range(1,remaining))/sp.Integer(2))))
    print("Symmetric finite and stationary thresholds and mixed probabilities checked for arbitrary q and delta.")
    reduced = index.subs(subs)
    equal(reduced, 15*g*(1+p)/(2*(5+2*p))-k)
    equal(sp.solve(reduced.subs(p, 0), g)[0]/k, sp.Rational(2,3))
    equal(sp.solve(reduced.subs(p, 1), g)[0]/k, sp.Rational(7,15))
    mixed = sp.solve(reduced.subs(g, ratio*k), p)[0]
    equal(mixed, (10-15*ratio)/(15*ratio-4))
    equal(sp.diff(mixed,ratio), -90/(15*ratio-4)**2)

    def refusal(n, next_opposition):
        return half*(g*sum(disc**r for r in range(n))-disc*next_opposition)-k

    O1, H1 = g/2, g/2  # both always concede at the terminal loss in this branch
    O2 = disc*O1/2 + (g+disc*H1)/2
    cutoffs = [sp.solve(refusal(n, o),g)[0]/k for n,o in ((1,0),(2,O1),(3,O2))]
    assert cutoffs == [2,sp.Rational(10,7),sp.Rational(50,43)]
    O2_refusal = disc*O1/2+(g+disc*H1)/4
    equal(refusal(3,O2_refusal), g-k)
    O1_ref, H1_ref = g/4, 3*g/4-k/2
    O2_both = disc*O1_ref/2+(g+disc*H1_ref)/4
    equal(refusal(3,O2_both), sp.Rational(51,50)*g-sp.Rational(24,25)*k)

    n = sp.symbols("n", integer=True, positive=True)
    lawful_T = half*g*(1+disc*(1-disc**(n-1))/(2*(1-disc)))-k
    equal(sp.limit(lawful_T,n,sp.oo), 3*g/2-k)
    print("Symmetric finite thresholds:", cutoffs)
    print("Stationary multiplicity thresholds: 7/15, 2/3; mixed probability:", mixed)


if __name__ == "__main__":
    idx, symbols = stationary_parties()
    symmetric_slice(idx, symbols)
    print("Stationary Bellman identities, best responses and finite-horizon thresholds proved symbolically.")
