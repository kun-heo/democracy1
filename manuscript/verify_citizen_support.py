"""Two elections with public backing for refusal. Run from any directory.

python verify_citizen_support.py
python verify_citizen_support.py --figures
"""

import argparse
from dataclasses import dataclass
from functools import lru_cache
from math import pi, sqrt
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq


# The parties, followed by the four citizen types. Both cohorts have the same types.
L, R = 0, 1
CITIZENS = slice(2, 6)
BASE = np.array([-.5, 1.5, -.5, .45, .505, 1.5])
GAMMA = np.array([-2., 2., 1., 1., 1., 1.])
DISCOUNT, S2 = .25, .05
SECOND_VALENCE = np.array([0., 0., 45., 45., 45., 45.])
SHOCKS = np.array([-100., 0., 100.])
SHOCK_PROB = np.array([.45, .10, .45])
TOL = 3e-12  # Implement indifference at a numerically computed boundary.


def flow(base, weight, mean, second_moment):
    """Conditional material payoff. The last dimension indexes actors."""
    mean = np.asarray(mean)[..., None]
    second_moment = np.asarray(second_moment)[..., None]
    weight = np.asarray(weight)[..., None]
    return (-(base - BASE) ** 2
            + (2 * GAMMA * base - 2 * weight * (base - BASE)) * mean
            + (2 * GAMMA * weight - weight**2) * second_moment)


class Election:
    """Posterior integration and payoffs at both report endpoints."""

    def __init__(self, order=64, responsive=True):
        self.responsive = responsive
        # When omega=+1, the two roots at y=2 and the lower root at y=1
        # delimit the only intervals where L wins the neutral final ballot.
        def roots(y):
            return np.sort(np.roots([32.25 * y*y, 1.25*y*y + 2.01*y,
                                     .0125*y*y + .02]))
        lo, full_lo, hi = roots(1)[0], *roots(2)
        self.switches = (lo, full_lo, hi)
        nodes, weights = leggauss(order)
        points, corrections = [], []
        for start, stop, full in [(lo, full_lo, False), (full_lo, hi, True)]:
            t = (start + stop)/2 + (stop - start)/2 * nodes
            v = t*t + S2 * (.5 + 25*t)**2
            y = 2 if full else (-2.01*t + np.sqrt((2.01*t)**2-.08*v))/(2*v)
            right, left = flow(1.5, 1, t, v), flow(-.5, -1, t, v)
            offered = flow(-.5, 1-y, t, v)
            correction = .55 * offered - .45 * left - .10 * right
            if not full:
                # The largest winning offer beats the full, losing offer.
                assert np.all(correction[:, L] > 0)
            correction = np.column_stack((correction, np.full(len(t), .10)))
            points.extend(t)
            corrections.extend(correction * (weights * (stop-start)/2)[:, None])
        self.points = np.asarray(points)
        self.corrections = np.asarray(corrections)

    @lru_cache(maxsize=8192)
    def trial(self, x, omega):
        """Democratic and autocratic terminal values and L's lawful win odds."""
        sd = .01 * abs(x) / sqrt(.26*x*x + .0625)
        ev = .0125 + 32.25 * sd*sd
        dem = .55 * flow(1.5, 1, 0, ev) + .45 * flow(-.5, -1, 0, ev)
        aut = flow(1.5, 2, 0, ev)
        p_left = .45
        if self.responsive and omega == 1 and sd > 1e-12:
            density = np.exp(-.5*(self.points/sd)**2)/(sqrt(2*pi)*sd)
            change = density @ self.corrections
            dem += change[:-1]
            p_left += change[-1]
        return np.r_[dem + SECOND_VALENCE, aut, p_left]

    @lru_cache(maxsize=8192)
    def endpoints(self, party, offer):
        base = BASE[party]
        weight = offer if party == R else -offer
        values = []
        for report in (0, 1):
            continuation = sum(self.trial(base + report*.5*weight*w, w)
                               for w in (-1, 1)) / 2
            old = flow(base, weight, 0, .25*report)
            dem = old + DISCOUNT * continuation[:6]
            aut = old + DISCOUNT * continuation[6:12]
            values.append(np.r_[old, dem, aut, continuation[-1]])
        return np.asarray(values)

    def branch(self, party, offer, s):
        low, high = self.endpoints(party, float(offer))
        vector = low + s*(high-low)
        return vector[:6], vector[6:12], vector[12:18], vector[-1]


@dataclass(frozen=True)
class Institutions:
    k: float = 2.4
    mass: float = .40
    intercept: float = .35
    slope: float = .60

    def __post_init__(self):
        assert 0 <= self.mass <= .40 and 0 <= self.intercept <= 1
        assert 0 <= self.slope <= 1-self.intercept and self.k > 0

    @property
    def weights(self):
        return np.array([.40-self.mass, self.mass, .35, .25])

    def success(self, backing):
        return self.intercept + self.slope*backing


@dataclass
class Outcome:
    ell: float
    r: float
    backing: float
    q: float
    refuse: bool
    p: float
    left_value: float
    right_value: float
    ballots: np.ndarray


def evaluate(game, inst, s, ell, r=1., guarantee=False):
    old_r, win, aut, _ = game.branch(R, r, s)
    old_l, concede, _, _ = game.branch(L, ell, s)
    old_gain, young_gain = old_r-old_l, aut-concede
    weights = inst.weights
    old_back = (old_gain[CITIZENS] > TOL) @ weights
    young_back = (young_gain[CITIZENS] > TOL) @ weights
    backing = (old_back+young_back)/2
    q = inst.success(backing)
    incentive = q*young_gain[R]-inst.k
    refuse = bool(incentive > TOL and not (guarantee and ell == 0))
    retention = q if refuse else 0.
    old_ballot = (1-retention)*old_gain[CITIZENS]
    young_ballot = (win-concede-retention*young_gain)[CITIZENS]
    ballots = np.array([((margin + shock > TOL) @ weights)
                        for shock in SHOCKS for margin in (old_ballot, young_ballot)])
    ballots = ballots.reshape((3, 2))
    p = float((ballots.mean(axis=1) <= .5+TOL) @ SHOCK_PROB)
    after_loss = concede + retention * young_gain
    left = (1-p)*win[L] + p*after_loss[L]
    right = (1-p)*win[R] + p*(after_loss[R]-inst.k*refuse)
    return Outcome(ell, r, backing, q, refuse, p, left, right, ballots)


def stakes(game, s, ell):
    return game.branch(R, 1., s)[2][R] - game.branch(L, ell, s)[1][R]


def civic_offer(s):
    return min(1., max(0., sqrt(.8/s)-1)) if s > 0 else 1.


def solve(game, inst, s, guarantee=False):
    # r=1 strictly dominates for S>0. Candidate sets follow from the increasing
    # L payoff within every interval with the same vote, support and transfer.
    offers = {0., 1., civic_offer(s)}
    if s > 0:
        for backing in (.30, .30 + inst.mass/2):
            q = inst.success(backing)
            def incentive(ell):
                return q*stakes(game, s, ell)-inst.k
            if incentive(0.) < 0 < incentive(1.):
                offers.add(brentq(incentive, 0., 1., xtol=1e-14))
    options = [evaluate(game, inst, s, ell, guarantee=guarantee) for ell in offers]
    best = max(o.left_value for o in options)
    return max((o for o in options if o.left_value >= best-TOL), key=lambda o: o.ell)


def repeal_gain(game, inst, s):
    return solve(game, inst, s).right_value - solve(game, inst, s, True).right_value


def voluntary_end(game, inst, ell):
    """Supremum of S with voluntary concession, accounting for citizen jumps."""
    switch = .8/(1+ell)**2
    breaks = sorted({0., 1., min(1., switch)})
    d0, d1 = (stakes(game, s, ell) for s in (0., 1.))
    cuts = set(breaks)
    for a,b in zip(breaks,breaks[1:]):
        q = evaluate(game, inst, (a+b)/2, ell).q
        if q > 0 and d1 != d0:
            root = (inst.k/q-d0)/(d1-d0)
            if a < root < b:
                cuts.add(root)
    feasible = [s for s in cuts if not evaluate(game, inst, s, ell).refuse]
    return max(feasible) if feasible else None


def repeal_cutoffs(game, inst):
    """All transitions for the three Table 3 specifications.

    On the constant-support outer segments the gain is affine. On the middle
    segment the changing-mandate gain for the first specification is increasing
    by the bounds in Appendix D. Include the jumps separately.
    """
    end = voluntary_end(game, inst, 0.)
    cuts = sorted({0., .2, .8, 1.} | ({float(end)} if end is not None else set()))
    results = []
    for cost in (0., .01):
        first = None
        for a, b in zip(cuts, cuts[1:]):
            lo, hi = a+1e-8, b-1e-8
            def excess(s):
                return repeal_gain(game, inst, s)-cost
            if excess(lo) > TOL:
                first = a
                break
            if excess(hi) > TOL:
                first = brentq(excess, lo, hi)
                break
        results.append(first)
    return results


def check():
    game = Election()
    ordinary = [stakes(game, 0., 0.), stakes(game, 1., 0.), stakes(game, 1., 1.)]
    np.testing.assert_allclose(ordinary, [4.489036, 5.238680, 6.490482], atol=6e-7)
    finer = Election(order=112)
    for ell in (0., .137, .5, .854, 1.):
        for party in (L,R):
            np.testing.assert_allclose(game.endpoints(party,ell), finer.endpoints(party,ell),
                                       atol=2e-12, rtol=0.)
    # Check state-contingent elections and both sides' deviations independently
    # of the root/citizen-boundary candidates selected by solve().
    print('q0     mass        S-         S+      free repeal    costly repeal')
    cases = [Institutions(mass=0.), Institutions(), Institutions(intercept=.29)]
    expected = [(None,None), (.309302974,.348547012), (.8,.8)]
    for inst, target in zip(cases, expected):
        cut = repeal_cutoffs(game, inst)
        limits = [voluntary_end(game, inst, ell) for ell in (1.,0.)]
        for a,b in zip(cut,target):
            assert (a is None and b is None) or (a is not None and abs(a-b)<1e-7)
        def fmt(x):
            return 'none' if x is None else f'{x:.6f}'
        print(f'{inst.intercept:.2f}   {inst.mass:.2f}', *(f'{fmt(x):>12}' for x in limits+cut))
        for s in sorted(set(np.linspace(0,1,15)) | {.01962,.05238,.2,.3093,.34855,.8,.80001}):
            for guarantee in (False, True):
                best = solve(game, inst, s, guarantee)
                for r in np.linspace(0,1,21):
                    dev = evaluate(game, inst, s, best.ell, r=r, guarantee=guarantee)
                    assert best.right_value + 1e-9 >= dev.right_value
                    assert abs(dev.p-.45)<TOL
                for ell in np.linspace(0,1,101):
                    dev = evaluate(game, inst, s, ell, guarantee=guarantee)
                    assert best.left_value + 1e-9 >= dev.left_value
                    assert abs(dev.p-.45)<TOL
    for s in (.06,.4,.8,.81):
        unprotected = solve(game, cases[1], s)
        installed = unprotected.p*(1-unprotected.q*unprotected.refuse)
        print(f'S={s:.2f} ell={unprotected.ell:.6f} support={unprotected.backing:.2f} '
              f'q={unprotected.q:.2f} L installed={installed:.4f}')
    # When the neutral second-election ballot is fixed, parties reoptimize.
    frozen = Election(responsive=False)
    ft = voluntary_end(frozen,cases[1],0.)
    np.testing.assert_allclose(ft,.06906186,atol=1e-8)
    for label, rule in [('ordinary',game),('fixed',frozen)]:
        sol = solve(rule,cases[1],.06)
        print(f'{label} second ballot at .06: ell={sol.ell:.6f}, refusal={sol.refuse}, '
              f'p={sol.p:.2f}, q={sol.q:.2f}')
    # Institutional slope changes the audience of an identical offer.
    for slope, expected_refusal in [(.4, False), (.6, True)]:
        chosen = solve(game, Institutions(slope=slope), .4)
        assert chosen.refuse == expected_refusal
        np.testing.assert_allclose(chosen.ell, sqrt(2)-1, atol=1e-10)
    # Continuous support for the marginal group in Appendix D.1.
    cutoff_offer = sqrt(1.98)-1
    def smooth_objective(ell):
        threshold = .5-.025*(1+ell)**2
        fraction = np.clip((.4505-threshold)/.001, 0., 1.)
        q = .53+.12*fraction
        surplus = game.branch(L, ell, .4)[1][L]-game.branch(R, 1., .4)[2][L]
        return (1-q)*surplus
    best = smooth_objective(cutoff_offer)
    assert all(smooth_objective(ell) <= best+1e-10 for ell in np.linspace(0,1,251))
    print(f'Uniform marginal types, S=.40: ell={cutoff_offer:.6f}, L installed=.2115')
    return game


def draw_figures(game, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'axes.axisbelow': True})
    palette = ['#c8e4dc', '#e1eaf0', '#79acbd', '#f4d19f', '#e8e1da']
    edge, sea, amber = '#244d65', '#197563', '#9a5319'
    # Exact bounds on each support cell. Draw jumps with the tie on the
    # low-support side; the curves are not interpolated through a jump.
    fig, ax = plt.subplots(figsize=(7.2,5.3))
    fig.subplots_adjust(left=.115,right=.98,top=.97,bottom=.29)
    bottom, top = 2.2, 4.35
    ql, qh = .53,.65
    for a,b, mixed in [(0.,.2,False),(.2,.8,True),(.8,1.,False)]:
        x = np.linspace(a,b,241)
        flat = np.array([stakes(game,t,0.) for t in x])
        full = np.array([stakes(game,t,1.) for t in x])
        if mixed:
            switch = np.array([stakes(game,t,civic_offer(t)) for t in x])
            levels = [np.full_like(x,bottom), ql*flat, ql*switch,
                      qh*switch, qh*full, np.full_like(x,top)]
            colors = [3,1,2,1,0]
        else:
            q = ql if a == 0 else qh
            levels = [np.full_like(x,bottom),q*flat,q*full,np.full_like(x,top)]
            colors = [4,1,0]
        for i, shade in enumerate(colors):
            ax.fill_between(x,levels[i],levels[i+1],facecolor=palette[shade],linewidth=0)
        for y in levels[1:-1]:
            ax.plot(x,y,color=edge,lw=.95)
    for t in (.2,.8):
        ax.axvline(t,color=edge,lw=.9,ls=':')
    ax.axhline(2.4,color='#422d48',ls=(0,(5,3)),lw=1.45)
    ax.text(.985,2.425,'$k=2.40$',ha='right',va='bottom',color='#422d48',fontsize=9)
    ax.set(xlim=(0,1),ylim=(bottom,top),xlabel='Variance of the forecast available in office, $S$',
           ylabel='Cost of refusing a certified defeat, $k$')
    ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks([2.4,2.8,3.2,3.6,4.,4.3])
    handles=[Patch(facecolor=palette[i],edgecolor=edge,linewidth=.55,label=l)
             for i,l in enumerate(['Concedes to full mandate','Concedes at payoff indifference',
                                    'Concedes when citizens withdraw support',
                                    'Refuses despite opposition restraint','Refuses a full mandate'])]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.505,.005),ncol=2,
               frameon=False,fontsize=10.2,columnspacing=1.25,handlelength=1.55)
    meta={'Creator':'verify_citizen_support.py','CreationDate':None,'ModDate':None}
    fig.savefig(destination/'citizen_transfer_regions.pdf',metadata=meta)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2,5.1))
    fig.subplots_adjust(left=.13,right=.98,top=.97,bottom=.31)
    cases = [(Institutions(),sea,'-',r'$q_0=.35$,  $m=.40$'),
             (Institutions(mass=0.),edge,'--',r'$q_0=.35$,  $m=0$'),
             (Institutions(intercept=.29),amber,'-.',r'$q_0=.29$,  $m=.40$')]
    for inst,color,line,label in cases:
        end = voluntary_end(game,inst,0.)
        cuts = sorted({0.,.2,.8,1.,float(end)})
        for lo,hi in zip(cuts,cuts[1:]):
            x = np.linspace(lo+1e-8,hi-1e-8,91)
            y = np.array([repeal_gain(game,inst,s) for s in x])
            ax.plot(x,y,c=color,ls=line,lw=2.)
        for point in sorted(set([float(end),.8])):
            left = repeal_gain(game,inst,point); right=repeal_gain(game,inst,point+1e-8)
            if abs(left-right)>.0001:
                ax.plot([point],[left],'o',mec=color,mfc=color,ms=4.5,zorder=5)
                ax.plot([point],[right],'o',mec=color,mfc='white',ms=5,zorder=6)
    ax.axhline(.01,ls=(0,(2,2)),lw=1.15,c='#353535')
    cut = repeal_cutoffs(game,cases[0][0])[1]
    ax.plot(cut,.01,'o',ms=3.8,c=sea,zorder=7)
    ax.annotate(r'$S_+=.052$',xy=(.052379,-.014),xytext=(.044,-.076),fontsize=9.5,color=edge,
                arrowprops={'arrowstyle':'-','color':edge,'lw':.9})
    ax.annotate(r'$S_E=.349$',xy=(cut,.01),xytext=(.215,.098),fontsize=9.5,color=sea,
                arrowprops={'arrowstyle':'-','color':sea,'lw':.9})
    ax.annotate(r'$S_+=S_E=.80$',xy=(.8,.0863),xytext=(.51,.145),fontsize=9.5,color=amber,
                arrowprops={'arrowstyle':'-','color':amber,'lw':.9})
    ax.set(xlim=(-.005,1.005),ylim=(-.112,.278),xlabel='Variance of the forecast available in office, $S$',
           ylabel='Incumbent payoff gain from repeal')
    ax.set_xticks(np.linspace(0,1,6)); ax.set_yticks([-.10,-.05,0.,.05,.10,.15,.20,.25])
    ax.grid(axis='y',color='#e0e4e6',lw=.55)
    handles=[Line2D([],[],color=c,ls=l,lw=2,label=s) for _,c,l,s in cases]
    handles.append(Line2D([],[],color='#353535',ls=':',lw=1.2,label=r'Repeal cost $c_E=.01$'))
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.53,.02),ncol=2,
               frameon=False,fontsize=10.6,columnspacing=1.8)
    fig.savefig(destination/'transfer_guarantee_repeal.pdf',metadata=meta)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',action='store_true',help='regenerate Figures 2 and 4')
    args = parser.parse_args()
    fitted = check()
    if args.figures:
        draw_figures(fitted,Path(__file__).resolve().parent)
