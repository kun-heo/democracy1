"""Three-election and stationary solutions for the public Bayesian mandate game.

Requires verify_bayesian_mandates.py beside this file. Only standard-library
arithmetic is used. All probabilities, values and optimality tests are exact.
"""
from collections import defaultdict
from fractions import Fraction as F
from functools import cache
from typing import NamedTuple

import verify_bayesian_mandates as base
from verify_bayesian_mandates import (
    LEFT, RIGHT, BASE, SIGN, PRIOR, WORLDS, ZERO, ONE, DISCOUNT,
    BLOC_COST, ATTEMPT_COST, FINAL_MONITORING, OFFERS, VALENCE, VALENCE_MASS,
    means, flow, certificates, election_payoffs, interpolate, ordered,
    zero_of_affine, insert_root, intervals,
)

PLAYERS = (LEFT, RIGHT)
AUT, DEM = "A", "D"
FIXED_CERT = (False, True, True)
Z = (ZERO,)*6
RECORDS = tuple(sorted(((0,), (1,), (2,), (3,), (0,2), (1,3), PRIOR), key=lambda h: (len(h),h)))
STATES = tuple((reg,j,h) for h in RECORDS for reg in (AUT,DEM) for j in PLAYERS)
VALUE_ZERO = {state:Z for state in STATES}


@cache
def posterior(record, instrument, state):
    centers = {i:instrument*(WORLDS[i][0]+WORLDS[i][1]*state) for i in record}
    width = base.ERROR_RADIUS
    endpoints = sorted({v+s*width for v in centers.values() for s in (-1, 1)})
    mass = defaultdict(F)
    for lo,hi in zip(endpoints,endpoints[1:]):
        mid=(lo+hi)/2
        compatible=tuple(i for i in record if abs(mid-centers[i])<width)
        if compatible:
            mass[compatible]+=(hi-lo)*len(compatible)/(len(record)*2*width)
    assert sum(mass.values())==1 and set(mass)<=set(RECORDS)
    return dict(mass)


@cache
def kernel(record, government, weight, monitoring):
    a,b=means(record)
    result=defaultdict(F)
    for state in (-1,1):
        for m,prob in ((0,1-monitoring),(state,monitoring)):
            if not prob:
                continue
            x=BASE[government]+weight*(a+b*m)
            for nxt,p in posterior(record,x,state).items():
                result[nxt]+=prob*p/2
    assert sum(result.values())==1
    return dict(result)


class Decision(NamedTuple):
    refuse: bool
    ballots: tuple  # R certificate at each of the three valence shocks
    supporters: tuple  # enforcers for each of the two quorums
    success: F
    right: F
    gain: F


def play_after_certificate(inc, branches, supporters, refuse, attempt_cost=ATTEMPT_COST):
    win,lawful,entrench=branches
    success=sum(F(1,2) for group in supporters if not group)
    loss=list(lawful)
    if refuse:
        loss=list(interpolate(lawful,entrench,success))
        loss[inc]-=attempt_cost
        for group in supporters:
            for bloc in group:
                loss[2+bloc]-=BLOC_COST/2
    return (win,tuple(loss)) if inc==LEFT else (tuple(loss),win)


def enforcement(inc,branches):
    _,c,a=branches
    priority=(1-inc,inc)
    willing=tuple(bloc for bloc in priority if c[2+bloc]-a[2+bloc]>=BLOC_COST)
    return tuple(willing[:q] if len(willing)>=q else () for q in (1,2))


def campaign_equilibria(payoffs):
    """Find a certified equilibrium; flag states having potential payoff multiplicity."""
    pure=[]
    for l,r in OFFERS:
        if payoffs[l,r][0]>=payoffs[1-l,r][0] and payoffs[l,r][1]>=payoffs[l,1-r][1]:
            pure.append((l,r))
    dl=[payoffs[1,r][0]-payoffs[0,r][0] for r in (0,1)]
    dr=[payoffs[l,1][1]-payoffs[l,0][1] for l in (0,1)]
    mixed=None
    if dl[0]*dl[1]<0 and dr[0]*dr[1]<0:
        p_left=-dr[0]/(dr[1]-dr[0]);p_right=-dl[0]/(dl[1]-dl[0])
        mixed=tuple((p_left if l else 1-p_left)*(p_right if r else 1-p_right) for l,r in OFFERS)
    unique_values=all(v==next(iter(payoffs.values())) for v in payoffs.values())
    # Strict dominance followed by a strict best response gives uniqueness even against mixtures.
    dominant=any(min(v)>0 or max(v)<0 for v in (dl,dr))
    unique=unique_values or (dominant and len(pure)==1) or (not pure and mixed is not None)
    if pure:
        choice=next((p for p in pure if p==(1,1)),pure[0])
        strategy=tuple(F(p==choice) for p in OFFERS)
    else:
        assert mixed is not None
        strategy=mixed
    return strategy,unique


class Stage:
    def __init__(self, future, fixed_vote=False, attempt_cost=ATTEMPT_COST):
        self.attempt_cost = attempt_cost
        self.future=future
        self.fixed_vote=fixed_vote
        self.endpoints={}

    def installed(self, regime, j, h, weight, s):
        key=(regime,j,h,weight,s)
        if key not in self.endpoints:
            current=flow(j,weight,h,s)
            probs=kernel(h,j,weight,s)
            following=tuple(sum(p*self.future[regime,j,h1][i] for h1,p in probs.items())
                            for i in (0,1,2,3,5))+(ZERO,)
            self.endpoints[key]=tuple(x+DISCOUNT*y for x,y in zip(current,following))
        return self.endpoints[key]

    def branches(self,inc,h,pair,s):
        return tuple(self.installed(reg,j,h,SIGN[j]*pair[j],s)
                     for reg,j in ((DEM,inc),(DEM,1-inc),(AUT,inc)))

    def contest(self,inc,h,pair,s, template=None):
        b=self.branches(inc,h,pair,s)
        if template is None:
            supporters=enforcement(inc,b)
            q=sum(F(1,2) for group in supporters if not group)
            gain=q*(b[2][inc]-b[1][inc])-self.attempt_cost
            refuse=gain>0
        else:
            supporters,refuse=template.supporters,template.refuse
            q=template.success
            gain=q*(b[2][inc]-b[1][inc])-self.attempt_cost
        left,right=play_after_certificate(inc,b,supporters,refuse,self.attempt_cost)
        if template is not None:
            ballots=template.ballots
        else:
            ballots=FIXED_CERT if self.fixed_vote else certificates(left,right)
        vals,pr=election_payoffs(left,right,ballots)
        return vals,Decision(refuse,ballots,supporters,q,pr,gain)

    def solve(self,s=FINAL_MONITORING):
        values,profile,unique={}, {}, {}
        for h in RECORDS:
            for inc in PLAYERS:
                values[AUT,inc,h]=self.installed(AUT,inc,h,2*SIGN[inc],s)
                pairs={p:self.contest(inc,h,p,s) for p in OFFERS}
                vals={p:v for p,(v,_) in pairs.items()}
                strat,uniq=campaign_equilibria(vals)
                profile[inc,h]=(strat,{p:d for p,(_,d) in pairs.items()})
                unique[inc,h]=uniq
                values[DEM,inc,h]=tuple(sum(w*vals[p][i] for p,w in zip(OFFERS,strat)) for i in range(6))
        return values,profile,unique


def pure_pair(mix):
    return next((p for p,v in zip(OFFERS,mix) if v==1),None)


def kernel_and_reward(profile,s,attempt_cost=ATTEMPT_COST):
    rows={}; rewards={}
    for reg,inc,h in STATES:
        nxt=defaultdict(F);reward=[ZERO]*6
        def put(gov,weight,regime,mass):
            if not mass:return
            v=flow(gov,weight,h,s)
            for i,x in enumerate(v):reward[i]+=mass*x
            for h1,p in kernel(h,gov,weight,s).items():nxt[regime,gov,h1]+=mass*p
        if reg==AUT:
            put(inc,2*SIGN[inc],AUT,ONE)
        else:
            strat,decisions=profile[inc,h]
            for pair,m in zip(OFFERS,strat):
                if not m:continue
                d=decisions[pair]
                win=d.right if inc==RIGHT else 1-d.right
                put(inc,SIGN[inc]*pair[inc],DEM,m*win)
                loss=m*(1-win)
                success=d.success if d.refuse else ZERO
                put(inc,SIGN[inc]*pair[inc],AUT,loss*success)
                put(1-inc,SIGN[1-inc]*pair[1-inc],DEM,loss*(1-success))
                if d.refuse:
                    reward[inc]-=loss*attempt_cost
                    for group in d.supporters:
                        for bloc in group:reward[2+bloc]-=loss*BLOC_COST/2
                val=m*sum(v*p for v,p,right in zip(VALENCE,VALENCE_MASS,d.ballots) if right)
                reward[4]+=val;reward[5]+=val
        rows[reg,inc,h]=dict(nxt);rewards[reg,inc,h]=tuple(reward)
        assert sum(nxt.values())==1
    return rows,rewards


def linear_solve(matrix, rhs):
    A=[list(row)+list(y) for row,y in zip(matrix,rhs)]
    n=len(A)
    for k in range(n):
        pivot=next(i for i in range(k,n) if A[i][k])
        A[k],A[pivot]=A[pivot],A[k]
        scale=A[k][k];A[k]=[x/scale for x in A[k]]
        for i in range(n):
            if i!=k and A[i][k]:
                c=A[i][k];A[i]=[v-c*w for v,w in zip(A[i],A[k])]
    return [row[n:] for row in A]


def evaluate(profile,s=FINAL_MONITORING,attempt_cost=ATTEMPT_COST):
    rows,r=kernel_and_reward(profile,s,attempt_cost)
    vals={}
    # Posteriors never expand. Each diagonal block contains four political states.
    for h in RECORDS:
        states=[(reg,i,h) for reg in (AUT,DEM) for i in PLAYERS]
        matrix=[[F(z==zp)-DISCOUNT*rows[z].get(zp,ZERO) for zp in states] for z in states]
        rhs=[[r[z][i]+DISCOUNT*sum(p*vals[zp][i] for zp,p in rows[z].items() if zp[-1]!=h)
              for i in range(4)] for z in states]
        for z,v in zip(states,linear_solve(matrix,rhs)):
            vals[z]=tuple(v)+(r[z][4]+DISCOUNT*sum(p*r[zp][5] for zp,p in rows[z].items()),r[z][5])
    return vals


def prior_cells(stage,inc):
    """Partition all comparisons at the initial prior as current monitoring varies."""
    h=PRIOR
    cuts={ZERO,ONE}
    for pair in OFFERS:
        lo=stage.branches(inc,h,pair,ZERO);hi=stage.branches(inc,h,pair,ONE)
        for k in (2,3):
            insert_root(cuts,zero_of_affine(lo[1][k]-lo[2][k]-BLOC_COST,hi[1][k]-hi[2][k]-BLOC_COST))
    for a,b in intervals(cuts):
        for pair in OFFERS:
            q=stage.contest(inc,h,pair,(a+b)/2)[1].success
            def gain(s):
                _,c,v=stage.branches(inc,h,pair,s)
                return q*(v[inc]-c[inc])-stage.attempt_cost
            insert_root(cuts,zero_of_affine(gain(ZERO),gain(ONE)),a,b)
    for a,b in intervals(cuts):
        for pair in OFFERS:
            d=stage.contest(inc,h,pair,(a+b)/2)[1]
            l0,r0=play_after_certificate(inc,stage.branches(inc,h,pair,ZERO),d.supporters,d.refuse,stage.attempt_cost)
            l1,r1=play_after_certificate(inc,stage.branches(inc,h,pair,ONE),d.supporters,d.refuse,stage.attempt_cost)
            for age in (4,5):
                for val in VALENCE:
                    insert_root(cuts,zero_of_affine(r0[age]-l0[age]+val,r1[age]-l1[age]+val),a,b)
    for a,b in intervals(cuts):
        mid=(a+b)/2
        v0={};v1={}
        for pair in OFFERS:
            d=stage.contest(inc,h,pair,mid)[1]
            v0[pair]=stage.contest(inc,h,pair,ZERO,d)[0];v1[pair]=stage.contest(inc,h,pair,ONE,d)[0]
        for i in PLAYERS:
            for j in (0,1):
                on,off=ordered(i,1,j),ordered(i,0,j)
                insert_root(cuts,zero_of_affine(v0[on][i]-v0[off][i],v1[on][i]-v1[off][i]),a,b)
    return cuts


def initial(stage,inc,s):
    opts={p:stage.contest(inc,PRIOR,p,s) for p in OFFERS}
    v={p:x for p,(x,_) in opts.items()}
    m,uniq=campaign_equilibria(v)
    return m,{p:d for p,(_,d) in opts.items()},uniq


