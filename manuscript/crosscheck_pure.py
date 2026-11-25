"""Independent enumeration of pure on-path profiles for each continuation."""
from itertools import product
from fractions import Fraction as F
from stationary_equilibria import ACTORS, P, block_data, classify, explore, m


def direct(block):
    """Solve party and old-voter values for every pure on-path pair of campaigns, neutral ballots and transfer decisions. Check the full branch response at the implied values."""
    per = []
    for j in P:
        per.append([(pair, mode, vs) for pair in m.OFFERS for mode, (y, vs) in block[j, pair][1].items()])
    kept = set()
    for a, b in product(*per):
        q = (a, b)
        exact = {}
        for actor in ACTORS:
            t, v = (a[2][actor], b[2][actor])
            den = (1 - t.l) * (1 - v.r) - t.r * v.l
            low = (t.c * (1 - v.r) + t.r * v.c) / den
            high = (v.c * (1 - t.l) + v.l * t.c) / den
            exact[actor] = [(low, low), (high, high)]
        current = {}
        good = True
        for (inc, pair), (T, opts) in block.items():
            r = int(T.range(exact[inc])[0] > 0)
            n = next((n for (ref, n), (yy, vs) in opts.items() if ref == r and (yy is None or bool(n) == (yy.range(exact[5])[0] > 0))), None)
            if n is None:
                good = False
                break
            current[inc, pair] = ((r, n), opts[r, n][1])
            if q[inc][0] == pair and q[inc][1] != (r, n):
                good = False
                break
        if not good:
            continue
        for inc, (pair, mo, vs) in enumerate(q):
            for actor in P:
                alt = tuple((1 - x if j == actor else x for j, x in enumerate(pair)))
                if current[inc, alt][1][actor].range(exact[actor])[0] > exact[actor][inc][0]:
                    good = False
        if good:
            kept.add(tuple(((p, mo) for p, mo, _ in q)))
    return kept


if __name__ == "__main__":
    for s, k in [(F('.05'), F('2.55')), (F('.14'), F('2.55')), (F('.05'), F('2.40'))]:
        base, _ = classify(s, k, False)
        for values, pr in base:
            for h in m.RECORDS:
                if h == (0, 2):
                    continue
                block = block_data(h, s, k, values)
                got = direct(block)
                pure, others, _ = explore(block, h)
                assert got == set((x for x, force in pure)), (s, k, h, got, pure)
        print('Independent pure enumeration agrees', s, k, flush=True)
