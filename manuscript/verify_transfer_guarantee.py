"""Reproduce the guarantee and policy-distance comparisons from the Gaussian endpoints."""

from fractions import Fraction as F


# Ordinary-voting endpoint stakes in Appendix D, rounded there to six places.
NULL = F("4.489036")
FLAT = F("5.238680")
FULL = F("6.490482")
P = F(".45")
RHO = F(1)


def stakes(monitoring, revealing):
    return (1 - monitoring) * NULL + monitoring * revealing


def conditional_repeal_gain(monitoring, q, k):
    flat = stakes(monitoring, FLAT)
    full = stakes(monitoring, FULL)
    return q * flat - k - (1 - q) * (full - flat)


def repeal_cutoff(q, k, cost):
    at_zero = conditional_repeal_gain(F(0), q, k)
    at_one = conditional_repeal_gain(F(1), q, k)
    required = cost / (RHO * P)
    if at_one <= required:
        return None
    return (required - at_zero) / (at_one - at_zero)


def policy_distance(monitoring, mandate):
    return 4 + monitoring * (1 + mandate) ** 2 / 4


if __name__ == "__main__":
    for q, k in [(F(".50"), F("2.40")), (F(".80"), F("3.84"))]:
        lower = (k / q - NULL) / (FULL - NULL)
        upper = (k / q - NULL) / (FLAT - NULL)
        assert round(float(lower), 4) == .1554 and round(float(upper), 4) == .4148
        assert conditional_repeal_gain(upper, q, k) < 0  # Limit from above.
        cutoffs = [repeal_cutoff(q, k, cost) for cost in (F(0), F(".01"))]
        reported = [round(float(s), 4) if s is not None else None for s in cutoffs]
        assert reported == ([None, None] if q == F(".50") else [.7121, .7757])
        for cost, cutoff in zip((F(0), F(".01")), cutoffs):
            if cutoff is not None:
                assert upper < cutoff < 1
                assert RHO * P * conditional_repeal_gain(cutoff, q, k) == cost
        for s in map(F, (".50", ".75", ".78", ".80")):
            net = RHO * P * conditional_repeal_gain(s, q, k) - F(".01")
            assert (net > 0) == (q == F(".80") and s >= F(".78"))
        print(f"q={float(q):.2f}, k={float(k):.2f}: S-={float(lower):.4f}, "
              f"S+={float(upper):.4f}, repeal cutoffs={reported}")

    distances = [policy_distance(F(".05"), 1), policy_distance(lower, 1),
                 policy_distance(upper, 0)]
    assert [round(float(d), 3) for d in distances] == [4.050, 4.155, 4.104]
    assert P * (1 - F(".80")) == F(".09")
    print("Expected squared distances:", ", ".join(f"{float(d):.3f}" for d in distances))
