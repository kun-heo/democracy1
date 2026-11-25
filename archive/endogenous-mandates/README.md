# Endogenous policy mandates and the transfer of power

The note gives a continuous-mandate equilibrium at two elections and exhausts the stationary equilibria at the three binary-menu calibrations in the main paper. The equilibrium probabilities and comparisons are exact. Decimal values printed for readability are rounded.

## Replicate

Use Python 3.10 or later. The programs use the standard library.

```bash
python continuous_mandates.py
python stationary_equilibria.py
python crosscheck_pure.py
```

The first command derives the exact coefficients and cutoff values for the right-incumbent continuous game. The last term permits both parties to choose any weight on the same interval as in the current term. The code checks the signs of piecewise polynomials using their endpoints and interior extrema.

The second command exhausts stationary continuation values one posterior at a time, starting with fully informative records. It encloses all feasible values, including those obtainable by mixed campaigns, and checks every remaining profile by its exact Bellman values. All three calibrations must finish with zero unresolved branches. Output gives takeover probabilities by initial incumbent and the cutoffs for a temporary monitoring change.

The third command independently enumerates all pure campaigns and on-path political decisions at each distinct continuation found by the second command. The economically equivalent labels at the unresolved-intercept, zero-slope record can be interchanged freely.

The three companion modules `bayesian_horizons.py`, `verify_bayesian_mandates.py`, and `verify_horizons.py` reproduce the primitives and checks of the Bayesian manuscript. They are included for portability. To build the note, run `pdflatex Endogenous_Policy_Mandates.tex` twice. 
