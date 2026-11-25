# Bayesian mandates with overlapping generations of voters

Run `python verify_horizons.py` with Python 3.10 or later. The verification uses the standard library and exact rational arithmetic. It prints cutoffs, refusal gains, return probabilities, and eventual takeover probabilities. It also checks the inequalities on either side of all first-term cutoffs and at ties.

- `Bayesian_Mandates_OLG_Horizons.tex` is the editable text of the four-page note. It can be compiled with pdfLaTeX.
- `verify_bayesian_mandates.py` supplies the two-election benchmark primitives and its original verifier.
- `bayesian_horizons.py` computes the public Bayesian transition law, voters’ and parties’ continuation values, campaigns, enforcement and transfer decisions.
- `verify_horizons.py` contains the finite backward induction, explicit stationary equilibrium strategies and deviation checks, electoral counterfactual, and absorption calculation.

The first-election cutoff experiment changes current monitoring and uses .05 at every later election. The permanent-monitoring experiment uses either .05 or .14 at every election and solves the stationary equilibrium separately in each case. In the counterfactual the neutral certificate is assigned to R only at the next election. All later elections retain ordinary voting.
