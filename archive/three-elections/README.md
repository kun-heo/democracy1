# A Three-Election Benchmark

The game has at most three elections, with two overlapping generations of voters in each democratic term. The voter horizon, policy menus, monitoring process, enforcement convention and numerical parameters are those of the OLG manuscript and the two-election benchmark. All payoffs after term 3 are zero.

Run `python three_election_benchmark.py` with Python 3.11+, NumPy, and SymPy. Keep the two imported Python files in the same directory. They are unchanged from the two-election replication. Build the note with `latexmk -pdf Three_Election_Benchmark.tex`, or run `pdflatex` until cross-references settle.

The script solves terms 3 and 2 by exact backward induction, reconstructs the first term's campaign matrices under those continuations, and checks the independent mixed equilibrium and all unilateral deviations using fractions. Independent event-tree reconstruction checks all values of the deterministic components of the first-term mixture. Program codes equal twice the zero-based mandate index plus the official account. The policies attached to the four mixed-equilibrium programs appear in Table 2.

The return calculations condition on losing the first certificate and impose a concession at that node. In the mixed campaign the calculation conditions the distribution of public offers on the defeat. Next-election victory, next-term control, a first return in term 3, actual term-3 control, and any return by the deadline are reported separately.
