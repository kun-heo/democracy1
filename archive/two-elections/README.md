# A Two-Election Benchmark

The PDF and its LaTeX source distinguish a single election followed by policy implementation from a game with two elections. The numerical benchmark uses the November 2026 OLG manuscript's parties, voter cohorts, public record, policy menus, quorum selection, and tie rules. Everyone's payoff after term 2 is zero.

Run `python two_election_benchmark.py` with Python 3.11+, NumPy, and SymPy. Keep `solve_equilibrium.py` in the same directory. That file is the OLG manuscript's solver. The benchmark uses its one-term game twice, in reverse order. The replication independently reconstructs world-specific values from the event tree and verifies the campaign dominance result at each term and democratic public state.

The output lists the exact comparisons in Tables 1 and 2 rounded for display. A program code equals twice the zero-based index of the party's mandate plus its official account. Code 0 for the left is the mandate (-.10, -.10) with account 0. Code 2 for the left is (.30, 1.20) with account 0. Code 3 for the right is (.50, .70) with account 1.

Build the note using `latexmk -pdf Two_Election_Benchmark.tex`, or run `pdflatex` until references settle.
