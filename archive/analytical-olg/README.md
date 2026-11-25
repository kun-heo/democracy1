# Political Realignment and the Transfer of Power

## Analytical equilibria with overlapping generations of voters

The note solves two-election and three-election games by backward induction and constructs stationary equilibria for an infinite horizon. A voter lives for two consecutive terms. Parties and enforcement blocs value later terms throughout the political horizon.

- Sections 2 through 4 and Appendix A use the full mandates, official causal accounts, public technology learning, and electorate of the accompanying draft. The finite campaign correspondence and a stationary equilibrium family hold on the specified two-dimensional parameter region with the stated enforcement and tie conventions.
- Sections 5 through 9 solve the fixed-mandate, known-technology model. They include endogenous OLG ballots and enforcement, and give the symmetric thresholds for arbitrary discount and enforcement parameters.

With Python 3.11 or later, NumPy, and SymPy, run the following from the directory containing this file.

    python check_identities.py
    python full_game.py

The first command independently checks the theoretical Bellman identities and closed-form results. The second uses exact rational arithmetic, polynomial identities, and interval sign proofs. Its printed output is included as full_game_formulas.txt. The inherited module solve_equilibrium.py supplies the original payoff and transition primitives. The two complete on-path stationary specifications are provided in stationary_paths.json. The new proof reconstructs off-path play and party deviations and does not require running the inherited equilibrium search.

Compile the note twice with pdfLaTeX.

    pdflatex Analytical_OLG_Equilibria.tex
    pdflatex Analytical_OLG_Equilibria.tex

Program codes 0 and 1 combine a party's first mandate with accounts 0 and 1. Codes 2 and 3 combine its second mandate with those accounts. In full_game_formulas.txt, T0 and T1 denote the post-defeat incentives of an L and an R incumbent respectively. The two digits after each label give the L and R program codes in that order.
