# Political Realignment and the Transfer of Power in an Unpredictable World

The editable manuscript is `Realignment_and_the_Transfer_of_Power.tex`. The source files in this directory include the recurring-election analysis in Section 6.5. The source ZIP contains the preceding version. Figures 2, 4, and 5 use the supplied vector PDFs. The other figures are drawn in LaTeX. Compile the document with pdfLaTeX at least twice, or run the following command from this directory.

```bash
latexmk -pdf Realignment_and_the_Transfer_of_Power.tex
```

## Citizen support for refusal in two elections

Run the following from this directory with Python 3, NumPy, SciPy, and Matplotlib.

```bash
python verify_citizen_support.py --figures
```

This recomputes both elections, the two cohorts' ballots and support for refusal, the party campaigns with and without a conditional transfer guarantee, and the first three rows of Table 3. It writes Figure 2 to `citizen_transfer_regions.pdf` and Figure 4 to `transfer_guarantee_repeal.pdf`. Omit `--figures` to run the numerical checks alone. Section 6.2.1 gives the distribution of citizen types and Appendix D.1 derives the continuous-mandate best responses.

## Citizen support with recurring elections

Run `python verify_citizen_mpe.py` with NumPy and SciPy to reproduce Section 6.5 and Appendix D.2. This checks the known-technology stationary branch with rational arithmetic and solves the finite Kalman approximation with indefinite elections, binary mandates, both incumbent identities and overlapping citizen cohorts. It checks parties' deviations and recovers values and absorption probabilities independently from the full Markov chain.

Use `--figure` with Matplotlib to regenerate Figure 5 (`citizen_stationary_transfer.pdf`). Use `--refinement` for the wider posterior grid. Each displayed threshold for a current shock holds the future stationary continuation fixed. The known-technology calculation also solves for the end of a branch when the change in monitoring is permanent.

## Transfer guarantees

Run `python verify_transfer_guarantee.py` to reproduce the fixed-enforcement repeal cutoffs (the last two rows of Table 3) and expected squared policy distances from the ordinary-voting Gaussian endpoint payoffs printed in Appendix D. It requires only the Python standard library.

## Transfer with weak valence

Run `python verify_weak_valence.py` with NumPy and SciPy to reproduce Section 6.3 and Appendix D's weak-valence comparison. At the specified first-term offers, the script integrates over the Gaussian posteriors, solves the final election, and checks the first count, the blocs' endorsement decisions, and the incumbent's transfer choice for each voting rule.

## Earlier specifications

The notes below describe the earlier specifications covered by the other scripts and the standalone figure in this archive.

Sections 2–5 of the manuscript and appendices A–B contain the symbolic setup, equilibrium propositions, and proofs. Section 6 and appendices C–E contain the worked calculations. They use exact rational arithmetic in Python 3.10 or later. Run the following commands from this directory.

```bash
python verify_draft.py
python verify_symbolic_family.py
python continuous_mandates.py
python stationary_equilibria.py
python crosscheck_pure.py
```

- `verify_draft.py` checks the last-election refusal formula, the constant continuation premium against a flat mandate, and the two-election experiment in which only the next neutral certificate is fixed.
- `verify_symbolic_family.py` checks the strict equilibrium inequalities at the asymmetric-prior illustration, including all continuous first- and last-election offers, and verifies that the closed-form mandate binds the transfer constraint. Appendix D proves why these strict inequalities persist in an open neighborhood.
- `continuous_mandates.py` solves the final election at all 15 nonempty posterior supports, integrates the observation likelihood, and verifies the full continuous-mandate equilibrium, the polynomial coefficients and the cutoffs for the first election. It bounds deviations on the entire interval.
- `stationary_equilibria.py` exhausts equilibria under the binary mandate menu at the three permanent-monitoring specifications. It includes independent mixing over distinct offers, evaluates future takeover probabilities, checks all deviations, and prints the thresholds for transitory monitoring changes.
- `crosscheck_pure.py` independently enumerates the pure campaign and political outcomes at each surviving stationary continuation.
- `verify_bayesian_mandates.py`, `bayesian_horizons.py`, and `verify_horizons.py` supply the shared preferences, posterior and value calculations, stationary profile evaluation, and absorption probabilities. The horizon-specific legacy routines in these shared modules are not needed as separate entry points for this manuscript.

To regenerate the figure, install NumPy and Matplotlib and run `python make_figure.py`. The drawing uses the analytically verified equilibrium curves. The proof checks require no external Python packages.

`Gallup_Korea_impeachment.csv` contains the three selected repeated cross-sections used in the discussion, their source URLs, weighted percentages, and unweighted sample counts. These observational numbers are separate from the probabilities generated by the model.
