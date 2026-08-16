# Analyses and visualizations involving the truelists/falselists for experiment 1

`list-agg.ipynb` produces graphs that relate the proportion of falselisted items to the attribution method and to the $p$ and $m$ parameters. For some of the analyses, it relies on word frequency lists, which are compiled with `compute_freq_index.ipynb` and stored in `freqs`. Some analyses are based on the `*_lists_overview.csv` files, which are produced using `make_overview_exp1.sh` and `parse_lists.py`.

`list-agg.ipynb` also contains code to run the OLS experiment (Table 2 / §4.2.2) and the Spearman correlations (Table 3 /B §4.2.2).

**TODO:**
- I tried to reconstruct `make_overview_exp1.sh` to regenerate the `*_lists_overview.csv` files, but didn't manage to reproduce the files.
- The OLS model in the notebook gives different results than those reported in the paper. Is there another script that was used for the final OLS model?
