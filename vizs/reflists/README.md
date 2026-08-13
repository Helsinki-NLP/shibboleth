# Analyses and visualizations involving the truelists/falselists for experiment 1

`list-agg.ipynb` produces graphs that relate the proportion of falselisted items to the attribution method and to the $p$ and $m$ parameters. For some of the analyses, it relies on word frequency lists, which are compiled with `compute_freq_index.ipynb` and stored in `freqs`. Some analyses are based on the `*_lists_overview.csv` files, which are produced using `make_overview_exp1.sh` and `parse_lists.py`.

`reflist-quality.ipynb` assesses the quality of the truelists and falselists by comparing them with the manual annotations done independently.

**TODO:**
- I tried to reconstruct `make_overview_exp1.sh` to regenerate the `*_lists_overview.csv` files, but didn't manage to reproduce the files.