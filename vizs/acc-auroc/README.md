# Analyses and visualizations concerning classifier accuracy and non-aggregated shibboleth detection performance

`classif-attribs.ipynb` generates:
- classifier accuracy plots for experiments 1 and 2,
- shibboleth detection performance plots based on attribution score sign for experiment 1 (precision, recall, F1-score),
- shibboleth detection performance plots based on AUROC.

The shibboleth detection performance plots are based on the data files `auroc_data.csv`, `auroc_data+wl.csv`. The accuracy plots are baed on the data file `evals_test_all.csv`.

**TODO:**

- Describe how `auroc_data.csv`, `auroc_data+wl.csv` and `evals_test_all.csv` are generated.
