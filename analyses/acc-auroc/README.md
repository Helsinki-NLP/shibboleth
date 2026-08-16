# Analyses and visualizations concerning classifier accuracy and non-aggregated shibboleth detection performance

`classif-attribs.ipynb` generates:
- classifier accuracy plots for experiments 1 and 2,
- shibboleth detection performance plots based on attribution score sign for experiment 1 (precision, recall, F1-score),
- shibboleth detection performance plots based on AUROC.

The shibboleth detection performance plots are based on the data files `auroc_data.csv`, `auroc_data+wl.csv`, which are created using `parse_gt_results.py`. The accuracy plots are based on the data file `evals_test_all.csv`.

**TODO:**
- `parse_gt_results.py` generates `auroc_data+wl.csv`, but it is not clear to me how `auroc_data.csv` was generated (different script or just different parameter setting).
- Describe how `evals_test_all.csv` is generated.
- Do we have a summary file of the accuracies with XLM-R vs lang-spec-BERTs, i.e. the numbers used to generate Table A.1 in the paper?
