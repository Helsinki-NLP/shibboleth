# Shibboleth Detection with Explainable AI Methods

### todo
practical cleaning steps for the repo worth doing at some point:
 - move unused scripts to e.g. `scripts/archival/` or `scripts/old/` or `scripts/unused/`
    - moved to `scripts/unused/` - please check that nothing important has been moved
	- there are also some random scripts and files in `other_stuff` - please check if there's anything worth keeping
 - move `manual_eval/*ipynb` to `vizs/` (maybe create subdirectories in `vizs/` to differentiate between the different sections we're considering)
   - renamed `vizs` to `analyses` and created subdirectories roughly corresponding to the paper sections
   - the notebooks, CSVs and PDFS originally in `manual_eval` were moved to `analyses/shib-eval-contents`
 - we have `*_overview.csv` files under `lists/`, but `eval_test_all.csv` and `auroc_data*` stuff directly under the main repository
   - moved everything into `analyses/` (the `*_overview.csv` files to `analyses/reflists`, the AUROC stuff to `analyses/acc_auroc`)
 - could we have all the generated plots in a directory?
   - they're now in the different subdirectories within `analyses` - is that good enough? Feel free to move them around if you had another system in mind :)
 - write actual instructions in `README.md`
   - the READMEs in the various subdirectories look fairly ok to me, but the global one (this file) needs to be updated
