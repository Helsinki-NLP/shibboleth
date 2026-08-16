# Analyses of the manually annotated lists

- `count-annotations.ipynb` unifies the annotations across language groups and produces three CSV files per language groups: `*_overview.csv` (shibboleth types per method), `*_perclass.csv` (shibboleth types per method and variety class) and `*_perlevel.csv` (linguistic level per method and variety class).
- `annotation-analyses.ipynb` creates the visualizations shown in Section §4.2.3 and §5.3.2 of the paper.
- `plot-beeronyms.ipynb` plots the named entity types in the German Jodel data (based on `ne_code_file1.csv`).
- `tvd_manual_annots.py` runs the TVD analyses presented in §4.2.3 and §5.3.2 of the paper.

**TODO:**
- The previous BCMS twitter levels graphs are flawed (used wrong randomized list correspondences), need to update them in the paper.
- The beeronym plot has been redrawn with a different style/font, need to update in the paper.
- `tables/tvd_results.csv` contains the raw data for the TVD analysis, but not the aggregated numbers shown in the paper. Document the aggregation step and make sure the numbers match.
