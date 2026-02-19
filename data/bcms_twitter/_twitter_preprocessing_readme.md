# Preprocessing BCMS Twitter data

The starting point for the experiments presented here is the BCMS Twitter corpus used in the VarDial2024 Shared Task on DSL-ML: 

Miletić, A., & Miletić, F. (2024). A multilabel dataset for distinguishing Bosnian, Croatian, Montenegrin, and Serbian [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10998042

The corpus was taken through several preprocessing steps, described below for documentation reasons. However, since the dataset was **downsampled randomly**, attempts to reproduce the results should start from the preprocessed files given here.

## Preprocessing steps
1. `1_sent_seg.py': Sentence segmentation of instances in the original dataset.
2.  `2_downsample.py': Downsampling the largest class (Serbian) to the second-largest (Croatian). Done on all corpus splits.
3.  `3_reformat.py': Reformating and creating single label version of dev and test splits.
4.  `4_clean_tokenize.py': Tokenizing instances.

Sentence segmentation and tokenization were done using the CLASSLA python library for BCMS (https://pypi.org/project/classla/).




