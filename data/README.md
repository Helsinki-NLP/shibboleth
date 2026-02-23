# Datasets used in the experiments

Each folder represents one of the datasets used in the experiments. The `process.sh` script shows:
- how the data is extracted, preprocessed and resampled from the original corpus,
- how the different iterations and folds are created.

It creates the following folder structure:

```
iter_folds
├── iter1
│   ├── dev_iter1_fold1.tsv
|   ..
│   ├── dev_iter1_fold10.tsv
│   ├── train_iter1_fold1.tsv
|   ..
│   └── train_iter1_fold10.tsv
..
├── iter10
│   ├── dev_iter10_fold1.tsv
|   ..
│   ├── dev_iter10_fold10.tsv
│   ├── train_iter10_fold1.tsv
|   ..
│   └── train_iter10_fold10.tsv
└── test.tsv
```

For the datasets used in experiment 1, the `wordlists` or `lexicons` folder shows how to extract the whitelists and blacklists. The `dev` sets are then further annotated with whitelist and blacklist tokens, using the `annotate_with_lexicon.py` script (part of `process.sh`). For example, the sentence *Han købte hende en hund.* (Danish) is annotated as follows:

```
{"label": "da", "white": ["købte"], "black": ["Han", "hund", "en", "hende"]}
```
