# Model training and evaluation pipeline

For each dataset, there is a `train_explain.sh` script that shows the experimental pipeline. It contains the following items:
- A SLURM array job is created to run the 10 iterations * 10 folds in parallel.
- An XLM-R classifier is fine-tuned on the training set.
- The classifier is evaluated on the held-out test set.
- The four attribution methods (LOO, IG, SHAP, LIME) are computed on the development set and the instance-level predictions stored as `jsonl` files.

The `train_explain.sh` script produces the following folder structure:

```
├── iter1
│   ├── model_fold1
|   |   ├── ig_token.jsonl
|   |   ├── lime_token.jsonl
|   |   ├── loo_token.jsonl
|   |   └── shap_token.jsonl
|   ..
│   └── model_fold10
|       ├── ig_token.jsonl
|       ├── lime_token.jsonl
|       ├── loo_token.jsonl
|       └── shap_token.jsonl
..
└── iter10
    ├── model_fold1
    |   ├── ig_token.jsonl
    |   ├── lime_token.jsonl
    |   ├── loo_token.jsonl
    |   └── shap_token.jsonl
    ..
    └── model_fold10
        ├── ig_token.jsonl
        ├── lime_token.jsonl
        ├── loo_token.jsonl
        └── shap_token.jsonl
```

The fine-tuned classifiers, the datasets and the predictions are not made available here due to space limitations.
