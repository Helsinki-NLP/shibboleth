# Model training and evaluation pipeline

For each dataset, there is a `train_explain.sh` script that shows the experimental pipeline. It contains the following items:
- A SLURM array job is created to run the 10 iterations * 10 folds in parallel.
- An XLM-R classifier is fine-tuned on the training set.
- The classifier is evaluated on the held-out test set (`eval_test.txt`).
- The four attribution methods (LOO, IG, SHAP, LIME) are computed on the development set and the instance-level predictions stored as `jsonl` files.
- For the language groups of experiment 1, the results are evaluated with respect to the ground truth (blacklists only, or whitelists and blacklists, see `METHOD_results.txt` and `METHOD_results+wl.txt` respectively).

The `train_explain_spec.sh` uses a language-group-specific BERT model as basis (instead of XLM-R) and only performs the following steps:
- A SLURM array job is created to run 1 iteration * 10 folds in parallel.
- A language-group-specific BERT model is fine-tuned on the training set.
- The classifier is evaluated on the held-out test set (`eval_test.txt`).

The `train_explain.sh` script produces the following folder structure:

```
├── iter1
│   ├── model_fold1
|   |   ├── eval_test.txt
|   |   ├── ig_token.jsonl
|   |   ├── lime_token.jsonl
|   |   ├── loo_token.jsonl
|   |   └── shap_token.jsonl
|   ..
│   └── model_fold10
|   |   ├── eval_test.txt
|       ├── ig_token.jsonl
|       ├── lime_token.jsonl
|       ├── loo_token.jsonl
|       └── shap_token.jsonl
..
└── iter10
    ├── model_fold1
    |   ├── eval_test.txt
    |   ├── ig_token.jsonl
    |   ├── lime_token.jsonl
    |   ├── loo_token.jsonl
    |   └── shap_token.jsonl
    ..
    └── model_fold10
        ├── eval_test.txt
        ├── ig_token.jsonl
        ├── lime_token.jsonl
        ├── loo_token.jsonl
        └── shap_token.jsonl
```

Below is an example of the data structure given in the `.jsonl` files (one line per classification instance, pretty-printed here for clarity):

```
{
  "correct": true,
  "pred_label": "sv",
  "gold_label": "sv",
  "tokens": ["Jag", "lade", "på", "och", "ringde", "henne", "igen."],
  "attribs": [0.18516787886619568, 0.6260251402854919, 0.24626116454601288, 0.1792948693037033, 0.4029402583837509, 0.2541979253292084, 0.5724918246269226]
}
```

- The fields `pred_label`, `gold_label` and `correct` refer to the variety labels produced by the classifier.
- `tokens` provides a list of the tokens in the current instance, and `attribs` provides the attribution scores corresponding to each token.

The fine-tuned classifiers, the datasets and the predictions are not made available here due to space limitations.
