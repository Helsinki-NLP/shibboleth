Explainability experiments - April 2025

# Model training

## Training scripts 

Location: /scratch/project_2005047/explainability/scripts

The ones used for BCMS:
- train_setimes_raw.py
- train_setimes_tok.py

Both expect the most recent input format we agreed on:
[label]\t[raw instance]\t[pretokenized instance]

train_setimes_raw.py reads the second column of the input
train_setimes_tok.py reads the third one

**NB**: When/if applying the scripts to other languages/corpora,
make sure to adapt the label mapping when reading in 
the training and the validation datasets.

For the other languages, I (Yves) used the `train.py` script, which I adapted to the input format above. It does the same as Aleksandra's scripts, but the raw/tok distinction is a parameter (and label mapping is automatic). I also shuffled the training data for Estonian and Scandinavian (but maybe HF does that by default anyway?).

**Note:** This breaks the compatibility of the `train.py` script with the data format used for the non-standard experiments, but it should be fairly straightforward to adapt.


## Batch scripts

Location: /scratch/project_2005047/explainability/models/bcms

The ones used for BCMS with the training scripts listed above:
- tr_set_bertic_raw.sh
- tr_set_bertic_tok.sh


## Script to select the best checkpoint

Location: /scratch/project_2005047/explainability/scripts/get_best_checkpoint.py

Argument: model folder.
 
Creates a new folder named "best" with the best performing model checkpoint.

## Models
Location: /scratch/project_2005047/explainability/models

There is one folder per language/macrolanguage/language group.

Naming convention: model_[corpus]_[model_name]_[raw input|pretokenized input]

- BCMS: `models/bcms/model_setimes_bertic_[raw|tok]`
- Estonian/Võro: `models/estonian_voro/model_ev_[raw|tok]`
- Scandinavian: `models/scandinavian/model_slide_[raw|tok]`

# Leave-one-out experiment

## Adapted script to run the experiment

Location: /scratch/project_2005047/explainability/scripts/loo_correct_masking_words_instance.py

For each evaluated instance, outputs all wordforms and their probability diff scores in a json format.

**NB**: Currently, the label mapping is inferred from the test file - this only works if the set of labels is identical between the training and test set!

## Batch scripts

- BCMS: `explainability/models/bcms/loo_btc_st_[r|t].sh`
- Estonian/Võro: `explainability/models/estonian_voro/loo_[tok|raw].sh`
- Scandinavian: `explainability/models/scandinavian/loo_[tok|raw].sh`

## Results in .jsonl format

- BCMS: `models/bcms/model_setimes_bertic_[raw|tok]/expl_correct_msk_w_set.json`
- Estonian/Võro: `models/estonian_voro/model_ev_[raw|tok]/loo_corr_msk_w.json`
- Scandinavian: `models/scandinavian/model_slide_[raw|tok]/loo_corr_msk_w.json`

# Evaluation of predicted shibboleths with ground truth lexicon data

## Script

`scripts/eval_instances_groundtruth.py`

## Batch scripts and result logs

- BCMS: `explainability/models/bcms/eval_loo_gt.sh` - `explainability/models/bcms/loo_eval.txt`
- Estonian/Võro: not done yet
- Scandinavian: `explainability/models/scandinavian/eval.sh` - explainability/models/scandinavian/loo_eval.txt`
