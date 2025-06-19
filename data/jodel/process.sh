module load python-data

SCRIPTDIR=../../scripts

# extract, rebalance and tokenize the data
python3 balance_and_truncate.py

# create 10 iterations of 10-fold cross-validation sets
python3 $SCRIPTDIR/make_iter_folds.py 10 10 jodel_all_tok.csv
