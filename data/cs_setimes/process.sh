module load python-data

SCRIPTDIR=../../scripts

python3 preprocess_setimes.py

# create 10 iterations of 10-fold cross-validation sets
python3 $SCRIPTDIR/make_iter_folds.py 10 10 train_tok.tsv dev_tok.tsv test_tok.tsv
