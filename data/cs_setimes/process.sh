module load python-data

SCRIPTDIR=../../scripts

# create 10 iterations of 10-fold cross-validation sets
python3 $SCRIPTDIR/make_iter_folds.py 10 10 ???
