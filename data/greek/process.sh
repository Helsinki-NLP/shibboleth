module load python-data

SCRIPTDIR=../../scripts

python3 balance_concatenate.py

# create 10 iterations of 10-fold cross-validation sets
python3 $SCRIPTDIR/make_iter_folds.py 10 10 grdc_all.csv
