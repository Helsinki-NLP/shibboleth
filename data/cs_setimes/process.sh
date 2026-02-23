module load python-data

SCRIPTDIR=../../scripts

python3 preprocess_setimes.py

# create 10 iterations of 10-fold cross-validation sets
python3 $SCRIPTDIR/make_iter_folds.py 10 10 train_tok.tsv dev_tok.tsv test_tok.tsv

# add feature annotation:
for ITER in {1..10}; do
        for FOLD in {1..10}; do
                python3 $SCRIPTDIR/annotate_with_lexicon.py \
                        -wl lexicons/filtered_whitelist.txt \
                        -bl lexicons/filtered_blacklist.txt \
                        -i iter_folds/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD".tsv \
                        -o iter_folds/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD"_features.jsonl
        done
done
