module load python-data

SCRIPTDIR=../../scripts

# merge the train, dev and test files from SLIDE, remove multilabel instances and tokenize them (the tokenized versions are not used)
python3 extract_from_slide.py

# load the Wiktionary/kaikki word lists, merge them, filter words with conflicting information in SLIDE, and organize them into a whitelist and a blacklist
python3 filter_lists.py

# create 10 iterations of 10-fold cross-validation sets
python3 $SCRIPTDIR/make_iter_folds.py 10 10 slide_sl_all_tok.csv

# add feature annotation for the dev sets
for ITER in {1..10}; do
	for FOLD in {1..10}; do
		python3 $SCRIPTDIR/annotate_with_lexicon.py \
			-wl filtered_whitelist.txt \
			-bl filtered_blacklist.txt \
			-i iter_folds/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD".tsv \
			-o iter_folds/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD"_features.jsonl
	done
done
