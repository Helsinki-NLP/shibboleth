#! /bin/bash -l

#SBATCH -J ev_proc
#SBATCH -o process_log.%j.out
#SBATCH -e process_log.%j.err
#SBATCH --mem=180G
#SBATCH -p test
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 0:15:00

module load python-data

SCRIPTDIR=../../scripts

# extract, clean and tokenize the data from paralleelkorpus (the tokenized versions are not used)
python3 extract_from_pk.py

# load the word lists from different resources (kaikki, synaq, giellatekno), merge them, filter out words that do not occur in paralleelkorpus, and organize them into a whitelist and a blacklist
python3 merge_lexicons.py

# create 10 iterations of 10-fold cross-validation sets
python3 $SCRIPTDIR/make_iter_folds.py 10 10 pkev_all_tok.csv

# add feature annotation:
for ITER in {1..10}; do
	for FOLD in {1..10}; do
		python3 $SCRIPTDIR/annotate_with_lexicon.py \
			-wl filtered_whitelist.txt \
			-bl filtered_blacklist.txt \
			-i iter_folds/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD".tsv \
			-o iter_folds/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD"_features.jsonl
	done
done
