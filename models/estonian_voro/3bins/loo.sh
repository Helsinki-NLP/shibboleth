#! /bin/bash -l

#SBATCH -J loo_ev_3b
#SBATCH -o loo_ev_3b.%j.out
#SBATCH -e loo_ev_3b.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 3:00:00

set -e

module load pytorch

BASEDIR=../../..
TESTSET=$BASEDIR/data_groundtruth/finnic/pkev_test_tok.csv

for BIN in bin1 bin2 bin3 randbin; do
	if [ ! -d "model_$BIN/best" ]; then
		python3 $BASEDIR/scripts/get_best_checkpoint.py "model_$BIN"
	fi

	python3 $BASEDIR/scripts/attrib_loo.py "model_$BIN/best" $TESTSET "model_$BIN/best"/loo_type.jsonl raw type
	python3 $BASEDIR/scripts/attrib_loo.py "model_$BIN/best" $TESTSET "model_$BIN/best"/loo_token.jsonl raw token

done
