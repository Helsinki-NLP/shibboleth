#! /bin/bash -l

#SBATCH -J shap_sc_3b
#SBATCH -o shap_sc_3b.%j.out
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 1-12:00:00

set -e

module load pytorch

BASEDIR=../../..
TESTSET=$BASEDIR/data_groundtruth/scandinavian/slide_sl_test_tok.csv

for BIN in bin1 bin2 bin3 randbin; do
	if [ ! -d "model_$BIN/best" ]; then
		python3 $BASEDIR/scripts/get_best_checkpoint.py "model_$BIN"
	fi

	python3 $BASEDIR/scripts/attrib_shap.py "model_$BIN/best" $TESTSET "model_$BIN/best"/shap_token.jsonl raw

done
