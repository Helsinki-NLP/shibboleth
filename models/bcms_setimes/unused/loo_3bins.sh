#! /bin/bash -l

#SBATCH -J loo_bcms_3b
#SBATCH -o loo_bcms_3b_2.%j.out
#SBATCH -e loo_bcms_3b_2.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 12:00:00

set -e

module load pytorch

BASEDIR=../..
TESTSET=$BASEDIR/data_groundtruth/bcms/setimes_large/test_tok.csv

for BIN in bin2 bin3 randbin; do

	python3 $BASEDIR/scripts/attrib_loo.py "model_set_btc_3bins_$BIN/best" $TESTSET "model_set_btc_3bins_$BIN"/loo_type.jsonl raw type
	python3 $BASEDIR/scripts/attrib_loo.py "model_set_btc_3bins_$BIN/best" $TESTSET "model_set_btc_3bins_$BIN"/loo_token.jsonl raw token

done