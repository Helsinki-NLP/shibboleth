#! /bin/bash -l

#SBATCH -J ig_bcms_3b
#SBATCH -o ig_bcms_3b.%j.out
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 1-12:00:00

set -e

module load pytorch

BASEDIR=/scratch/project_2005047/explainability
TOK=raw
TESTSET=$BASEDIR/data_groundtruth/bcms/setimes_large/test_tok.csv


for BIN in bin1 bin2 bin3 randbin; do
        MODELID=$BASEDIR/models/bcms/model_set_btc_3bins_$BIN
	if [ ! -d "model_$BIN/best" ]; then
		python3 $BASEDIR/scripts/get_best_checkpoint.py "$MODELID"
	fi

	python3 $BASEDIR/scripts/attrib_ig.py "$MODELID/best" $TESTSET "$MODELID/best"/ig_token.jsonl raw

done
