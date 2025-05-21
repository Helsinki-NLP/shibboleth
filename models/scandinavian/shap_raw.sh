#! /bin/bash -l

#SBATCH -J shap_slide_r_token
#SBATCH -o shap_slide_r_token.out
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 12:00:00

set -e

module load pytorch

BASEDIR=/scratch/project_2005047/explainability
TOK=raw
MODELID=$BASEDIR/models/scandinavian/model_slide_"$TOK"
TESTSET=$BASEDIR/data_groundtruth/scandinavian/slide_sl_test_tok.csv

if [ ! -d "$MODELID/best" ]; then
	python3 $BASEDIR/scripts/get_best_checkpoint.py $MODELID
fi

python3 $BASEDIR/scripts/attrib_shap.py \
    $MODELID/best \
    $TESTSET \
    $MODELID/shap_token.jsonl \
    $TOK
