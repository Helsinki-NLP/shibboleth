#! /bin/bash -l

#SBATCH -J bcms_lime_setimes_r_token
#SBATCH -o bcms_lime_setimes_r_token.%j.out
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 36:00:00

set -e

module load pytorch

BASEDIR=/scratch/project_2005047/explainability
TOK=raw
MODELID=$BASEDIR/models/bcms/model_setimes_bertic_"$TOK"
TESTSET=$BASEDIR/data_groundtruth/bcms/setimes_large/test_tok.csv

if [ ! -d "$MODELID/best" ]; then
	python3 $BASEDIR/scripts/get_best_checkpoint.py $MODELID
fi

python3 $BASEDIR/scripts/attrib_lime.py \
    $MODELID/best \
    $TESTSET \
    $MODELID/lime_token.jsonl \
    $TOK
