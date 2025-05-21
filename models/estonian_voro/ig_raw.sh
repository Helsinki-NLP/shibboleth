#! /bin/bash -l

#SBATCH -J finnic_ig_ev_r_token
#SBATCH -o finnic_ig_ev_r_token.%j.out
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
MODELID=$BASEDIR/models/estonian_voro/model_ev_"$TOK"
TESTSET=$BASEDIR/data_groundtruth/finnic/pkev_test_tok.csv

if [ ! -d "$MODELID/best" ]; then
	python3 $BASEDIR/scripts/get_best_checkpoint.py $MODELID
fi

python3 $BASEDIR/scripts/attrib_ig.py \
    $MODELID/best \
    $TESTSET \
    $MODELID/ig_token.jsonl \
    $TOK
