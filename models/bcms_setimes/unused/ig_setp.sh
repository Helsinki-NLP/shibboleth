#! /bin/bash -l

#SBATCH -J ig_setp
#SBATCH -o ig_setp.%j.out
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
TESTSET=$BASEDIR/data_groundtruth/bcms/setimes/setp_test.csv
MODELID=$BASEDIR/models/bcms/model_setp_btc

python3 $BASEDIR/scripts/attrib_ig.py "$MODELID/best" $TESTSET "$MODELID"/ig_token.jsonl raw
