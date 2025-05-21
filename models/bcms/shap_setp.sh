#! /bin/bash -l

#SBATCH -J bcms_shap_setp
#SBATCH -o bcms_shap_setp.%j.out
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
MODELID=$BASEDIR/models/bcms/model_setp_btc/
TESTSET=$BASEDIR/data_groundtruth/bcms/setimes/setp_test.csv


python3 $BASEDIR/scripts/attrib_shap.py \
    $MODELID/best \
    $TESTSET \
    $MODELID/shap_token.jsonl \
    $TOK
