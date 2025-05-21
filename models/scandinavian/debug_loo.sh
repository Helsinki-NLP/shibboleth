#! /bin/bash -l

#SBATCH -J debug_loo_r
#SBATCH -o debug_loo_r.%j.out
#SBATCH -e debug_loo_r.%j.err
#SBATCH --mem=12G
#SBATCH -p gputest
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 0:15:00

set -e

module load pytorch

BASEDIR=../..
TOK=raw
MODELID=model_slide_"$TOK"
TESTSET=$BASEDIR/data_groundtruth/scandinavian/slide_sl_test_tok.csv

python3 $BASEDIR/scripts/loo_instance2.py $MODELID/best $TESTSET $MODELID/loo2_debug.json $TOK type
