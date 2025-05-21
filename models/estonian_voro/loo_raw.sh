#! /bin/bash -l

#SBATCH -J loo_ev_r
#SBATCH -o loo_ev_r.%j.out
#SBATCH -e loo_ev_r.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 1:00:00

set -e

module load pytorch

BASEDIR=../..
TOK=raw
MODELID=model_ev_"$TOK"
TESTSET=$BASEDIR/data_groundtruth/finnic/pkev_test_tok.csv

if [ ! -d "$MODELID/best" ]; then
	python3 $BASEDIR/scripts/get_best_checkpoint.py $MODELID
fi

#python3 $BASEDIR/scripts/loo_correct_masking_words_instance.py $MODELID/best $TESTSET $MODELID/loo_corr_msk_w.json $TOK

python3 $BASEDIR/scripts/loo_instance2.py $MODELID/best $TESTSET $MODELID/loo_type.jsonl $TOK type
python3 $BASEDIR/scripts/loo_instance2.py $MODELID/best $TESTSET $MODELID/loo_token.jsonl $TOK token
