#! /bin/bash -l

#SBATCH -J loo_btc_st_r
#SBATCH -o loo_btc_st_r.%j.out
#SBATCH -e loo_btc_st_r.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 8:00:00

set -e

BASEDIR=../..
MODELID=model_setimes_bertic_raw
TESTSET=/scratch/project_2005047/explainability/data_groundtruth/bcms/setimes_large/test_tok.csv

module load pytorch

# echo "LOO with correct examples"
# python3 $SCRIPTDIR/loo_correct_masking_words_instance.py $MODELID/best $TESTSET $MODELID/expl_correct_msk_w_set.json raw

#echo "Filter and sort explanations"
#python3 $SCRIPTDIR/filter_sort_explanations.py $MODELID/expl_correct_msk_w_sl.json 100 > $MODELID/filtered_words_msk_w_sl_sc.txt
#python3 $SCRIPTDIR/filter_sort_explanations_mc.py $MODELID/expl_correct_msk_w_sl.json 100 4 > $MODELID/filtered_words_msk_w_sl_mc.txt

python3 $BASEDIR/scripts/loo_instance2.py $MODELID/best $TESTSET $MODELID/loo_type.jsonl raw type
python3 $BASEDIR/scripts/loo_instance2.py $MODELID/best $TESTSET $MODELID/loo_token.jsonl raw token
