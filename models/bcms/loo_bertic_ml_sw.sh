#! /bin/bash -l

#SBATCH -J loo_btc_ml_sw
#SBATCH -o loo_btc_ml_sw.%j.out
#SBATCH -e loo_btc_ml_sw.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 72:00:00


set -e

SCRIPTDIR=../explainability_scripts
MODELTYPE=classla/bcms-bertic
MODELID=model_twitter_dssr_bertic
TESTSET=data_twitter_dssr_test_ml.tsv

module load pytorch

echo "LOO with correct examples"
python3 $SCRIPTDIR/loo_correct_masking_subwords.py $MODELTYPE $MODELID/best $TESTSET $MODELID/expl_correct_msk_sw_ml.json -ml

echo "Filter and sort explanations"
python3 $SCRIPTDIR/filter_sort_explanations.py $MODELID/expl_correct_msk_sw_ml.json 100 > $MODELID/filtered_words_msk_sw_ml_sc.txt
python3 $SCRIPTDIR/filter_sort_explanations_mc.py $MODELID/expl_correct_msk_sw_ml.json 100 4 > $MODELID/filtered_words_msk_sw_ml_mc.txt

echo "Done"