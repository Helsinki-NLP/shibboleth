#! /bin/bash -l

#SBATCH -J loo_gyb_sw
#SBATCH -o loo_gyb_sw.%j.out
#SBATCH -e loo_gyb_sw.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 72:00:00


set -e

SCRIPTDIR=../explainability_scripts
MODELTYPE=emanjavacas/GysBERT-v2
MODELID=model_gml_time_shuffled_gysbert
TESTSET=ReN_time_shuffled_test.txt

module load pytorch

echo "LOO with correct examples"
python3 $SCRIPTDIR/loo_correct_masking_subwords_maxlen.py $MODELTYPE $MODELID/best $TESTSET $MODELID/expl_correct_msk_sw.json

echo "Filter and sort explanations"
python3 $SCRIPTDIR/filter_sort_explanations.py $MODELID/expl_correct_msk_sw.json 100 > $MODELID/filtered_words_msk_sw_sc.txt
python3 $SCRIPTDIR/filter_sort_explanations_mc.py $MODELID/expl_correct_msk_sw.json 100 3 > $MODELID/filtered_words_msk_sw_mc.txt

echo "Done"