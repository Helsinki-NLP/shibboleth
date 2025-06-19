#! /bin/bash -l

#SBATCH -J loo_xlmrf_w
#SBATCH -o loo_xlmrf_w.%j.out
#SBATCH -e loo_xlmrf_w.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 2:00:00


set -e

SCRIPTDIR=../explainability_scripts
MODELID=model_2cl_xlmr
TESTSET=fin24_test_2cl.tsv

module load pytorch

echo "LOO with correct examples"
python3 $SCRIPTDIR/loo_correct_masking_words.py $MODELID/best_model $TESTSET $MODELID/expl_correct_msk_w.json

echo "Filter and sort explanations"
python3 $SCRIPTDIR/filter_sort_explanations.py $MODELID/expl_correct_msk_w.json 100 > $MODELID/filtered_words_msk_w_sc.txt
python3 $SCRIPTDIR/filter_sort_explanations_mc.py $MODELID/expl_correct_msk_w.json 100 2 > $MODELID/filtered_words_msk_w_mc.txt

echo "Done"