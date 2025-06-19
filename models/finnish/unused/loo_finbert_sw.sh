#! /bin/bash -l

#SBATCH -J loo_fbt_sw
#SBATCH -o loo_fbt_sw.%j.out
#SBATCH -e loo_fbt_sw.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 8:00:00


set -e

SCRIPTDIR=../explainability_scripts
MODELTYPE=TurkuNLP/bert-base-finnish-cased-v1
MODELID=model_2cl_finbert
TESTSET=fin24_test_2cl.tsv

module load pytorch

echo "LOO with correct examples"
python3 $SCRIPTDIR/loo_correct_masking_subwords.py $MODELTYPE $MODELID/best_model $TESTSET $MODELID/expl_correct_msk_sw.json

echo "Filter and sort explanations"
python3 $SCRIPTDIR/filter_sort_explanations.py $MODELID/expl_correct_msk_sw.json 100 > $MODELID/filtered_words_msk_sw_sc.txt
python3 $SCRIPTDIR/filter_sort_explanations_mc.py $MODELID/expl_correct_msk_sw.json 100 2 > $MODELID/filtered_words_msk_sw_mc.txt

echo "Done"