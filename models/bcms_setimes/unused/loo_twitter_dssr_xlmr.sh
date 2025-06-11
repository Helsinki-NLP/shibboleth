#! /bin/bash -l

#SBATCH -J loo_t_ds_xlmr
#SBATCH -o loo_t_ds_xlmr.%j.out
#SBATCH -e loo_t_ds_xlmr.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 2:00:00

set -e

SCRIPTDIR=../explainability_scripts
MODELID=model_twitter_dssr_xlmr
TESTSET=data_twitter_dssr_test_ml.tsv

module load pytorch

echo "Select best checkpoint"
python3 $SCRIPTDIR/get_best_checkpoint.py $MODELID

echo "LOO with correct examples"
python3 $SCRIPTDIR/loo_correct.py $MODELID/best $TESTSET $MODELID/expl_correct.json -ml

echo "Filter and sort explanations"
python3 $SCRIPTDIR/filter_sort_explanations.py $MODELID/expl_correct.json 100 > $MODELID/filtered_words.txt

# echo "LOO with incorrect examples"
# python3 $SCRIPTDIR/loo_incorrect.py $MODELID/best $TESTSET $MODELID/expl_incorrect.json

echo "Done"
