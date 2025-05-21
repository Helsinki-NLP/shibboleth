#! /bin/bash -l

#SBATCH -J loo_5cl_xlmr
#SBATCH -o loo_5cl_xlmr.%j.out
#SBATCH -e loo_5cl_xlmr.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 2:00:00

set -e

SCRIPTDIR=../explainability_scripts

module load pytorch

echo "Select best checkpoint"
python3 $SCRIPTDIR/get_best_checkpoint.py model_5cl_xlmr

echo "LOO with correct examples"
python3 $SCRIPTDIR/loo_correct.py model_5cl_xlmr/best data_cl5_test_20k.tsv model_5cl_xlmr/expl_correct.json

echo "Filter and sort explanations"
python3 $SCRIPTDIR/filter_sort_explanations.py model_5cl_xlmr/expl_correct.json 100 > model_5cl_xlmr/filtered_words.txt

echo "LOO with incorrect examples"
python3 $SCRIPTDIR/loo_incorrect.py model_5cl_xlmr/best data_cl5_test_20k.tsv model_5cl_xlmr/expl_incorrect.json

echo "Done"
