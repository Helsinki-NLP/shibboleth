#! /bin/bash -l

#SBATCH -J loo_gml_time_xlmr
#SBATCH -o loo_gml_time_xlmr.%j.out
#SBATCH -e loo_gml_time_xlmr.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 6:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=janine.siewert@helsinki.fi

SCRIPTDIR=../explainability_scripts

module load pytorch

echo "Select best checkpoint"
python3 $SCRIPTDIR/get_best_checkpoint.py model_gml_time_shuffled_xlmr

echo "LOO with correct examples"
python3 $SCRIPTDIR/loo_correct.py model_gml_time_shuffled_xlmr/best ReN_time_shuffled_test.txt model_gml_time_shuffled_xlmr/expl_correct2.json

echo "Filter and sort explanations"
python3 $SCRIPTDIR/filter_sort_explanations.py model_gml_time_shuffled_xlmr/expl_correct2.json 100 > model_gml_time_shuffled_xlmr/filtered_words2.txt

echo "LOO with incorrect examples"
python3 $SCRIPTDIR/loo_incorrect.py model_gml_time_shuffled_xlmr/best ReN_time_shuffled_test.txt model_gml_time_shuffled_xlmr/expl_incorrect2.json

echo "Done"

