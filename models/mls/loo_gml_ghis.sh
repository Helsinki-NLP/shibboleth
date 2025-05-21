#! /bin/bash -l

#SBATCH -J loo_gml_time_ghis
#SBATCH -o loo_gml_time_ghis.%j.out
#SBATCH -e loo_gml_time_ghis.%j.err
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
python3 $SCRIPTDIR/get_best_checkpoint.py /scratch/project_2005047/explainability_middle-low-saxon/model_gml_time_shuffled_ghisbert

echo "LOO with correct examples"
python3 $SCRIPTDIR/loo_correct.py /scratch/project_2005047/explainability_middle-low-saxon/model_gml_time_shuffled_ghisbert/best /scratch/project_2005047/explainability_middle-low-saxon/ReN_time_shuffled_test.txt /scratch/project_2005047/explainability_middle-low-saxon/model_gml_time_shuffled_ghisbert/expl_correct.json

echo "Filter and sort explanations"
python3 $SCRIPTDIR/filter_sort_explanations.py /scratch/project_2005047/explainability_middle-low-saxon/model_gml_time_shuffled_ghisbert/expl_correct.json 100 > /scratch/project_2005047/explainability_middle-low-saxon/model_gml_time_shuffled_ghisbert/filtered_words.txt

echo "LOO with incorrect examples"
python3 $SCRIPTDIR/loo_incorrect.py /scratch/project_2005047/explainability_middle-low-saxon/model_gml_time_shuffled_ghisbert/best /scratch/project_2005047/explainability_middle-low-saxon/ReN_time_shuffled_test.txt /scratch/project_2005047/explainability_middle-low-saxon/model_gml_time_shuffled_ghisbert/expl_incorrect.json

echo "Done"

