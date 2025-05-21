#! /bin/bash -l

#SBATCH -J 4clxlmr_loo
#SBATCH -o 4clxlmr_loo.%j.out
#SBATCH -e 4clxlmr_loo.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 24:00:00

module load pytorch

export 'TOKENIZERS_PARALLELISM=false'
python3 loo.py model_4cl_xlmr/checkpoint-15625-epoch-5 xlmroberta 4 data_cl4_test_20k.tsv model_4cl_xlmr/explanations.json