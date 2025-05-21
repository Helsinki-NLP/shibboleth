#! /bin/bash -l

#SBATCH -J 4clgb_loo
#SBATCH -o 4clgb_loo.%j.out
#SBATCH -e 4clgb_loo.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 24:00:00

module load pytorch

export 'TOKENIZERS_PARALLELISM=false'
python3 loo.py model_4cl_gb/checkpoint-1008-epoch-7 bert 4 data_cl4_test_1k.tsv model_4cl_gb/explanations.json