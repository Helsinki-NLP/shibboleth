#! /bin/bash -l

#SBATCH -J 2clfin_loo
#SBATCH -o 2clfin_loo.%j.out
#SBATCH -e 2clfin_loo.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 24:00:00

module load pytorch

export 'TOKENIZERS_PARALLELISM=false'
python3 loo.py model_2cl_finbert/checkpoint-110-epoch-10 bert 2 fin24_test_2cl.tsv model_2cl_finbert/explanations.json