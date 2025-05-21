#! /bin/bash -l

#SBATCH -J 2clxlmr_loo
#SBATCH -o 2clxlmr_loo.%j.out
#SBATCH -e 2clxlmr_loo.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 24:00:00

module load pytorch

export 'TOKENIZERS_PARALLELISM=false'
python3 loo.py model_2cl_xlmr/checkpoint-110-epoch-10 xlmroberta 2 fin24_test_2cl.tsv model_2cl_xlmr/explanations.json