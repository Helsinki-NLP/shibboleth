#! /bin/bash -l

#SBATCH -J 3clxlmr_loo
#SBATCH -o 3clxlmr_loo.%j.out
#SBATCH -e 3clxlmr_loo.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 24:00:00

module load pytorch

export 'TOKENIZERS_PARALLELISM=false'
python3 loo.py model_3cl_xlmr/checkpoint-110-epoch-10 xlmroberta 3 fin24_test_3cl.tsv model_3cl_xlmr/explanations.json