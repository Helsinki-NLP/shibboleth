#! /bin/bash -l

#SBATCH -J 11cl_xlmr
#SBATCH -o 11cl_xlmr.%j.out
#SBATCH -e 11cl_xlmr.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 24:00:00

module load pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=../cache_dir/

python3 ../explainability_scripts/train.py xlm-roberta-base data_cl11_train.tsv data_cl11_dev.tsv model_11cl_xlmr
