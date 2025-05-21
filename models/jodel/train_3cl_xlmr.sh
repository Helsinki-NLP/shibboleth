#! /bin/bash -l

#SBATCH -J 3cl_xlmr
#SBATCH -o 3cl_xlmr.%j.out
#SBATCH -e 3cl_xlmr.%j.err
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

python3 ../explainability_scripts/train.py xlm-roberta-base data_cl3_train_200k.tsv data_cl3_dev_20k.tsv model_3cl_xlmr
