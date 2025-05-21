#! /bin/bash -l

#SBATCH -J 3bins_randbin
#SBATCH -o t_3bins_randbin.%j.out
#SBATCH -e t_3bins_randbin.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 18:00:00
#SBATCH --mail-type=ALL

module load pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=../cache_dir/

python3 /scratch/project_2005047/explainability/scripts/train_setimes_raw.py  classla/bcms-bertic  /scratch/project_2005047/explainability/data_groundtruth/bcms/setimes_large/3bins/train_3bins_randbin.csv /scratch/project_2005047/explainability/data_groundtruth/bcms/setimes_large/dev_tok.csv  model_set_btc_3bins_randbin
