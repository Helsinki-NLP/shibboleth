#! /bin/bash -l

#SBATCH -J t_set_btc_tk
#SBATCH -o t_set_btc_tk.%j.out
#SBATCH -e t_set_btc_tk.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL

module load pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=../cache_dir/

python3 /scratch/project_2005047/explainability/scripts/train_setimes_tok.py  classla/bcms-bertic /scratch/project_2005047/explainability/data_groundtruth/bcms/setimes_large/train_tok.csv /scratch/project_2005047/explainability/data_groundtruth/bcms/setimes_large/dev_tok.csv  model_setimes_bertic_tok
