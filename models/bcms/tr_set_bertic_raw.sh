#! /bin/bash -l

#SBATCH -J setp_btc
#SBATCH -o t_setp_btc.%j.out
#SBATCH -e t_setp_btc.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 06:00:00
#SBATCH --mail-type=ALL

module load pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=../cache_dir/

python3 /scratch/project_2005047/explainability/scripts/train_setimes_raw.py  classla/bcms-bertic /scratch/project_2005047/explainability/data_groundtruth/bcms/setimes/setp_train.csv /scratch/project_2005047/explainability/data_groundtruth/bcms/setimes/setp_dev.csv  model_setp_btc
