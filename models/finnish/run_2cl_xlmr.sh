#! /bin/bash -l

#SBATCH -J 2cl_xlmr
#SBATCH -o 2cl_xlmr.%j.out
#SBATCH -e 2cl_xlmr.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 24:00:00

module load pytorch

#export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'

export 'TOKENIZERS_PARALLELISM=false'
python3 train_simpletf.py xlm-roberta-base fin24_train_2cl.tsv fin24_dev_2cl.tsv ./model_2cl_xlmr