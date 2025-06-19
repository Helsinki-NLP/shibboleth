#! /bin/bash -l

#SBATCH -J 3cl_finbert
#SBATCH -o 3cl_finbert.%j.out
#SBATCH -e 3cl_finbert.%j.err
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
python3 train_simpletf.py TurkuNLP/bert-base-finnish-cased-v1 fin24_train_3cl.tsv fin24_dev_3cl.tsv ./model_3cl_finbert