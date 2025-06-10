#! /bin/bash -l

#SBATCH -J gml_time_xlmr
#SBATCH -o gml_time_xlmr.%j.out
#SBATCH -e gml_time_xlmr.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=janine.siewert@helsinki.fi

module load pytorch
pip3 install -U transformers

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=../cache_dir/

python3 ../explainability_scripts/train.py xlm-roberta-base ReN_time_shuffled_train.txt ReN_time_shuffled_dev.txt model_gml_time_shuffled_xlmr
