#! /bin/bash -l

#SBATCH -J t_ds_xlmr
#SBATCH -o t_ds_xlmr.%j.out
#SBATCH -e t_ds_xlmr.%j.err
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

python3 ../explainability_scripts/train.py xlm-roberta-base data_twitter_dssr_train_sl.tsv data_twitter_dssr_dev_sl.tsv model_twitter_dssr_xlmr

for CHKPT in `ls -v model_twitter_dssr_xlmr`; do
	python3 ../explainability_scripts/eval_multilabel.py model_twitter_dssr_xlmr/$CHKPT data_twitter_dssr_dev_ml.tsv
done
