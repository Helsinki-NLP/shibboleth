#! /bin/bash -l

#SBATCH -J tn_eq_xlmr
#SBATCH -o tn_eq_xlmr.%j.out
#SBATCH -e tn_eq_xlmr.%j.err
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

python3 ../explainability_scripts/train.py xlm-roberta-base data_twitter_news_eqcl_train_sl.tsv data_twitter_news_eqcl_dev_sl.tsv model_twitter_news_eqcl_xlmr

for CHKPT in `ls -v model_twitter_news_eqcl_xlmr`; do
	python3 ../explainability_scripts/eval_multilabel.py model_twitter_news_eqcl_xlmr/$CHKPT data_twitter_news_eqcl_dev_ml.tsv
done
