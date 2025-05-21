#! /bin/bash -l

#SBATCH -J t_set_btc
#SBATCH -o t_set_btc.%j.out
#SBATCH -e t_set_btc.%j.err
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

python3 ./../../explainability/scripts/train.py  xlm-roberta-base ../../data_groundtruth/bcms/setimes_large/train_tok.csv ../../data_groundtruth/bcms/setimes_large/dev_tok.csv  model_setimes_xlmr

#for CHKPT in `ls -v model_twitter_eqcl_bertic`; do
#	python3 ../explainability_scripts/eval_multilabel.py model_twitter_eqcl_bertic/$CHKPT data_twitter_eqcl_dev_ml.tsv
#done

