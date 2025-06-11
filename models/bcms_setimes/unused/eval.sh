#! /bin/bash -l

#SBATCH -J e_tn_eq_xlmr
#SBATCH -o e_tn_eq_xlmr.%j.out
#SBATCH -e e_tn_eq_xlmr.%j.err
#SBATCH --mem=24G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 1:00:00

module load pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=../cache_dir/

MODELID=model_twitter_news_eqcl_xlmr
DEVSET=data_twitter_news_eqcl_dev_ml.tsv

for CHKPT in `ls -v $MODELID`; do
	python3 ../explainability_scripts/eval_multilabel.py $MODELID/$CHKPT $DEVSET
done
