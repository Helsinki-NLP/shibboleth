#! /bin/bash -l

#SBATCH -J tr_slide_r
#SBATCH -o tr_slide_r.%j.out
#SBATCH -e tr_slide_r.%j.err
#SBATCH --mem=8G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 12:00:00

module load pytorch

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE=../cache_dir/

BASEDIR=../..

python3 $BASEDIR/scripts/train.py \
	-model vesteinn/ScandiBERT \
	-train $BASEDIR/data_groundtruth/scandinavian/slide_sl_train_tok.csv \
	-valid $BASEDIR/data_groundtruth/scandinavian/slide_sl_dev_tok.csv \
	-outdir model_slide_raw \
	-column raw \
	-shuffle
