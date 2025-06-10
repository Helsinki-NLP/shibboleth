#! /bin/bash -l

#SBATCH -J tr_ev_3
#SBATCH -o tr_ev_3.%j.out
#SBATCH -e tr_ev_3.%j.err
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
export HF_DATASETS_CACHE=../../cache_dir/

BASEDIR=../../..
BIN=3

python3 $BASEDIR/scripts/train.py \
	-model tartuNLP/EstBERT \
	-train $BASEDIR/data_groundtruth/finnic/bins/train_bin"$BIN".csv \
	-valid $BASEDIR/data_groundtruth/finnic/pkev_dev_tok.csv \
	-outdir model_bin"$BIN" \
	-column raw
