#! /bin/bash -l

#SBATCH -J shib_ev
#SBATCH -o shib_ev.%a.%j.out
#SBATCH -e shib_ev.%a.%j.err
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --array=0-99:1
#SBATCH -A project_2005047
#SBATCH -t 8:00:00

module load pytorch

DATADIR=/scratch/project_2005047/shibboleth/data/estonian_voro/iter_folds
SCRIPTDIR=/scratch/project_2005047/shibboleth/scripts
MODELDIR=/scratch/project_2005047/shibboleth/estonian_voro
#MODELDIR=/scratch/project_2006235/shibboleth/estonian_voro

BASEMODEL=tartuNLP/EstBERT
TOK=raw

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$MODELDIR/cache

ITER=$(( (SLURM_ARRAY_TASK_ID / 10) + 1 ))
FOLD=$(( (SLURM_ARRAY_TASK_ID % 10) + 1 ))
echo "Iteration $ITER - Fold $FOLD"

echo "Training"
mkdir -p iter"$ITER"
python3 $SCRIPTDIR/train.py \
	-model $BASEMODEL \
	-train $DATADIR/iter"$ITER"/train_iter"$ITER"_fold"$FOLD".tsv \
	-valid $DATADIR/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD".tsv \
	-outdir iter"$ITER"/model_fold"$FOLD" \
	-column $TOK

echo "TODO: evaluate on joint test set"

echo "Compute attributions"
echo "- LOO (per token)"
python3 $SCRIPTDIR/attrib_loo.py \
	iter"$ITER/model_fold"$FOLD"/best \
	$DATADIR/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD".tsv \
	iter"$ITER/model_fold"$FOLD"/loo_token.jsonl \
	$TOK

echo "- IG"
python3 $SCRIPTDIR/attrib_ig.py \
	iter"$ITER/model_fold"$FOLD"/best \
	$DATADIR/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD".tsv \
	iter"$ITER/model_fold"$FOLD"/ig_token.jsonl \
	$TOK

echo "- SHAP"
python3 $SCRIPTDIR/attrib_shap.py \
	iter"$ITER/model_fold"$FOLD"/best \
	$DATADIR/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD".tsv \
	iter"$ITER/model_fold"$FOLD"/shap_token.jsonl \
	$TOK

echo "- LIME"
python3 $SCRIPTDIR/attrib_lime.py \
	iter"$ITER/model_fold"$FOLD"/best \
	$DATADIR/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD".tsv \
	iter"$ITER/model_fold"$FOLD"/shap_token.jsonl \
	$TOK
