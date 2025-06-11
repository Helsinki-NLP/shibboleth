#! /bin/bash -l

#SBATCH -J shib_bcmstw
#SBATCH -o shib_bcmstw.%a.%j.out
#SBATCH -e shib_bcmstw.%a.%j.err
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --array=0-99:1
#SBATCH -A project_2005047
#SBATCH -t 71:00:00

module load pytorch

DATADIR=/scratch/project_2005047/explainability/data_groundtruth/bcms/twitter/iter_folds
SCRIPTDIR=/scratch/project_2005047/shibboleth/scripts
TRAINDIR=/scratch/project_2005047/shibboleth/models/bcms_twitter
#TRAINDIR=/scratch/project_2006235/shibboleth/models/bcms_twitter

BASEMODEL=xlm-roberta-base
TOK=raw

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$TRAINDIR/../cache

ITER=$(( (SLURM_ARRAY_TASK_ID / 10) + 1 ))
FOLD=$(( (SLURM_ARRAY_TASK_ID % 10) + 1 ))
echo "Iteration $ITER - Fold $FOLD"

TRAINFILE=$DATADIR/iter"$ITER"/train_iter"$ITER"_fold"$FOLD".tsv
DEVFILE=$DATADIR/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD".tsv
TESTFILE=$DATADIR/test.tsv
MODELDIR=$TRAINDIR/iter"$ITER"/model_fold"$FOLD"

echo "Training"
mkdir -p $MODELDIR
python3 $SCRIPTDIR/train.py -model $BASEMODEL -train $TRAINFILE -valid $DEVFILE -outdir $MODELDIR -column $TOK

echo "Evaluate on joint test set"
python3 $SCRIPTDIR/eval_label.py $MODELDIR/best $TESTFILE $MODELDIR/eval_test.txt $TOK

echo "Compute attributions"
echo "- LOO (per token)"
python3 $SCRIPTDIR/attrib_loo.py $MODELDIR/best $DEVFILE $MODELDIR/loo_token.jsonl $TOK token

echo "- IG"
python3 $SCRIPTDIR/attrib_ig.py $MODELDIR/best $DEVFILE $MODELDIR/ig_token.jsonl $TOK

echo "- SHAP"
python3 $SCRIPTDIR/attrib_shap.py $MODELDIR/best $DEVFILE $MODELDIR/shap_token.jsonl $TOK

echo "- LIME"
python3 $SCRIPTDIR/attrib_lime.py $MODELDIR/best $DEVFILE $MODELDIR/lime_token.jsonl $TOK

echo "Done"

