#! /bin/bash -l

#SBATCH -J shib_jo_spec
#SBATCH -o shib_jo_spec.%a.%j.out
#SBATCH -e shib_jo_spec.%a.%j.err
#SBATCH --mem=64G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --array=0-9:1
#SBATCH -A project_2005047
#SBATCH -t 71:00:00

module load pytorch/2.7

DATADIR=/scratch/project_2005047/shibboleth/data/jodel/iter_folds
SCRIPTDIR=/scratch/project_2005047/shibboleth/scripts
#TRAINDIR=/scratch/project_2005047/shibboleth/models/jodel
TRAINDIR=/scratch/project_2006235/shibboleth/models/jodel_spec

BASEMODEL=dbmdz/bert-base-german-cased
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

echo "Done"
