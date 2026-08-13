#! /bin/bash -l
module load pytorch/2.6

DATADIR=/scratch/project_2005047/explainability/data_groundtruth/bcms/twitter/iter_folds
SCRIPTDIR=/scratch/project_2005047/shibboleth/scripts
#TRAINDIR=/scratch/project_2005047/shibboleth/models/bcms_twitter
TRAINDIR=/scratch/project_2006235/shibboleth/models/bcms_twitter

BASEMODEL=xlm-roberta-base
TOK=raw

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export HF_HOME=$TRAINDIR/../cache

for ITER in {1..10}; do
  for FOLD in {1..10}; do
    echo "Iteration $ITER - Fold $FOLD"

    TRAINFILE=$DATADIR/iter"$ITER"/train_iter"$ITER"_fold"$FOLD".tsv
    DEVFILE=$DATADIR/iter"$ITER"/dev_iter"$ITER"_fold"$FOLD".tsv
    TESTFILE=$DATADIR/test.tsv
    MODELDIR=$TRAINDIR/iter"$ITER"/model_fold"$FOLD"

    if [ ! -f $MODELDIR/eval_test.txt ]; then
      echo "Evaluate on joint test set"
      python3 $SCRIPTDIR/eval_label.py $MODELDIR/best $TESTFILE $MODELDIR/eval_test.txt $TOK
    fi;
  done
done
