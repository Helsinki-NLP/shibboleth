#! /bin/bash -l

#SBATCH -J loo_bcms_setp
#SBATCH -o loo_bcms_setp.%j.out
#SBATCH -e loo_bcms_setp.%j.err
#SBATCH --mem=12G
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 04:00:00

set -e

module load pytorch

BASEDIR=../..
TESTSET=$BASEDIR/data_groundtruth/bcms/setimes/setp_test.csv

python3 $BASEDIR/scripts/attrib_loo.py "$BASEDIR/models/bcms/model_setp_btc/best/" $TESTSET "$BASEDIR/models/bcms/model_setp_btc"/loo_type.jsonl raw type
python3 $BASEDIR/scripts/attrib_loo.py "$BASEDIR/models/bcms/model_setp_btc/best/" $TESTSET "$BASEDIR/models/bcms/model_setp_btc"/loo_token.jsonl raw token



#for BIN in bin2 bin3 randbin; do

#	python3 $BASEDIR/scripts/attrib_loo.py "model_set_btc_3bins_$BIN/best" $TESTSET "model_set_btc_3bins_$BIN"/loo_type.jsonl raw type
#	python3 $BASEDIR/scripts/attrib_loo.py "model_set_btc_3bins_$BIN/best" $TESTSET "model_set_btc_3bins_$BIN"/loo_token.jsonl raw token

#done