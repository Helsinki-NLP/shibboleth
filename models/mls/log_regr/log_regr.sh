#! /bin/bash -l

#SBATCH -J gml_time_logregr
#SBATCH -o gml_time_logregr.%j.out
#SBATCH -e gml_time_logregr.%j.err
#SBATCH --mem=300G
#SBATCH -p small
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=janine.siewert@helsinki.fi

module load pytorch

python3 ../../explainability_scripts/logistic_regression.py --trainfile ../ReN_time_shuffled_train.txt --testfile ../ReN_time_shuffled_test.txt --iterations 500 --dtype 'int8'

