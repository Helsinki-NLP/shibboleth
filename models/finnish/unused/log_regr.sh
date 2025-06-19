#! /bin/bash -l

#SBATCH -J fi2cl_logregr
#SBATCH -o fi2cl_logregr.%j.out
#SBATCH -e fi2cl_logregr.%j.err
#SBATCH --mem=300G
#SBATCH -p small
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=janine.siewert@helsinki.fi

module load pytorch

python3 ../explainability_scripts/logistic_regression.py --trainfile fin24_train_2cl.tsv --testfile fin24_test_2cl.tsv --iterations 500 --dtype 'int8'

