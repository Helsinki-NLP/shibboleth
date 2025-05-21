#! /bin/bash -l

#SBATCH -J greek_logregr
#SBATCH -o greek_logregr.%j.out
#SBATCH -e greek_logregr.%j.err
#SBATCH --mem=300G
#SBATCH -p small
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -A project_2005047
#SBATCH -t 10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=janine.siewert@helsinki.fi

module load pytorch

python3 ../explainability_scripts/logistic_regression.py --trainfile data_cl4_train_9k.tsv --testfile data_cl4_test_1k.tsv --iterations 500 --dtype 'int8'

