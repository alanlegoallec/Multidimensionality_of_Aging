#!/bin/bash
#SBATCH --parsable 
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

set -e
module load gcc/6.2.0
module load python/3.6.0
source /home/al311/python_3.6.0/bin/activate

python -u ../scripts/MI09A_GWAS_preprocessing.py $1 && echo "PYTHON SCRIPT COMPLETED"

