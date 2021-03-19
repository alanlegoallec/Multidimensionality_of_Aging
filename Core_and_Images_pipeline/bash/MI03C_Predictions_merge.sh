#!/bin/bash
#SBATCH --parsable
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

set -e
module load gcc/6.2.0
module load python/3.6.0
source ~/python_3.6.0/bin/activate
python -u ../scripts/MI03C_Predictions_merge.py $1 $2 && echo "PYTHON SCRIPT COMPLETED"

