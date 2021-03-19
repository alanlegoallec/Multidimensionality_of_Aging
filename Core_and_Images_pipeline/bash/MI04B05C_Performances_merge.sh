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

python -u ../scripts/MI04B05C_Performances_merge.py $1 $2 $3 $4 && echo "PYTHON SCRIPT COMPLETED"

