#!/bin/bash
#SBATCH --parsable
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

set -e
module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.1
source /home/al311/python_3.6.0/bin/activate

python -u ../scripts/MI08_Attentionmaps.py $1 $2 $3 $4 && echo "PYTHON SCRIPT COMPLETED"

