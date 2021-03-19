#!/bin/bash
#SBATCH --output=../eo/MI01B.out
#SBATCH --error=../eo/MI01B.err
#SBATCH --mem-per-cpu=8G 
#SBATCH -c 1
#SBATCH -t 45
#SBATCH --parsable 
#SBATCH -p priority
#SBATCH --open-mode=truncate
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=alanlegoallec@g.harvard.edu

set -e
module load gcc/6.2.0
module load python/3.6.0
source /home/al311/python_3.6.0/bin/activate

python -u ../scripts/MI01B_Preprocessing_imagesIDs.py && echo "PYTHON SCRIPT COMPLETED"

