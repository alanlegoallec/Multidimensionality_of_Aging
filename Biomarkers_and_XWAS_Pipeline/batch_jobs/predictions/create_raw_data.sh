#!/bin/bash
#SBATCH -p short
#SBATCH -t 0-11:59
#SBATCH --mail-user=samuel_diai@hms.harvard.edu


module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.0
source /home/sc646/alan_jupytervenv/bin/activate
python3 batch_jobs/predictions/python_caller_create_raw_data.py
