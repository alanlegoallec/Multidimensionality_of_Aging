#!/bin/bash
#SBATCH --parsable
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --open-mode=truncate

module load gcc/6.2.0
module load python/3.6.0
module load cuda/10.1
source /home/al311/python_3.6.0/bin/activate

srun -n 1 -t "$((${14}-3))" --mem "$(($SLURM_MEM_PER_CPU-1))" bash -c "{ python -u ../scripts/MI02_Training.py $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13}; } && touch ../eo/$SLURM_JOB_NAME.success"
sleep 5 # wait slurm get the job status into its database
echo \n\nSummary:
sacct --format=JobID,Submit,Start,End,State,Partition,ReqTRES%30,CPUTime,MaxRSS,NodeList%30 --units=M -j $SLURM_JOBID
sh MI02_Training_sendJobFinishEmail.sh $SLURM_JOB_NAME
success=([ -f myJob.success ])
if $success; then
	rm $SLURM_JOB_NAME.success
fi
$success && exit 0 || exit 1

