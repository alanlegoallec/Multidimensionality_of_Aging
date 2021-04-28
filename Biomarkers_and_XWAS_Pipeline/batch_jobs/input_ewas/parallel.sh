memory=8G
n_cores=1
splits=200
for ((split=0; split <= $splits-1; split++))
	do
		job_name="Input_data_${split}.job"
		out_file="./logs/Input_data_${split}.out"
		err_file="./logs/Input_data_${split}.err"
		sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/input_ewas/input_dataset.sh $split
	done
