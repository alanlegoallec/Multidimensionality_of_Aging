#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
pred_types=( "instances" "eids" )
memory=20G
n_cpu_cores=10
time=600
declare -a IDs=()
for target in "${targets[@]}"; do
	for pred_type in "${pred_types[@]}"; do
		version=MI05A_${target}_${pred_type}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		ID=$(sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cpu_cores -t $time MI05A_Ensembles_predictions_generate_and_merge.sh $target $pred_type)
		IDs+=($ID)
	done
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

