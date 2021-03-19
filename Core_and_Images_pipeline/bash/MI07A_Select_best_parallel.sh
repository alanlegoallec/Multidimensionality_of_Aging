#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
pred_types=( "instances" "eids" )
memory=128G
time=15
#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	for pred_type in "${pred_types[@]}"; do
		version=MI07A_${target}_${pred_type}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		ID=$(sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI07A_Select_best.sh $target $pred_type)
		IDs+=($ID)
	done
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

