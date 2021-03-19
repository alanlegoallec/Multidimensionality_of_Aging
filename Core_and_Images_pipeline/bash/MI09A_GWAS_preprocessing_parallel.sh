#!/bin/bash
targets=( "Age" )
time=60
memory=8G

declare -a IDs=()
for target in "${targets[@]}"; do
	version=MI09A_${target}
	job_name="$version.job"
	out_file="../eo/$version.out"
	err_file="../eo/$version.err"
	ID=$(sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI09A_GWAS_preprocessing.sh $target)
	IDs+=($ID)
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies
