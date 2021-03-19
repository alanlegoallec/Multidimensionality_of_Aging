#!/bin/bash
targets=( "Age" )
time=60
memory=24G
for target in "${targets[@]}"; do
	version=MI09G_${target}
	job_name="$version.job"
	out_file="../eo/$version.out"
	err_file="../eo/$version.err"
	ID=$(sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI09G_GWAS_plots.sh $target)
	IDs+=($ID)
done
# Produce the list of job dependencies for the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies
