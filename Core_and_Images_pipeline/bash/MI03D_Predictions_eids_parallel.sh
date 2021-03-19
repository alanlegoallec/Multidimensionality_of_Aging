#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
folds=( "val" "test" )

declare -a IDs=()
for fold in "${folds[@]}"; do
	if [ $fold == "train" ]; then
		time=120
		memory=128G
	else
		time=10
		memory=64G
	fi
	for target in "${targets[@]}"; do
		version=MI03D_${target}_${fold}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		ID=$(sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI03D_Predictions_eids.sh $target $fold)
		IDs+=($ID)
	done
done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

