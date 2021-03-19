#!/bin/bash
targets=( "Age" )
organs=( "Brain" "Eyes" "Arterial" "Heart" "Abdomen" "Musculoskeletal" "PhysicalActivity" )
time=60
memory=8G
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		version=MI01C_${target}_${organ}
		job_name="$version.job"
		out_file="../eo/$version.out"
		err_file="../eo/$version.err"
		if ls ../data/data-features_${organ}_*_${target}_*.csv 1> /dev/null 2>&1; then
			echo "The files for ../data/data-features_${organ}_*_${target}_*.csv already exist."
		else
			sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI01C_Preprocessing_folds.sh $target $organ	
		fi
	done
done

