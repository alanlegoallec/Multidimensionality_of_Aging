#!/bin/bash
chromosomesS=( "autosome" "X" )
#loop through the jobs to submit
declare -a IDs=()
for chromosomes in "${chromosomesS[@]}"; do
	version=MI09B_${chromosomes}
	ID=$(sbatch --dependency=$1 --error=../eo/$version.err --output=../eo/$version.out --job-name=$version.job MI09B_GWAS_remove.sh ${chromosomes}) 
	IDs+=($ID)
done

# Produce the list of job dependencies for the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

