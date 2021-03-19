#!/bin/bash
regenerate_data=false
targets=( "Age" )
organs_to_run=( "*" "*instances01" "*instances1.5x" "*instances23" "Brain" "BrainCognitive" "BrainMRI" "Eyes" "EyesFundus" "EyesOCT" "Hearing" "Lungs" "Arterial" "ArterialPulseWaveAnalysis" "ArterialsCarotids" "Heart" "HeartECG" "HeartMRI" "Abdomen" "AbdomenLiver" "AbdomenPancreas" "Musculoskeletal" "MusculoskeletalSpine" "MusculoskeletalHips" "MusculoskeletalKnees" "MusculoskeletalFullBody" "MusculoskeletalScalars" "PhysicalActivity" "Biochemistry" "BiochemistryUrine" "BiochemistryBlood" "ImmuneSystem" )

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	while IFS= read -r line; do
		organ1=$(echo $line | cut -d ',' -f1)
		organ2=$(echo $line | cut -d ',' -f2)
		sample_size=$(wc -l < ../data/GWAS_data_"${target}"_"${organ1}"_"${organ2}".tab)
		mem_per_cpu=$(printf "%.0f" $(expr 0.27*${sample_size}*0.1+1000 | bc))
		# Use multiplicative coef 1.2 at first, then double it for the rare jobs that time out.
		time=$(printf "%.0f" $( echo "2.4*e(1.5*l(${sample_size}/1000))+45" | bc -l))
		if [ $time -lt 720 ]; then
			partition=short
		elif [ $time -lt 7200 ]; then
			partition=medium
		else
			partition=long
		fi
		version=MI09D_"${target}"_"${organ1}"_"${organ2}"
		job_name="${version}.job"
		out_file="../eo/${version}.out"
		err_file="../eo/${version}.err"
		to_run=false
		path_out=../eo/MI09D_"${target}"_"${organ1}"_"${organ2}".out
		path_err=../eo/MI09D_"${target}"_"${organ1}"_"${organ2}".err
		if $regenerate_data || ! test -f "${path_out}" || ( ( ! grep -q "gen corr (1,2)" "${path_out}" ) && ( ! grep -q "ERROR: Heritability estimate is close to" "${path_err}" ) && ( ! grep -q "terminate called after throwing an instance of 'boost::numeric::ublas::internal_logic'" "${path_err}" ) ); then
			to_run=true
		fi
		if [[ ! " ${organs_to_run[@]} " =~ " ${organ1} " ]] || [[ ! " ${organs_to_run[@]} " =~ " ${organ2} " ]]; then
			to_run=false
		fi
		if [ $(sacct -u al311 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ] || [ $(sacct -u cp179 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ] || [ $(sacct -u jp379 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ] || [ $(sacct -u sc646 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ] || [ $(sacct -u sd375 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ] || [ $(sacct -u mj209 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ]; then
			to_run=false
		fi
		if $to_run; then
			echo "${version}"
			ID=$(sbatch --dependency=$1 --parsable -p $partition -t $time -c 10 --mem-per-cpu $mem_per_cpu --error="${err_file}" --output="${out_file}" --job-name="${job_name}" MI09CD_GWAS_bolt.sh "${target}" "${organ1}" "${organ2}" "reml_correlation")
			IDs+=($ID)
		fi
	done < ../data/GWAS_genetic_correlations_pairs_"${target}".csv
outer_folds=( "0" )
done

# Produce the list of job dependencies for the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo "${dependencies}"

