#!/bin/bash
regenerate_data=false
#targets=( "Age" "Sex" )
targets=( "Age" )
organs=( "*" "*instances01" "*instances1.5x" "*instances23" "Brain" "BrainCognitive" "BrainMRI" "Eyes" "EyesFundus" "EyesOCT" "Hearing" "Lungs" "Arterial" "ArterialPulseWaveAnalysis" "ArterialCarotids" "Heart" "HeartECG" "HeartMRI" "Abdomen" "AbdomenLiver" "AbdomenPancreas" "Musculoskeletal" "MusculoskeletalSpine" "MusculoskeletalHips" "MusculoskeletalKnees" "MusculoskeletalFullBody" "MusculoskeletalScalars" "PhysicalActivity" "Biochemistry" "BiochemistryUrine" "BiochemistryBlood" "ImmuneSystem" )
chromosomesS=( "autosome" "X" )

#loop through the jobs to submit
declare -a IDs=()
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
		sample_size=$(wc -l < ../data/GWAS_data_"${target}"_"${organ}".tab)
		mem_per_cpu=$(printf "%.0f" $(expr 0.27*$sample_size*0.1+1000 | bc))
		base_time=$(printf "%.0f" $( echo "0.3*e(1.5*l($sample_size/1000))+45" | bc -l))
		for chromosomes in "${chromosomesS[@]}"; do
			if [ $chromosomes == "X" ]; then
				analyses=( "lmm" "reml" )
			elif [ $chromosomes == "autosome" ]; then
				analyses=( "lmm" )
			fi
			for analysis in "${analyses[@]}"; do
				if [ $analysis == reml ]; then
					time=$(( 2*$base_time ))
				else
					time=$base_time
				fi
				if [ $time -lt 720 ]; then
					partition=short
				elif [ $time -lt 7200 ]; then
					partition=medium
				else
					partition=long
				fi
				version=MI09C_"${analysis}"_"${target}"_"${organ}"_"${chromosomes}"
				job_name="${version}.job"
				out_file="../eo/${version}.out"
				err_file="../eo/${version}.err"
				path_results="../data/GWAS_"${target}"_"${organ}"_"${chromosomes}".stats"
				to_run=true
				if ! $regenerate_data && ( ( [ $analysis == "lmm" ] && ( test -f "${path_results}" || grep -q "ERROR: Heritability estimate is close to" "${err_file}" ) ) || ( [ $analysis == "reml" ] && test -f "${out_file}" && grep -q "h2g" "${out_file}" ) ) ; then
					to_run=false
				fi
				if [ $(sacct -u al311 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ] || [ $(sacct -u cp179 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ] || [ $(sacct -u jp379 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ] || [ $(sacct -u sc646 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ] || [ $(sacct -u sd375 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ] || [ $(sacct -u mj209 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep `printf '%q' "${job_name}"` | egrep 'PENDING|RUNNING' | wc -l) -ne 0 ]; then
					to_run=false
				fi
				if $to_run; then
					echo $version
					ID=$(sbatch --dependency=$1 --parsable -p $partition -t $time -c 10 --mem-per-cpu $mem_per_cpu --error=$err_file --output=$out_file --job-name=$job_name MI09CD_GWAS_bolt.sh "${target}" "${organ}" "${chromosomes}" "${analysis}")
					IDs+=($ID)
				fi
			done
		done
	done
done

# Produce the list of job dependencies for the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

