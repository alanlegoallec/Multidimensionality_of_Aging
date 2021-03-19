#!/bin/bash
targets=( "Age" )

#loop through the jobs to submit
declare -a IDs=()
organs_groups=( "TimeSeries" "Images" "Videos" )
for organs_group in "${organs_groups[@]}"; do
	if [ $organs_group == "TimeSeries" ]; then
		organs=( "Arterial" "Heart" "PhysicalActivity" )
		memory=8G
		time=5
		partition=short
	elif [ $organs_group == "Images" ]; then
		organs=( "Brain" "Eyes" "Arterial" "Heart" "Abdomen" "Musculoskeletal" "PhysicalActivity" )
		memory=32G
		partition=gpu
		# time will be defined further below depending on the model
	elif [ $organs_group == "Videos" ]; then
		organs=( "Heart" )
		memory=8G
		time=5
		partition=short
	fi
	for target in "${targets[@]}"; do
		for organ in "${organs[@]}"; do
			if [ $organs_group == "TimeSeries" ]; then
				if [ $organ == "Arterial" ]; then
					views=( "PulseWaveAnalysis" )
				elif [ $organ == "Heart" ]; then
					views=( "ECG" )
				elif [ $organ == "PhysicalActivity" ]; then
					views=( "FullWeek" "Walking" )
				else
					echo "Organ $organ does not match any TimeSeries organs."
				fi
			elif [ $organs_group == "Images" ]; then
				if [ $organ == "Brain" ]; then
					views=( "MRI" )
				elif [ $organ == "Eyes" ]; then
					views=( "Fundus" "OCT" )
				elif [ $organ == "Arterial" ]; then
					views=( "Carotids" )
				elif [ $organ == "Heart" ]; then
					views=( "MRI" )
				elif [ $organ == "Abdomen" ]; then
					views=( "Liver" "Pancreas" )
				elif [ $organ == "Musculoskeletal" ]; then
					views=( "Spine" "Hips" "Knees" "FullBody" )
				elif [ $organ == "PhysicalActivity" ]; then
					views=( "FullWeek" )
				else
					views=( "MRI" )
				fi
			elif [ $organs_group == "Videos" ]; then
				if [ $organ == "Heart" ]; then
					views=( "MRI" )
				fi
			fi
			for view in "${views[@]}"; do
				if [ $organs_group == "TimeSeries" ]; then
					if [ $organ == "Arterial" ] || [ $organ == "ECG" ]; then
						transformations=( "TimeSeries" )
					elif [ $organ == "PhysicalActivity" ]; then
						if [ $view == "FullWeek" ]; then
							transformations=( "Acceleration" "TimeSeriesFeatures" )
						elif [ $view == "Walking" ]; then
							transformations=( "3D" )
						fi
					fi
				elif [ $organs_group == "Images" ]; then
					if [ $organ == "Eyes" ] || [ $organ == "Arterial" ] || [ $view == "Hips" ] || [ $view == "Knees" ]; then
						time=300
					else
						time=150
					fi
					if [ $organ == "Brain" ]; then
						transformations=( "SagittalRaw" "SagittalReference" "CoronalRaw" "CoronalReference" "TransverseRaw" "TransverseReference" )
					elif [ $organ == "Eyes" ]; then
						transformations=( "Raw" )
					elif [ $organ == "Arterial" ]; then
						transformations=( "Mixed" "LongAxis" "CIMT120" "CIMT150" "ShortAxis" )
					elif [ $organ == "Heart" ]; then
						transformations=( "2chambersRaw" "2chambersContrast" "3chambersRaw" "3chambersContrast" "4chambersRaw" "4chambersContrast" )
					elif [ $organ == "Abdomen" ]; then
						transformations=( "Raw" "Contrast" )
					elif [ $organ == "Musculoskeletal" ]; then
						if [ $view == "Spine" ]; then
							transformations=( "Sagittal" "Coronal" )
						elif [ $view == "Hips" ] || [ $view == "Knees" ]; then
							transformations=( "MRI" )
						elif [ $view == "FullBody" ]; then
							transformations=( "Mixed" "Figure" "Skeleton" "Flesh" )
						fi
					elif [ $organ == "PhysicalActivity" ]; then
						if [ $view == "FullWeek" ]; then
							transformations=( "GramianAngularField1minDifference" "GramianAngularField1minSummation" "MarkovTransitionField1min" "RecurrencePlots1min" )
						fi
					fi
				elif [ $organs_group == "Videos" ]; then
					if [ $organ == "Heart" ]; then
						transformations=( "3chambersRawVideo" "4chambersRawVideo" "34chambersRawVideo" )
					fi
				else
					echo "organs_group ${organs_group} is not among Scalars, TimeSeries, Images, or Videos"
				fi
				for transformation in "${transformations[@]}"; do
					version=MI08_${target}_${organ}_${view}_${transformation}
					job_name="$version.job"
					out_file="../eo/$version.out"
					err_file="../eo/$version.err"
					if ( ! test -f "$out_file" || ( ! grep -q "Done." "$out_file" ) ) && ( [ $(sacct -u $USER --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep $job_name | egrep 'PENDING|RUNNING' | wc -l) -eq 0 ] ); then
						echo $version
						args=( --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time )
						if [ $organs_group == "Images" ]; then
							args+=( --partition=gpu --gres=gpu:1 -x compute-g-16-175,compute-g-16-176,compute-g-16-197 )
						else
							args+=( --partition=short )
						fi
						ID=$(sbatch "${args[@]}" MI08_Attentionmaps.sh $target $organ $view $transformation)
						#ID=$(sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time -p gpu  MI08_Attentionmaps.sh $target $organ $view $transformation)
						IDs+=($ID)
					fi
				done
			 done
		 done
	 done
 done
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

