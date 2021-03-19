#!/bin/bash
#targets=( "Age" "Sex" )
targets=( "Age" )
organs=( "Brain" "Eyes" "Arterial" "Heart" "Abdomen" "Musculoskeletal" "PhysicalActivity" )
architectures=( "DenseNet121" "DenseNet169" "DenseNet201" "Xception" "InceptionV3" "InceptionResNetV2" "ResNeXt50" "ResNeXt101" "ResNet50V2" "ResNet101V2" "ResNet152V2" )
#n_fc_layersS=( "0" "1" "2" "3" "4" "5" )
n_fc_layersS=( "1" )
#n_fc_nodesS=( "16" "64" "128" "256" "512" "1024" )
n_fc_nodesS=( "1024" )
#optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
#learning_rates=( "0.01" "0.001" "0.0001" "0.00001" "0.000001" )
learning_rates=( "0.0001" )
#weight_decays=( "0.0" "0.1" "0.2" "0.3" "0.4" )
weight_decays=( "0.1" )
#dropout_rates=( "0.0" "0.25" "0.5" "0.75" "0.95" )
dropout_rates=( "0.5" )
#data_augmentation_factors=( "0.0" "0.1" "0.5" "1.0" "2.0" )
data_augmentation_factors=( "1.0" )
outer_folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )
#outer_folds=( "0" )
memory=8G
time=600
for target in "${targets[@]}"; do
	for organ in "${organs[@]}"; do
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
		for view in "${views[@]}"; do
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
			for transformation in "${transformations[@]}"; do
				for architecture in "${architectures[@]}"; do
					for n_fc_layers in "${n_fc_layersS[@]}"; do
						if [ $n_fc_layers == "0" ]; then
							n_fc_nodesS_amended=( "0" )
						else
							n_fc_nodesS_amended=( "${n_fc_nodesS[@]}" )
						fi
						for n_fc_nodes in "${n_fc_nodesS_amended[@]}"; do
							for optimizer in "${optimizers[@]}"; do
								for learning_rate in "${learning_rates[@]}"; do
									for weight_decay in "${weight_decays[@]}"; do
										for dropout_rate in "${dropout_rates[@]}"; do
											for outer_fold in "${outer_folds[@]}"; do
												for data_augmentation_factor in "${data_augmentation_factors[@]}"; do
													version=MI02_${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${outer_fold}
													job_name="$version"
													out_file="../eo/$version.out"
													err_file="../eo/$version.err"
													if ! test -f "$out_file" || ( ! grep -q "Done." "$out_file" && grep -q " improved from " "$out_file" ); then
														similar_models=MI02_${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${outer_fold}	
														if [ $(sacct -u al311 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep $similar_models | egrep 'PENDING|RUNNING' | wc -l) -eq 0 ] && [ $(sacct -u cp179 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep $similar_models | egrep 'PENDING|RUNNING' | wc -l) -eq 0 ] && [ $(sacct -u jp379 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep $similar_models | egrep 'PENDING|RUNNING' | wc -l) -eq 0 ] && [ $(sacct -u sc646 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep $similar_models | egrep 'PENDING|RUNNING' | wc -l) -eq 0 ] && [ $(sacct -u sd375 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep $similar_models | egrep 'PENDING|RUNNING' | wc -l) -eq 0 ] && [ $(sacct -u mj209 --format=JobID,JobName%150,MaxRSS,NNodes,Elapsed,State | grep $similar_models | egrep 'PENDING|RUNNING' | wc -l) -eq 0 ]; then
															echo SUBMITTING: $version
															sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI02_Training.sh $target $organ $view $transformation $architecture $n_fc_layers $n_fc_nodes $optimizer $learning_rate $weight_decay $dropout_rate $data_augmentation_factor $outer_fold $time
														#else
														#	echo "Pending/Running: $version (or similar model)"
														fi
													#else
													#	echo "Already converged: $version"
													fi
												done
											done
										done
									done
								done
							done
						done
					done
				done
			done
		done
	done
done

