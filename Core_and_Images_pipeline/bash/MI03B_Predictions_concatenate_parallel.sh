#!/bin/bash
regenerate_predictions=false
#targets=( "Age" "Sex" )
targets=( "Age" )
organs=( "Brain" "Eyes" "Arterial" "Heart" "Abdomen" "Musculoskeletal" "PhysicalActivity" )
#architectures=( "VGG16" "VGG19" "DenseNet121" "DenseNet169" "DenseNet201" "Xception" "InceptionV3" "InceptionResNetV2" "ResNeXt101" "EfficientNetB7" )
#architectures=( "DenseNet201" "ResNext101" "InceptionResNetV2" "EfficientNetB7" )
architectures=( "InceptionV3" "InceptionResNetV2" )
n_fc_layersS=( "1" )
n_fc_nodesS=( "1024" )
#optimizers=( "Adam" "RMSprop" "Adadelta" )
optimizers=( "Adam" )
learning_rates=( "0.0001" )
weight_decays=( "0.1" )
dropout_rates=( "0.5" )
data_augmentation_factors=( "1.0" )
folds=( "train" "val" "test" )
outer_folds=( "0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )
time=5
memory=8G
declare -a IDs=()
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
				transformations=( "GramianAngularField1minDifference" "GramianAngularField1minSummation" "MarkovTransitionField1min" "RecurrencePlots1min" )
			fi
			for transformation in "${transformations[@]}"; do
				for architecture in "${architectures[@]}"; do
					for n_fc_layers in "${n_fc_layersS[@]}"; do
						for n_fc_nodes in "${n_fc_nodesS[@]}"; do
							for optimizer in "${optimizers[@]}"; do
								for learning_rate in "${learning_rates[@]}"; do
									for weight_decay in "${weight_decays[@]}"; do
										for dropout_rate in "${dropout_rates[@]}"; do
											for data_augmentation_factor in "${data_augmentation_factors[@]}"; do
												version=${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}
												name=MI03B_$version
												job_name="$name.job"
												out_file="../eo/$name.out"
												err_file="../eo/$name.err"
												#check if all weights have already been generated. If not, do not run the model.
												missing_weights=false
												for outer_fold in "${outer_folds[@]}"; do
													path_weights="../data/model-weights_${version}_${outer_fold}.h5"
													if ! test -f $path_weights; then
														missing_weights=true
														echo The weights at $path_weights cannot be found. The job cannot be run.
														#some weights are missing despite having an associated .out file with "THE MODEL CONVERGED!"
														#delete these files to allow the model to be run during phase MI02.
														#rm "../eo/MI02_${version}_${outer_fold}.out"
														#rm "../eo/MI02_${version}_${outer_fold}.err"
														#break
													fi
												done
												if $missing_weights; then
													continue
												fi
												#if regenerate_predictions option is on or if one of the predictions is missing, run the job
												to_run=false
												for fold in "${folds[@]}"; do
													path_predictions="../data/Predictions_instances_${version}_${fold}.csv"
													if ! test -f $path_predictions; then
														to_run=true
													fi
												done
												if $regenerate_predictions; then
													to_run=true
												fi
												if $to_run; then
													echo Submitting job for $version
													ID=$(sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI03B_Predictions_concatenate.sh $target $organ $view $transformation $architecture $n_fc_layers $n_fc_nodes $optimizer $learning_rate $weight_decay $dropout_rate $data_augmentation_factor)
													IDs+=($ID)
												#else
												#	echo Predictions for $version have already been generated.
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

# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

