#!/bin/bash
targets=( "Age" )
#targets=( "Age" )
#models=( "LightGbm" )
models=( "NeuralNetwork" )
datasets=( 'CognitiveReactionTime' )
#datasets=( 'BrainGreyMatterVolumes' 'BrainSubcorticalVolumes' 'BraindMRIWeightedMeans' 'BrainMRIAllBiomarkers' 'CognitiveReactionTime' 'CognitiveMatrixPatternCompletion' 'CognitiveTowerRearranging' 'CognitiveSymbolDigitSubstitution' 'CognitivePairedAssociativeLearning' 'CognitiveProspectiveMemory' 'CognitiveNumericMemory' 'CognitiveFluidIntelligence' 'CognitiveTrailMaking' 'CognitivePairsMatching' 'CognitiveAllBiomarkers' 'BrainAndCognitive' 'EyeAutorefraction' 'EyeAcuity' 'EyeIntraocularPressure' 'EyesAllBiomarkers' 'HearingTest' 'Spirometry' 'BloodPressure' 'CarotidUltrasound' 'ArterialStiffness' 'VascularAllBiomarkers' 'HeartAllBiomarkers' 'HeartSize' 'HeartPWA' 'HeartMRIAll' 'ECGAtRest' 'AnthropometryImpedance' 'AnthropometryBodySize' 'BoneDensitometryOfHeel' 'HandGripStrength' 'MusculoskeletalAllBiomarkers' 'BloodBiochemistry' 'UrineBiochemistry' 'Biochemistry' 'BloodCount' 'PhysicalActivity' )
#datasets=( 'PhysicalActivity' )
#datasets=( "PhysicalActivity1250" "PhysicalActivity1500" "PhysicalActivity1750" "PhysicalActivity2000" )
outer_splits=10
inner_splits=9
n_iter=30
n_splits=10

memory=20G
n_cores=1

search_dir_clusters='/n/groups/patel/samuel/AutomaticClusters'
search_dir_base='/n/groups/patel/samuel/final_inputs'


# declare -a IDsLoads=()
# for dataset in "${datasets[@]}"
# do
# 	job_name="Load_${dataset}.job"
# 	out_file="./logs/Load_${dataset}.out"
# 	err_file="./logs/Load_${dataset}.err"
# 	IDLoad=$(sbatch --parsable --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/predictions/load_datasets.sh $dataset)
# 	IDsLoads+=($IDLoad)
# done
#
# printf -v joinedIDsLoads '%s:' "${IDsLoads[@]}"
# job_name="Create_raw_data.job"
# out_file="./logs/Create_raw_data.out"
# err_file="./logs/Create_raw_data.err"
# ID_raw=$(sbatch --parsable --dependency=afterok:${joinedIDsLoads%:} --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=32 -c $n_cores -p short -t 0-11:59 batch_jobs/predictions/create_raw_data.sh)
#
#
# n_cores=1
# job_name="Create_clusters.job"
# out_file="./logs/Create_clusters.out"
# err_file="./logs/Create_clusters.err"
# ID_clusters=$(sbatch --parsable --dependency=afterok:$ID_raw --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=50G -c $n_cores -p medium -t 4-11:59 batch_jobs/predictions/create_clusters.sh)


#
for target in "${targets[@]}"
do
	for model in "${models[@]}"
	do
# # 		# for dataset in "$search_dir_clusters"/*
# # 		# do
# # 		# 	dataset_clean=$(basename $dataset .csv)
# # 		# 	declare -a IDs=()
# # 		# 	for ((fold=0; fold <= $outer_splits-1; fold++))
# # 		# 	do
# # 		# 		job_name="${target}_${model}_${dataset_clean}_${fold}.job"
# # 		# 		out_file="./logs/${target}_${model}_${dataset_clean}_${fold}.out"
# # 		# 		err_file="./logs/${target}_${model}_${dataset_clean}_${fold}.err"
# # 		# 		#ID=$(sbatch --parsable --dependency=afterok:$ID_clusters --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/predictions/single.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold)
# # 		# 		# To del :
# # 		# 		ID=$(sbatch --parsable  --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/predictions/single.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold)
# # 		# 		IDs+=($ID)
# # 		# 	done
# # 		#
# # 		# 	job_name="${target}_${model}_${dataset_clean}_features.job"
# # 		# 	out_file="./logs/${target}_${model}_${dataset_clean}_features.out"
# # 		# 	err_file="./logs/${target}_${model}_${dataset_clean}_features.err"
# # 		# 	# To del :
# # 		# 	sbatch  --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/predictions/single_features.sh $model $n_iter $target $dataset $n_splits
# # 		# 	#sbatch --dependency=afterok:$ID_clusters --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/predictions/single_features.sh $model $n_iter $target $dataset $n_splits
# # 		#
# # 		#
# # 		# 	job_name="${target}_${model}_${dataset_clean}_postprocessing.job"
# # 		# 	out_file="./logs/${target}_${model}_${dataset_clean}_postprocessing.out"
# # 		# 	err_file="./logs/${target}_${model}_${dataset_clean}_postprocessing.err"
# # 		#
# # 		# 	printf -v joinedIDS '%s:' "${IDs[@]}"
# # 		# 	sbatch --dependency=afterok:${joinedIDS%:} --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/predictions/postprocessing.sh $model $target $dataset $outer_splits
# # 		#
# # 		# done
		for dataset in "${datasets[@]}"
		do
		# 	declare -a IDs=()
		# 	for ((fold=0; fold <= $outer_splits-1; fold++))
		# 	do
		# 		job_name="${target}_${model}_${dataset}_${fold}.job"
		# 		out_file="./logs/${target}_${model}_${dataset}_${fold}.out"
		# 		err_file="./logs/${target}_${model}_${dataset}_${fold}.err"
		#
		# 		# To del :
		# 		ID=$(sbatch --parsable  --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-11:59 batch_jobs/predictions/single.sh $model $outer_splits $inner_splits $n_iter $target $dataset $fold)
		# 		IDs+=($ID)
		# 	done

			job_name="${target}_${model}_${dataset}_features.job"
			out_file="./logs/${target}_${model}_${dataset}_features.out"
			err_file="./logs/${target}_${model}_${dataset}_features.err"

			# To del :

			sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-11:59 batch_jobs/predictions/single_features.sh $model $n_iter $target $dataset $n_splits
			#sbatch --error=$err_file --dependency=afterok:$ID_raw --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p medium -t 4-23:59 batch_jobs/predictions/single_features.sh $model $n_iter $target $dataset $n_splits

			# job_name="${target}_${model}_${dataset}_postprocessing.job"
			# out_file="./logs/${target}_${model}_${dataset}_postprocessing.out"
			# err_file="./logs/${target}_${model}_${dataset}_postprocessing.err"
			#
			# printf -v joinedIDS '%s:' "${IDs[@]}"
			# sbatch --dependency=afterok:${joinedIDS%:} --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/predictions/postprocessing.sh $model $target $dataset $outer_splits

			# if [ $target = "Age" ] && [ $model = "LightGbm" ]
			# then
			# 	job_name="Linear_${dataset}.job"
			# 	out_file="./logs/Linear_${dataset}.out"
			# 	err_file="./logs/Linear_${dataset}.err"
			# 	sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/predictions/linear_study.sh $dataset
			# fi

		done
	done
done
