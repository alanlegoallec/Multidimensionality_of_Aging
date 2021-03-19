#!/bin/bash
regenerate_performances=false
memory=2G
#generate file with list of ensemble models (avoids the trouble with parsing files with * character)
file_list_ensemble_models="../data/list_ensemble_models.txt"
ls ../data/Predictions_*_\*_* > $file_list_ensemble_models

#parse the file line by line to submit a job for each ensemble model
declare -a IDs=()
while IFS= read -r model
do
	IFS='_' read -ra PARAMETERS <<< ${model%".csv"}
	pred_type="${PARAMETERS[1]}"
	target="${PARAMETERS[2]}"
	organ="${PARAMETERS[3]}"
	view="${PARAMETERS[4]}"
	transformation="${PARAMETERS[5]}"
	architecture="${PARAMETERS[6]}"
	n_fc_layers="${PARAMETERS[7]}"
	n_fc_nodes="${PARAMETERS[8]}"
	optimizer="${PARAMETERS[9]}"
	learning_rate="${PARAMETERS[10]}"
	weight_decay="${PARAMETERS[11]}"
	dropout_rate="${PARAMETERS[12]}"
	data_augmentation_factor="${PARAMETERS[13]}"
	fold="${PARAMETERS[14]}"
	version="${pred_type}_${target}_${organ}_${view}_${transformation}_${architecture}_${n_fc_layers}_${n_fc_nodes}_${optimizer}_${learning_rate}_${weight_decay}_${dropout_rate}_${data_augmentation_factor}_${fold}"
	name=MI05B-"$version"
	job_name="$name.job"
	out_file="../eo/$name.out"
	err_file="../eo/$name.err"
	time=15 #5
	#time=2 #debug mode
	#allocate more time for the training fold because of the larger sample size
	if [ $fold == "train" ]; then
		time=$(( 8*$time ))
	fi
	#check if the predictions have already been generated. If not, do not run the model.
	if ! test -f "../data/Predictions_${version}.csv"; then
		echo The predictions at "../data/Predictions_${version}.csv" cannot be found. The job cannot be run.
		break
	fi
	#if regenerate_performances option is on or if the performances have not yet been generated, run the job
	if ! test -f "../data/Performances_${version}.csv" || $regenerate_performances; then
		echo Submitting job for "$version"
		ID=$(sbatch --dependency=$1 --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -t $time MI04A05B_Performances_generate.sh "${target}" "${organ}" "${view}" "${transformation}" "${architecture}" "${n_fc_layers}" "${n_fc_nodes}" "${optimizer}" "${learning_rate}" "${weight_decay}" "${dropout_rate}" "${data_augmentation_factor}" "${fold}" "${pred_type}")
		IDs+=($ID)
	#else
	#	echo Performance for $version have already been generated.
	fi
done < "$file_list_ensemble_models"
# Produce the list of job dependencies fr the next step
printf -v IDs_list '%s:' "${IDs[@]}"
dependencies="${IDs_list%:}"
echo $dependencies

