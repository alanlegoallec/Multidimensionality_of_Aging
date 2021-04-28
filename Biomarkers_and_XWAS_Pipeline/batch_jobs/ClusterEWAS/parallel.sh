#!/bin/bash

# Done :
target_datasets=( 'ImmuneSystem' )
#target_datasets=( '\*' '*instances01' '*instances1.5x' '*instances23' 'Abdomen' 'AbdomenLiver' 'AbdomenPancreas' 'Arterial' 'ArterialPulseWaveAnalysis' 'ArterialCarotids' 'Biochemistry' 'BiochemistryUrine' 'BiochemistryBlood' 'Brain' 'BrainCognitive' 'BrainMRI' 'Eyes' 'EyesAll' 'EyesFundus' 'EyesOCT' 'Hearing' 'Heart' 'HeartECG' 'HeartMRI' 'ImmuneSystem' 'Lungs' 'Musculoskeletal' 'MusculoskeletalSpine' 'MusculoskeletalHips' 'MusculoskeletalKnees' 'MusculoskeletalFullBody' 'MusculoskeletalScalars' 'PhysicalActivity' )
input_datasets=( 'PhysicalActivity' )
#input_datasets=( 'Alcohol' 'Diet' 'Education' 'ElectronicDevices' 'Employment' 'FamilyHistory' 'Eyesight' 'Mouth' 'GeneralHealth' 'Breathing' 'Claudification' 'GeneralPain' 'ChestPain' 'CancerScreening' 'Medication' 'Hearing' 'Household' 'MentalHealth' 'OtherSociodemographics' 'PhysicalActivityQuestionnaire' 'SexualFactors' 'Sleep' 'SocialSupport' 'SunExposure' 'EarlyLifeFactors' 'Smoking' 'PhysicalActivity' 'ArterialStiffness' 'BloodBiochemistry' 'BloodCount' 'BloodPressure' 'BoneDensitometryOfHeel' 'BraindMRIWeightedMeans' 'BrainGreyMatterVolumes' 'BrainSubcorticalVolumes' 'CarotidUltrasound' 'CognitiveFluidIntelligence' 'CognitiveMatrixPatternCompletion' 'CognitiveNumericMemory' 'CognitivePairedAssociativeLearning' 'CognitivePairsMatching' 'CognitiveProspectiveMemory' 'CognitiveReactionTime' 'CognitiveSymbolDigitSubstitution' 'CognitiveTowerRearranging' 'CognitiveTrailMaking' 'ECGAtRest' 'EyeAcuity' 'EyeAutorefraction' 'EyeIntraocularPressure' 'UrineBiochemistry' 'Spirometry' 'HeartSize' 'HeartPWA' 'HandGripStrength' 'HearingTest' 'AnthropometryImpedance' 'AnthropometryBodySize' )

#input_datasets=( 'medical_diagnoses_A' 'medical_diagnoses_B' 'medical_diagnoses_C' 'medical_diagnoses_D' 'medical_diagnoses_E' 'medical_diagnoses_F' 'medical_diagnoses_G' 'medical_diagnoses_H' 'medical_diagnoses_I' 'medical_diagnoses_J' 'medical_diagnoses_K' 'medical_diagnoses_L' 'medical_diagnoses_M' 'medical_diagnoses_N' 'medical_diagnoses_O' 'medical_diagnoses_P' 'medical_diagnoses_Q' 'medical_diagnoses_R' 'medical_diagnoses_S' 'medical_diagnoses_T' 'medical_diagnoses_U' 'medical_diagnoses_V' 'medical_diagnoses_W' 'medical_diagnoses_X' 'medical_diagnoses_Y' 'medical_diagnoses_Z' )



memory=64G
n_cores=1


for input_dataset in "${input_datasets[@]}"
	do
	for target_dataset in "${target_datasets[@]}"
	do
			job_name="create_cluster_${target_dataset}_${input_dataset}.job"
			out_file="./logs/create_cluster_${target_dataset}_${input_dataset}.out"
			err_file="./logs/create_cluster_${target_dataset}_${input_dataset}.err"

			sbatch --error=$err_file --output=$out_file --job-name=$job_name --mem-per-cpu=$memory -c $n_cores -p short -t 0-11:59 batch_jobs/ClusterEWAS/single.sh $target_dataset $input_dataset
	done
done
