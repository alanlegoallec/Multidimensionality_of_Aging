import pandas as pd
import glob
from string import ascii_uppercase
from functools import partial

## Environmental Variables + Medical condition
from ..processing.base_processing import read_ethnicity_data
from ..environment_processing.base_processing import path_features , path_predictions, path_inputs_env, path_target_residuals, ETHNICITY_COLS, path_input_env_inputed, path_input_env
from ..environment_processing.disease_processing import read_infectious_diseases_data, read_infectious_disease_antigens_data
from ..environment_processing.FamilyHistory import read_family_history_data
from ..environment_processing.HealthAndMedicalHistory import read_breathing_data, read_cancer_screening_data, read_chest_pain_data, read_claudication_data, read_eye_history_data, \
                                                             read_general_health_data, read_general_pain_data, read_hearing_data, read_medication_data, read_mouth_teeth_data
from ..environment_processing.LifestyleAndEnvironment import read_alcohol_data, read_diet_data, read_electronic_devices_data, read_physical_activity_data, read_sexual_factors_data,\
                                                             read_sleep_data, read_sun_exposure_data, read_smoking_data
from ..environment_processing.PsychosocialFactors import read_mental_health_data, read_social_support_data
from ..environment_processing.SocioDemographics import read_education_data, read_employment_data, read_household_data, read_other_sociodemographics_data
from ..environment_processing.HealthRelatedOutcomes import read_medical_diagnoses_data
from ..environment_processing.EarlyLifeFactors import read_early_life_factors_data

## Biomarkers as environmental predictors
from ..processing.anthropometry_processing import read_anthropometry_impedance_data, read_anthropometry_body_size_data
from ..processing.arterial_stiffness_processing import read_arterial_stiffness_data
from ..processing.biochemestry_processing import read_blood_biomarkers_data, read_urine_biomarkers_data, read_blood_count_data
from ..processing.blood_pressure_processing import read_blood_pressure_data
#from ..processing.body_composition_processing import read_body_composition_data
#from ..processing.bone_composition_processing import read_bone_composition_data
from ..processing.brain_processing import read_grey_matter_volumes_data, read_subcortical_volumes_data, read_brain_dMRI_weighted_means_data
from ..processing.carotid_ultrasound_processing import read_carotid_ultrasound_data
from ..processing.ecg_processing import read_ecg_at_rest_data
from ..processing.eye_processing import  read_eye_acuity_data, read_eye_autorefraction_data, read_eye_intraocular_pressure_data
from ..processing.heart_processing import  read_heart_PWA_data, read_heart_size_data
from ..processing.spirometry_processing import read_spirometry_data
from ..processing.bone_densitometry_processing import read_bone_densitometry_data
from ..processing.hand_grip_strength_processing import read_hand_grip_strength_data
from ..processing.hearing_tests_processing import read_hearing_test_data
from ..processing.cognitive_tests_processing import read_reaction_time_data, read_matrix_pattern_completion_data, read_tower_rearranging_data, \
                                                    read_symbol_digit_substitution_data, read_paired_associative_learning_data, \
                                                    read_prospective_memory_data, read_numeric_memory_data, read_fluid_intelligence_data, read_trail_making_data , \
                                                    read_pairs_matching_data

dict_target_to_instance_and_id = {"Brain" : (2, 100),
                           "UrineAndBlood" : (0, 100079),
                           "HeartPWA" : (2, 128),
                           "Heart" : (2, 102),
                           "Eye" : (0, 100013),
                           "EyeIntraoculaPressure" : (0, 100015),
                           "AnthropometryImpedance" : (0, 100008),
                           "BrainGreyMatterVolumes" : (2, 1101),
                           "AnthropometryBodySize" : (0, 100010),
                           "UrineBiochemestry" : (0, 100083),
                           "ArterialAndBloodPressure" : (0, 'Custom'),
                           "Spirometry" : (0, 100020),
                           "ECGAtRest" : (2, 12657),
                           "EyeAutorefraction" : (0, 100014),
                           "ArterialStiffness" : (0, 100007),
                           "BloodCount" : (0, 100081),
                           "BrainSubcorticalVolumes" : (2, 1102),
                           "EyeAcuity" : (0, 100017),
                           "HeartSize" : (2, 133),
                           "BloodPressure" : (0, 100011),
                           "SpiroAndArterialAndBp" : (0, 'Custom'),
                           "HeartImages" : (2, 'Alan'),
                           "BloodBiochemestry" : (0, 17518),
                           "Blood" : (0, 100080),
                           "Anthropometry" : (0, 100008),
                           "LiverImages" : (2, 'Alan')
                           }



map_envdataset_to_dataloader_and_field = {
    ## Environmental
    'Alcohol' : (read_alcohol_data, 100051),
    'Diet' : (read_diet_data, 100052),
    'Education' : (read_education_data, 100063),
    'ElectronicDevices' : (read_electronic_devices_data, 100053),
    'Employment' : (read_employment_data, 100064),
    'FamilyHistory' : (read_family_history_data, 100034),
    'Eyesight' : (read_eye_history_data, 100041),
    'Mouth' : (read_mouth_teeth_data, 100046),
    'GeneralHealth' : (read_general_health_data, 100042),
    'Breathing' : (read_breathing_data, 100037),
    'Claudification' : (read_claudication_data, 100038),
    'GeneralPain' : (read_general_pain_data, 100048),
    'ChestPain' : (read_chest_pain_data, 100039),
    'CancerScreening' : (read_cancer_screening_data, 100040),
    'Medication': (read_medication_data, 100045),
    'Hearing' : (read_hearing_data, 100043),
    'Household' : (read_household_data, 100066),
    'MentalHealth' : (read_mental_health_data, 100060),
    'OtherSociodemographics' : (read_other_sociodemographics_data, 100067),
    'PhysicalActivityQuestionnaire' : (read_physical_activity_data, 100054),
    'SexualFactors' : (read_sexual_factors_data, 100056),
    'Sleep' : (read_sleep_data, 100057),
    'SocialSupport' : (read_social_support_data, 100061),
    'SunExposure' : (read_sun_exposure_data, 100055),
    'EarlyLifeFactors' : (read_early_life_factors_data, 100033),
    'Smoking' : (read_smoking_data, 3466),
    ## Biomarkers
    # Anthropometry
    'AnthropometryImpedance' : (read_anthropometry_impedance_data, 100008),
    'AnthropometryBodySize' : (read_anthropometry_body_size_data, 100010),
    # Arterial Stiffness
    'ArterialStiffness' : (read_arterial_stiffness_data, 100007),
    # Urine And Blood
    'BloodBiochemistry' : (read_blood_biomarkers_data, 17518),
    'BloodCount' : (read_blood_count_data, 100081),
    'UrineBiochemistry' : (read_urine_biomarkers_data, 100083),
    # BloodPressure
    'BloodPressure' : (read_blood_pressure_data, 100011),
    # Brain
    'BrainGreyMatterVolumes' : (read_grey_matter_volumes_data, 1101),
    'BrainSubcorticalVolumes' : (read_subcortical_volumes_data, 1102),
    'BraindMRIWeightedMeans' : (read_brain_dMRI_weighted_means_data, 135),
    # carotid_ultrasound
    'CarotidUltrasound' : (read_carotid_ultrasound_data, 101),
    # ECG
    'ECGAtRest' : (read_ecg_at_rest_data, 12657),
    # Eye
    'EyeAcuity' : (read_eye_acuity_data, 100017),
    'EyeAutorefraction' : (read_eye_autorefraction_data, 100014),
    'EyeIntraocularPressure' : (read_eye_intraocular_pressure_data, 100015),
    # Heart
    'HeartPWA' : (read_heart_PWA_data, 128),
    'HeartSize' : (read_heart_size_data, 133),
    # Spirometry
    'Spirometry' : (read_spirometry_data, 100020),
    # Hearing
    'HearingTest' : (read_hearing_test_data, 100049),
    # HandGripStrength :
    'HandGripStrength' : (read_hand_grip_strength_data, 100019),
    # Bone BoneDensitometryOfHeel :
    'BoneDensitometryOfHeel' : (read_bone_densitometry_data, 100018),
    # Cognitive
    'CognitiveReactionTime' : (read_reaction_time_data, 100032),
    'CognitiveMatrixPatternCompletion' : (read_matrix_pattern_completion_data, 501),
    'CognitiveTowerRearranging' : (read_tower_rearranging_data, 503),
    'CognitiveSymbolDigitSubstitution' : (read_symbol_digit_substitution_data, 502),
    'CognitivePairedAssociativeLearning' : (read_paired_associative_learning_data, 506),
    'CognitiveProspectiveMemory' : (read_prospective_memory_data, 100031),
    'CognitiveNumericMemory' : (read_numeric_memory_data, 100029),
    'CognitiveFluidIntelligence' : (read_fluid_intelligence_data, 100027),
    'CognitiveTrailMaking' : (read_trail_making_data, 505),
    'CognitivePairsMatching' : (read_pairs_matching_data, 100030),
    'PhysicalActivity' : (read_physical_activity_data, 'Sasha')
}

## Medical
medical_diagnoses_dict = dict(zip(['medical_diagnoses_%s' % letter for letter in ascii_uppercase], [(partial(read_medical_diagnoses_data, letter = letter), 41270) for letter in ascii_uppercase]))
map_envdataset_to_dataloader_and_field = {**map_envdataset_to_dataloader_and_field, **medical_diagnoses_dict}


def create_data(env_dataset, **kwargs):
    ## env_dataset is just name not a path at this point.
    if env_dataset not in map_envdataset_to_dataloader_and_field.keys():
        raise ValueError('Wrong dataset name ! ')
    else :
        dataloader, field = map_envdataset_to_dataloader_and_field[env_dataset]
        df = dataloader(**kwargs)
    df.to_csv(path_inputs_env + env_dataset + '.csv')

def load_sex_age_ethnicity_data(**kwargs):
    df_sex_age_ethnicity_eid = pd.read_csv('/n/groups/patel/samuel/sex_age_eid_ethnicity.csv').set_index('id')
    return df_sex_age_ethnicity_eid


def load_data_env(env_dataset, **kwargs):
    if 'medical_diagnoses' in env_dataset :
        path_env = path_inputs_env + env_dataset + '.csv'
        df = pd.read_csv(path_env)
        df_sex_age_ethnicity = pd.read_csv('/n/groups/patel/Alan/Aging/Medical_Images/data/data-features_instances.csv').drop(columns = ['Abdominal_images_quality', 'instance', 'outer_fold'])
        df_sex_age_ethnicity = df_sex_age_ethnicity.rename(columns = {'Age' : 'Age when attended assessment centre'})
        df = df_sex_age_ethnicity.merge(df, on = 'eid').set_index('id')
        return df
    else :
        env_dataset = env_dataset.replace('\\', '')
        use_inputed = True
        ## Find columns to read
        path_env = path_inputs_env + env_dataset + '.csv'
        cols_to_read = pd.read_csv(path_env, nrows = 2).set_index('id').columns
        ## Features + eid + id / without ethnicity + age + sex
        cols_to_read = [elem for elem in cols_to_read if elem not in ['Ethnicity', 'Age when attended assessment centre', 'Sex', 'eid', 'Unnamed: 0'] + ETHNICITY_COLS + ['Ethnicity.' + elem for elem in ETHNICITY_COLS]] + ['id']
        if use_inputed == True:
            df = pd.read_csv(path_input_env_inputed, usecols = cols_to_read).set_index('id')
        else :
            df = pd.read_csv(path_input_env, usecols = cols_to_read).set_index('id')
        df_sex_age_ethnicity = pd.read_csv('/n/groups/patel/Alan/Aging/Medical_Images/data/data-features_instances.csv').set_index('id').drop(columns = ['Abdominal_images_quality', 'instance', 'outer_fold'])
        df_sex_age_ethnicity = df_sex_age_ethnicity.rename(columns = {'Age' : 'Age when attended assessment centre'})
        #df_age_new =
        df = df.join(df_sex_age_ethnicity)
        return df

def load_cluster(env_dataset, target_dataset, **kwargs):
    df = pd.read_csv('/n/groups/patel/samuel/EWAS/AutomaticClusters/%s_%s.csv' % (env_dataset, target_dataset)).set_index('id')
    return df

def load_target_residuals(target_dataset, **kwargs):
    target_dataset =  target_dataset.replace('\\', '')
    Alan_residuals = pd.read_csv(path_target_residuals, usecols = [target_dataset, 'id']).set_index('id')
    Alan_residuals.columns = ['residuals']
    return Alan_residuals


def load_data(env_dataset, target_dataset, **kwargs):
    if 'Clusters' in env_dataset:
        df_env = load_cluster(env_dataset, target_dataset, **kwargs)
    else :
        df_env = load_data_env(env_dataset, **kwargs)
    print(df_env)
    df_residuals = load_target_residuals(target_dataset, **kwargs)
    print(df_residuals)
    final_df = df_env.join(df_residuals, how = 'outer')
    print(final_df)
    return final_df[~final_df['residuals'].isna()]


## Saving
def save_features_to_csv(cols, features_imp, organ_target, dataset_env, model_name):
    final_df = pd.DataFrame(data = {'features' : cols, 'weight' : features_imp})
    final_df.set_index('features').to_csv(path_features + '/' + 'FeatureImp_' + dataset_env + '_' + organ_target + '_' + model_name + '.csv')


def save_predictions_to_csv(predicts_df, step, organ_target, dataset_env, model_name, fold, best_params):
    hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
    if len(best_params) != 7:
        hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])

    filename = 'Predictions_' + dataset_env + '_' + organ_target + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '.csv'
    predicts_df.set_index('id').to_csv(path_predictions + '/' + filename)
