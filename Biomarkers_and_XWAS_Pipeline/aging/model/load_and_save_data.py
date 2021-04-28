import pandas as pd
import glob
import numpy as np
from ..processing.base_processing import path_features , path_predictions, path_inputs, read_ethnicity_data
from ..processing.abdominal_composition_processing import read_abdominal_data
from ..processing.brain_processing import read_grey_matter_volumes_data, read_subcortical_volumes_data, read_brain_data, read_brain_dMRI_weighted_means_data
from ..processing.heart_processing import read_heart_data, read_heart_size_data, read_heart_PWA_data
from ..processing.body_composition_processing import read_body_composition_data
from ..processing.bone_composition_processing import read_bone_composition_data
from ..processing.ecg_processing import read_ecg_at_rest_data
from ..processing.anthropometry_processing import read_anthropometry_impedance_data, read_anthropometry_body_size_data, read_anthropometry_data
from ..processing.biochemestry_processing import read_blood_biomarkers_data, read_urine_biomarkers_data, read_blood_count_data, read_blood_data, read_urine_and_blood_data
from ..processing.eye_processing import read_eye_autorefraction_data, read_eye_acuity_data, read_eye_intraocular_pressure_data, read_eye_data
from ..processing.spirometry_processing import read_spirometry_data
from ..processing.blood_pressure_processing import read_blood_pressure_data
from ..processing.arterial_stiffness_processing import read_arterial_stiffness_data
from ..processing.mix_processing import read_vascular_all_data, read_all_brain_and_cognitive, read_heart_MRI_data, read_heart_all_data, read_biochemistry_data, read_musculoskeletal_data
from ..processing.carotid_ultrasound_processing import read_carotid_ultrasound_data
from ..processing.bone_densitometry_processing import read_bone_densitometry_data
from ..processing.hand_grip_strength_processing import read_hand_grip_strength_data
from ..processing.hearing_tests_processing import read_hearing_test_data
from ..processing.cognitive_tests_processing import read_reaction_time_data, read_matrix_pattern_completion_data, read_tower_rearranging_data, \
                                                    read_symbol_digit_substitution_data, read_paired_associative_learning_data, \
                                                    read_prospective_memory_data, read_numeric_memory_data, read_fluid_intelligence_data, read_trail_making_data , \
                                                    read_pairs_matching_data, read_all_cognitive_data
from ..processing.physical_activity_processing import read_physical_activity_data

map_dataset_to_field_and_dataloader = {
                    'BrainGreyMatterVolumes' : (1101, read_grey_matter_volumes_data),
                    'BrainSubcorticalVolumes': (1102, read_subcortical_volumes_data),
                    'BraindMRIWeightedMeans' : (135, read_brain_dMRI_weighted_means_data),
                    'BrainMRIAllBiomarkers' : (100, read_brain_data),
                    'CognitiveReactionTime' : (100032, read_reaction_time_data),
                    'CognitiveMatrixPatternCompletion' : (501, read_matrix_pattern_completion_data),
                    'CognitiveTowerRearranging' : (503, read_tower_rearranging_data),
                    'CognitiveSymbolDigitSubstitution' : (502, read_symbol_digit_substitution_data),
                    'CognitivePairedAssociativeLearning' : (506, read_paired_associative_learning_data),
                    'CognitiveProspectiveMemory' : (100031, read_prospective_memory_data),
                    'CognitiveNumericMemory' : (100029, read_numeric_memory_data),
                    'CognitiveFluidIntelligence' : (100027, read_fluid_intelligence_data),
                    'CognitiveTrailMaking' : (505, read_trail_making_data),
                    'CognitivePairsMatching' : (100030, read_pairs_matching_data),
                    'CognitiveAllBiomarkers' : ('Custom', read_all_cognitive_data),
                    'BrainAndCognitive' : ('Custom', read_all_brain_and_cognitive),


                    'EyeAutorefraction' : (100014, read_eye_autorefraction_data),
                    'EyeAcuity' : (100017, read_eye_acuity_data),
                    'EyeIntraocularPressure' : (100015, read_eye_intraocular_pressure_data),
                    'EyesAllBiomarkers' : (100013, read_eye_data),
                    # Hearing
                    'HearingTest' : (100049, read_hearing_test_data),
                    # Lungs
                    'Spirometry' :  (100020, read_spirometry_data),

                    # Vascular
                    'BloodPressure' : (100011, read_blood_pressure_data),
                    'CarotidUltrasound' : (101, read_carotid_ultrasound_data),
                    'ArterialStiffness' : (100007, read_arterial_stiffness_data),
                    'VascularAllBiomarkers' : ('Custom', read_vascular_all_data),
                    # Heart
                    'HeartAllBiomarkers' : ('Custom', read_heart_all_data),
                    'HeartSize' : (133, read_heart_size_data),
                    'HeartPWA' : (128, read_heart_PWA_data),
                    'HeartMRIAll' : ('Custom', read_heart_MRI_data),
                    'ECGAtRest' : (104, read_ecg_at_rest_data),

                    # Musculoskeletal
                    'AnthropometryImpedance' : (100008, read_anthropometry_impedance_data),
                    'AnthropometryBodySize' : (100010, read_anthropometry_body_size_data),
                    'BoneDensitometryOfHeel' : (100018, read_bone_densitometry_data),
                    'HandGripStrength' : (100019, read_hand_grip_strength_data),
                    'MusculoskeletalAllBiomarkers' : ('Custom', read_musculoskeletal_data),

                    #Biochemistry
                    'BloodBiochemistry' : (17518, read_blood_biomarkers_data),
                    'UrineBiochemistry' : (100083, read_urine_biomarkers_data),
                    'Biochemistry' : ('Custom', read_biochemistry_data),
                    #ImmuneSystem
                    'BloodCount' : (100081, read_blood_count_data),   # Need to do blood infection
                    'PhysicalActivity' : ('Custom', read_physical_activity_data),
                    #'Demographics' : ('Demographics',read_demographics_data)

                    }


dict_dataset_to_organ_and_view = {
    ## Brain
    'BrainGreyMatterVolumes' : ('Brain', 'MRI', 'GreyMatterVolumes'),
    'BrainSubcorticalVolumes': ('Brain', 'MRI', 'SubcorticalVolumes'),
    'BraindMRIWeightedMeans' : ('Brain', 'MRI', 'dMRIWeightedMeans'),
    'BrainMRIAllBiomarkers' : ('Brain', 'MRI', 'AllBiomarkers'),
    'CognitiveReactionTime' : ('Brain', 'Cognitive', 'ReactionTime'),
    'CognitiveMatrixPatternCompletion' : ('Brain', 'Cognitive', 'MatrixPatternCompletion'),
    'CognitiveTowerRearranging' : ('Brain', 'Cognitive', 'TowerRearranging'),
    'CognitiveSymbolDigitSubstitution' : ('Brain', 'Cognitive', 'SymbolDigitSubstitution'),
    'CognitivePairedAssociativeLearning' : ('Brain', 'Cognitive', 'PairedAssociativeLearning'),
    'CognitiveProspectiveMemory' : ('Brain', 'Cognitive', 'ProspectiveMemory'),
    'CognitiveNumericMemory' : ('Brain', 'Cognitive', 'NumericMemory'),
    'CognitiveFluidIntelligence' : ('Brain', 'Cognitive', 'FluidIntelligence'),
    'CognitiveTrailMaking' : ('Brain', 'Cognitive', 'TrailMaking'),
    'CognitivePairsMatching' : ('Brain', 'Cognitive', 'PairsMatching'),
    'CognitiveAllBiomarkers' : ('Brain', 'Cognitive', 'AllScalars'),
    'BrainAndCognitive' : ('Brain', 'All', 'Scalars'),
    ## Eyes
    'EyeAutorefraction' : ('Eyes', 'Autorefraction', 'Scalars'),
    'EyeAcuity' : ('Eyes', 'Acuity', 'Scalars'),
    'EyeIntraocularPressure' : ('Eyes', 'IntraocularPressure', 'Scalars'),
    'EyesAllBiomarkers' : ('Eyes', 'All', 'Scalars'),
    # Hearing
    'HearingTest' : ('Hearing', 'HearingTest', 'Scalars'),
    # Lungs
    'Spirometry' :  ('Lungs', 'Spirometry', 'Scalars'),
    # Vascular
    'BloodPressure' : ('Arterial', 'BloodPressure', 'Scalars'),
    'CarotidUltrasound' : ('Arterial', 'Carotids', 'Scalars'),
    'ArterialStiffness' : ('Arterial', 'PWA', 'Scalars'),
    'VascularAllBiomarkers' : ('Arterial', 'All', 'Scalars'),
    # Heart
    'HeartAllBiomarkers' : ('Heart', 'All', 'Scalars'),
    'HeartSize' : ('Heart', 'MRI', 'Size'),
    'HeartPWA' : ('Heart', 'MRI', 'PWA'),
    'HeartMRIAll' : ('Heart', 'MRI', 'AllScalars'),
    'ECGAtRest' : ('Heart', 'ECG', 'Scalars'),

    # Musculoskeletal
    'AnthropometryImpedance' : ('Musculoskeletal', 'Scalars', 'Impedance'),
    'AnthropometryBodySize' : ('Musculoskeletal', 'Scalars', 'Anthropometry'),
    'BoneDensitometryOfHeel' : ('Musculoskeletal', 'Scalars', 'HeelBoneDensitometry'),
    'HandGripStrength' : ('Musculoskeletal', 'Scalars', 'HandGripStrength'),
    'MusculoskeletalAllBiomarkers' : ('Musculoskeletal', 'Scalars', 'AllScalars'),

    #Biochemistry
    'BloodBiochemistry' : ('Biochemistry', 'Blood', 'Scalars'),
    'UrineBiochemistry' : ('Biochemistry', 'Urine', 'Scalars'),
    'Biochemistry' : ('Biochemistry', 'All', 'Scalars'),
    #ImmuneSystem
    'BloodCount' : ('ImmuneSystem', 'BloodCount', 'Scalars'),  # Need to do blood infection
    'PhysicalActivity' : ('PhysicalActivity', 'FullWeek', 'Scalars'),
    'Demographics' : ('Demographics', 'All', 'Scalars')
}

def load_data(dataset, **kwargs):
    if 'Cluster' in dataset :
        df = pd.read_csv(dataset).set_index('id')
        organ, view = 'Cluster', 'main'
    elif '/n' not in dataset:
        if dataset == 'Demographics':
            df = pd.read_csv('/n/groups/patel/samuel/sex_age_eid_ethnicity.csv').set_index('id')
        elif dataset == 'PhysicalActivity' :
            path_dataset = path_inputs + dataset + '.csv'
            df = pd.read_csv(path_dataset).set_index('id')
        ## TEST PA different nb of features
        elif dataset != 'PhysicalActivity' and 'PhysicalActivity'  in dataset :
            path_dataset = path_inputs + dataset + '.csv'
            df = pd.read_csv(path_dataset).set_index('id')
        else :
            path_dataset = path_inputs + dataset + '.csv'
            df = pd.read_csv(path_dataset).set_index('id')
            df_ethnicity_sex_age = pd.read_csv('/n/groups/patel/samuel/sex_age_eid_ethnicity.csv').set_index('id')
            df = df_ethnicity_sex_age.join(df, rsuffix = '_r')
            df = df[df.columns[~df.columns.str.contains('_r')]]
        organ, view, transformation = dict_dataset_to_organ_and_view[dataset]
    return df.dropna(), organ, view, transformation

def create_data(dataset, **kwargs):
    if dataset not in map_dataset_to_field_and_dataloader.keys():
        raise ValueError('Wrong dataset name ! ')
    else :
        field, dataloader = map_dataset_to_field_and_dataloader[dataset]
        df = dataloader(**kwargs)
    df.to_csv(path_inputs + dataset + '.csv')

def save_features_to_csv(cols, features_imp, target, organ, view, transformation, model_name, method):
    final_df = pd.DataFrame(data = {'features' : cols, 'weight' : features_imp})
    full_name = 'FeatureImp_'
    if method  == 'sd':
        full_name += 'sd_'
    elif method == 'mean':
        full_name += 'mean_'
    final_df.set_index('features').to_csv(path_features + '/' + full_name + target + '_' + organ + '_' + view + '_' + transformation + '_' + model_name + '.csv')

def save_predictions_to_csv(predicts_df, step, target, dataset, model_name, fold, best_params):
    hyper_parameters_name = '_'.join([str(elem) for elem in best_params])
    if len(best_params) != 7:
        hyper_parameters_name = hyper_parameters_name + '_' + '_'.join(['NA' for elem in range(7 - len(best_params))])

    filename = 'Predictions_' + target + '_' + dataset + '_main' +  '_raw' + '_' + model_name + '_' + hyper_parameters_name + '_' + str(fold) + '_' + step + '.csv'
    predicts_df.set_index('id').to_csv(path_predictions + '/' + filename)
