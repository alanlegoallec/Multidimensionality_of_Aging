# from .arterial_stiffness_processing import read_arterial_stiffness_data
# from .blood_pressure_processing import read_blood_pressure_data
# from .ecg_processing import read_ecg_at_rest_data
# from .spirometry_processing import read_spirometry_data
import pandas as pd
from .base_processing import path_inputs


# def read_arterial_and_bp_data(**kwargs):
#     df_1 = read_arterial_stiffness_data(**kwargs)
#     df_2 = read_blood_pressure_data(**kwargs).drop(columns = ['Age when attended assessment centre', 'Sex', 'eid'])
#     return df_1.join(df_2, how = 'inner')
#
# def read_cardiac_data(**kwargs):
#     df_1 = read_arterial_stiffness_data(**kwargs)
#     df_2 = read_blood_pressure_data(**kwargs).drop(columns = ['Age when attended assessment centre', 'Sex', 'eid'])
#     res = df_1.join(df_2, how = 'inner')
#     df_3 = read_ecg_at_rest_data(**kwargs).drop(columns = ['Age when attended assessment centre', 'Sex', 'eid'])
#     return res.join(df_3, how = 'inner')
#
# def read_spiro_and_arterial_and_bp_data(**kwargs):
#     df_1 = read_arterial_stiffness_data(**kwargs)
#     df_2 = read_blood_pressure_data(**kwargs).drop(columns = ['Age when attended assessment centre', 'Sex', 'eid'])
#     res = df_1.join(df_2, how = 'inner')
#     df_3 = read_spirometry_data(**kwargs).drop(columns = ['Age when attended assessment centre', 'Sex', 'eid'])
#     return res.join(df_3, how = 'inner')

def read_vascular_all_data(**kwargs):
    df_bp = pd.read_csv(path_inputs + 'BloodPressure.csv').set_index('id')
    df_carotid = pd.read_csv(path_inputs + 'CarotidUltrasound.csv').set_index('id')
    df_arterial_stiffness  = pd.read_csv(path_inputs + 'ArterialStiffness.csv').set_index('id')
    df = df_bp.join(df_carotid, rsuffix = '_rsuffix')
    df = df.join(df_arterial_stiffness, rsuffix = '_rsuffix')
    df = df[df.columns[~df.columns.str.contains('_rsuffix')]]
    return df.dropna()

def read_all_brain_and_cognitive(**kwars):
    df_cogn = pd.read_csv(path_inputs + 'CognitiveAllBiomarkers.csv').set_index('id')
    df_brain = pd.read_csv(path_inputs + 'BrainMRIAllBiomarkers.csv').set_index('id')
    df = df_cogn.join(df_brain, rsuffix = '_rsuffix')
    df = df[df.columns[~df.columns.str.contains('_rsuffix')]]
    return df.dropna()

def read_heart_MRI_data(**kwargs):
    df_pwa = pd.read_csv(path_inputs + 'HeartPWA.csv').set_index('id')
    df_size = pd.read_csv(path_inputs + 'HeartSize.csv').set_index('id')
    df = df_pwa.join(df_size, rsuffix = '_rsuffix')
    df = df[df.columns[~df.columns.str.contains('_rsuffix')]]
    return df.dropna()

def read_heart_all_data(**kwargs):
    df_pwa = pd.read_csv(path_inputs + 'HeartPWA.csv').set_index('id')
    df_size = pd.read_csv(path_inputs + 'HeartSize.csv').set_index('id')
    df_ecg = pd.read_csv(path_inputs + 'ECGAtRest.csv').set_index('id')
    df = df_pwa.join(df_size, rsuffix = '_rsuffix')
    df = df.join(df_ecg, rsuffix = '_rsuffix')
    df = df[df.columns[~df.columns.str.contains('_rsuffix')]]
    return df.dropna()

def read_musculoskeletal_data(**kwargs):
    df_imp = pd.read_csv(path_inputs + 'AnthropometryImpedance.csv').set_index('id')
    df_size = pd.read_csv(path_inputs + 'AnthropometryBodySize.csv').set_index('id')
    df_heel = pd.read_csv(path_inputs + 'BoneDensitometryOfHeel.csv').set_index('id')
    df_hand = pd.read_csv(path_inputs + 'HandGripStrength.csv').set_index('id')
    df = df_imp.join(df_size, rsuffix = '_rsuffix')
    df = df.join(df_heel, rsuffix = '_rsuffix')
    df = df.join(df_hand, rsuffix = '_rsuffix')
    df = df[df.columns[~df.columns.str.contains('_rsuffix')]]
    return df.dropna()


def read_biochemistry_data(**kwargs):
    df_blood = pd.read_csv(path_inputs + 'BloodBiochemistry.csv').set_index('id')
    df_urine = pd.read_csv(path_inputs + 'UrineBiochemistry.csv').set_index('id')
    df = df_blood.join(df_urine, rsuffix = '_rsuffix')
    df = df[df.columns[~df.columns.str.contains('_rsuffix')]]
    return df.dropna()
