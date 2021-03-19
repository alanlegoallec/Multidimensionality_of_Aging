# LIBRARIES
# set up backend for ssh -x11 figures
import matplotlib

matplotlib.use('Agg')

# read and write
import os
import sys
import glob
import re
import fnmatch
import csv
import shutil
from datetime import datetime

# maths
import numpy as np
import pandas as pd
import math
import random

# miscellaneous
import warnings
import gc
import timeit

# sklearn
from sklearn.utils import resample
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, log_loss, roc_auc_score, accuracy_score, \
    f1_score, precision_score, recall_score, confusion_matrix, average_precision_score
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import KFold, PredefinedSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
# other metrics
from scipy.stats import pearsonr

# Other tools for ensemble models building (Samuel Diai's InnerCV class)
from hyperopt import fmin, tpe, space_eval, Trials, hp, STATUS_OK
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# CPUs
from multiprocessing import Pool
# GPUs
from GPUtil import GPUtil
# tensorflow
import tensorflow as tf
# keras
from keras_preprocessing.image import ImageDataGenerator, Iterator
from keras_preprocessing.image.utils import load_img, img_to_array, array_to_img
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, concatenate
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError, AUC, BinaryAccuracy, Precision, Recall, \
    TruePositives, FalsePositives, FalseNegatives, TrueNegatives
from tensorflow_addons.metrics import RSquare, F1Score

# Plots
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from bioinfokit import visuz

# Model's attention
from keract import get_activations, get_gradients_of_activations
from scipy.ndimage.interpolation import zoom

# Survival
from lifelines.utils import concordance_index

# Necessary to define MyCSVLogger
import collections
import csv
import io
import six
from tensorflow.python.lib.io import file_io
from tensorflow.python.util.compat import collections_abc
from tensorflow.keras.backend import eval

# Set display parameters
pd.set_option('display.max_rows', 200)


# CLASSES
class Basics:
    
    def __init__(self):
        # seeds for reproducibility
        self.seed = 0
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)
        
        # other parameters
        self.path_data = '../data/'
        self.folds = ['train', 'val', 'test']
        self.n_CV_outer_folds = 10
        self.outer_folds = [str(x) for x in list(range(self.n_CV_outer_folds))]
        self.modes = ['', '_sd', '_str']
        self.id_vars = ['id', 'eid', 'instance', 'outer_fold']
        self.instances = ['0', '1', '1.5', '1.51', '1.52', '1.53', '1.54', '2', '3']
        self.ethnicities_vars_forgot_Other = \
            ['Ethnicity.White', 'Ethnicity.British', 'Ethnicity.Irish', 'Ethnicity.White_Other', 'Ethnicity.Mixed',
             'Ethnicity.White_and_Black_Caribbean', 'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian',
             'Ethnicity.Mixed_Other', 'Ethnicity.Asian', 'Ethnicity.Indian', 'Ethnicity.Pakistani',
             'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other', 'Ethnicity.Black', 'Ethnicity.Caribbean',
             'Ethnicity.African', 'Ethnicity.Black_Other', 'Ethnicity.Chinese', 'Ethnicity.Other_ethnicity',
             'Ethnicity.Do_not_know', 'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA']
        self.ethnicities_vars = \
            ['Ethnicity.White', 'Ethnicity.British', 'Ethnicity.Irish', 'Ethnicity.White_Other', 'Ethnicity.Mixed',
             'Ethnicity.White_and_Black_Caribbean', 'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian',
             'Ethnicity.Mixed_Other', 'Ethnicity.Asian', 'Ethnicity.Indian', 'Ethnicity.Pakistani',
             'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other', 'Ethnicity.Black', 'Ethnicity.Caribbean',
             'Ethnicity.African', 'Ethnicity.Black_Other', 'Ethnicity.Chinese', 'Ethnicity.Other',
             'Ethnicity.Other_ethnicity', 'Ethnicity.Do_not_know', 'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA']
        self.demographic_vars = ['Age', 'Sex'] + self.ethnicities_vars
        self.names_model_parameters = ['target', 'organ', 'view', 'transformation', 'architecture', 'n_fc_layers',
                                       'n_fc_nodes', 'optimizer', 'learning_rate', 'weight_decay', 'dropout_rate',
                                       'data_augmentation_factor']
        self.targets_regression = ['Age']
        self.targets_binary = ['Sex']
        self.models_types = ['', '_bestmodels']
        self.dict_prediction_types = {'Age': 'regression', 'Sex': 'binary'}
        self.dict_side_predictors = {'Age': ['Sex'] + self.ethnicities_vars_forgot_Other,
                                     'Sex': ['Age'] + self.ethnicities_vars_forgot_Other}
        self.organs = ['Brain', 'Eyes', 'Arterial', 'Heart', 'Abdomen', 'Musculoskeletal']
        self.left_right_organs_views = ['Eyes_Fundus', 'Eyes_OCT', 'Arterial_Carotids', 'Musculoskeletal_Hips',
                                        'Musculoskeletal_Knees']
        self.dict_organs_to_views = {'Brain': ['MRI'],
                                     'Eyes': ['Fundus', 'OCT'],
                                     'Arterial': ['Carotids'],
                                     'Heart': ['MRI'],
                                     'Abdomen': ['Liver', 'Pancreas'],
                                     'Musculoskeletal': ['Spine', 'Hips', 'Knees', 'FullBody'],
                                     'PhysicalActivity': ['FullWeek']}
        self.dict_organsviews_to_transformations = \
            {'Brain_MRI': ['SagittalRaw', 'SagittalReference', 'CoronalRaw', 'CoronalReference', 'TransverseRaw',
                               'TransverseReference'],
             'Arterial_Carotids': ['Mixed', 'LongAxis', 'CIMT120', 'CIMT150', 'ShortAxis'],
             'Heart_MRI': ['2chambersRaw', '2chambersContrast', '3chambersRaw', '3chambersContrast', '4chambersRaw',
                           '4chambersContrast'],
             'Musculoskeletal_Spine': ['Sagittal', 'Coronal'],
             'Musculoskeletal_FullBody': ['Mixed', 'Figure', 'Skeleton', 'Flesh'],
             'PhysicalActivity_FullWeek': ['GramianAngularField1minDifference', 'GramianAngularField1minSummation',
                                           'MarkovTransitionField1min', 'RecurrencePlots1min']}
        self.dict_organsviews_to_transformations.update(dict.fromkeys(['Eyes_Fundus', 'Eyes_OCT'], ['Raw']))
        self.dict_organsviews_to_transformations.update(
            dict.fromkeys(['Abdomen_Liver', 'Abdomen_Pancreas'], ['Raw', 'Contrast']))
        self.dict_organsviews_to_transformations.update(
            dict.fromkeys(['Musculoskeletal_Hips', 'Musculoskeletal_Knees'], ['MRI']))
        self.organsviews_not_to_augment = []
        self.organs_instances23 = ['Brain', 'Eyes', 'Arterial', 'Heart', 'Abdomen', 'Musculoskeletal',
                                   'PhysicalActivity']
        self.organs_XWAS = \
            ['*', '*instances01', '*instances1.5x', '*instances23', 'Brain', 'BrainCognitive', 'BrainMRI', 'Eyes',
             'EyesFundus', 'EyesOCT', 'Hearing', 'Lungs', 'Arterial', 'ArterialPulseWaveAnalysis', 'ArterialCarotids',
             'Heart', 'HeartECG', 'HeartMRI', 'Abdomen', 'AbdomenLiver', 'AbdomenPancreas', 'Musculoskeletal',
             'MusculoskeletalSpine', 'MusculoskeletalHips', 'MusculoskeletalKnees', 'MusculoskeletalFullBody',
             'MusculoskeletalScalars', 'PhysicalActivity', 'Biochemistry', 'BiochemistryUrine', 'BiochemistryBlood',
             'ImmuneSystem']
        
        # Others
        if '/Users/Alan/' in os.getcwd():
            os.chdir('/Users/Alan/Desktop/Aging/Medical_Images/scripts/')
        else:
            os.chdir('/n/groups/patel/Alan/Aging/Medical_Images/scripts/')
        gc.enable()  # garbage collector
        warnings.filterwarnings('ignore')
    
    def _version_to_parameters(self, model_name):
        parameters = {}
        parameters_list = model_name.split('_')
        for i, parameter in enumerate(self.names_model_parameters):
            parameters[parameter] = parameters_list[i]
        if len(parameters_list) > 11:
            parameters['outer_fold'] = parameters_list[11]
        return parameters
    
    @staticmethod
    def _parameters_to_version(parameters):
        return '_'.join(parameters.values())
    
    @staticmethod
    def convert_string_to_boolean(string):
        if string == 'True':
            boolean = True
        elif string == 'False':
            boolean = False
        else:
            print('ERROR: string must be either \'True\' or \'False\'')
            sys.exit(1)
        return boolean


class Metrics(Basics):
    
    def __init__(self):
        # Parameters
        Basics.__init__(self)
        self.metrics_displayed_in_int = ['True-Positives', 'True-Negatives', 'False-Positives', 'False-Negatives']
        self.metrics_needing_classpred = ['F1-Score', 'Binary-Accuracy', 'Precision', 'Recall']
        self.dict_metrics_names_K = {'regression': ['RMSE'],  # For now, R-Square is buggy. Try again in a few months.
                                     'binary': ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Binary-Accuracy', 'Precision',
                                                'Recall', 'True-Positives', 'False-Positives', 'False-Negatives',
                                                'True-Negatives'],
                                     'multiclass': ['Categorical-Accuracy']}
        self.dict_metrics_names = {'regression': ['RMSE', 'MAE', 'R-Squared', 'Pearson-Correlation'],
                                   'binary': ['ROC-AUC', 'F1-Score', 'PR-AUC', 'Binary-Accuracy', 'Sensitivity',
                                              'Specificity', 'Precision', 'Recall', 'True-Positives', 'False-Positives',
                                              'False-Negatives', 'True-Negatives'],
                                   'multiclass': ['Categorical-Accuracy']}
        self.dict_losses_names = {'regression': 'MSE', 'binary': 'Binary-Crossentropy',
                                  'multiclass': 'categorical_crossentropy'}
        self.dict_main_metrics_names_K = {'Age': 'MAE', 'Sex': 'PR-AUC', 'imbalanced_binary_placeholder': 'PR-AUC'}
        self.dict_main_metrics_names = {'Age': 'R-Squared', 'Sex': 'ROC-AUC',
                                        'imbalanced_binary_placeholder': 'PR-AUC'}
        self.main_metrics_modes = {'loss': 'min', 'R-Squared': 'max', 'Pearson-Correlation': 'max', 'RMSE': 'min',
                                   'MAE': 'min', 'ROC-AUC': 'max', 'PR-AUC': 'max', 'F1-Score': 'max', 'C-Index': 'max',
                                   'C-Index-difference': 'max'}
        
        self.n_bootstrap_iterations = 1000
        
        def rmse(y_true, y_pred):
            return math.sqrt(mean_squared_error(y_true, y_pred))
        
        def sensitivity_score(y, pred):
            _, _, fn, tp = confusion_matrix(y, pred.round()).ravel()
            return tp / (tp + fn)
        
        def specificity_score(y, pred):
            tn, fp, _, _ = confusion_matrix(y, pred.round()).ravel()
            return tn / (tn + fp)
        
        def true_positives_score(y, pred):
            _, _, _, tp = confusion_matrix(y, pred.round()).ravel()
            return tp
        
        def false_positives_score(y, pred):
            _, fp, _, _ = confusion_matrix(y, pred.round()).ravel()
            return fp
        
        def false_negatives_score(y, pred):
            _, _, fn, _ = confusion_matrix(y, pred.round()).ravel()
            return fn
        
        def true_negatives_score(y, pred):
            tn, _, _, _ = confusion_matrix(y, pred.round()).ravel()
            return tn
        
        self.dict_metrics_sklearn = {'mean_squared_error': mean_squared_error,
                                     'mean_absolute_error': mean_absolute_error,
                                     'RMSE': rmse,
                                     'Pearson-Correlation': pearsonr,
                                     'R-Squared': r2_score,
                                     'Binary-Crossentropy': log_loss,
                                     'ROC-AUC': roc_auc_score,
                                     'F1-Score': f1_score,
                                     'PR-AUC': average_precision_score,
                                     'Binary-Accuracy': accuracy_score,
                                     'Sensitivity': sensitivity_score,
                                     'Specificity': specificity_score,
                                     'Precision': precision_score,
                                     'Recall': recall_score,
                                     'True-Positives': true_positives_score,
                                     'False-Positives': false_positives_score,
                                     'False-Negatives': false_negatives_score,
                                     'True-Negatives': true_negatives_score}
    
    def _bootstrap(self, data, function):
        results = []
        for i in range(self.n_bootstrap_iterations):
            data_i = resample(data, replace=True, n_samples=len(data.index))
            results.append(function(data_i['y'], data_i['pred']))
        return np.mean(results), np.std(results)


class PreprocessingMain(Basics):
    
    def __init__(self):
        Basics.__init__(self)
        self.data_raw = None
        self.data_features = None
        self.data_features_eids = None
    
    def _add_outer_folds(self):
        outer_folds_split = pd.read_csv(self.path_data + 'All_eids.csv')
        outer_folds_split.rename(columns={'fold': 'outer_fold'}, inplace=True)
        outer_folds_split['eid'] = outer_folds_split['eid'].astype('str')
        outer_folds_split['outer_fold'] = outer_folds_split['outer_fold'].astype('str')
        outer_folds_split.set_index('eid', inplace=True)
        self.data_raw = self.data_raw.join(outer_folds_split)
    
    def _impute_missing_ecg_instances(self):
        data_ecgs = pd.read_csv('/n/groups/patel/Alan/Aging/TimeSeries/scripts/age_analysis/missing_samples.csv')
        data_ecgs['eid'] = data_ecgs['eid'].astype(str)
        data_ecgs['instance'] = data_ecgs['instance'].astype(str)
        for _, row in data_ecgs.iterrows():
            self.data_raw.loc[row['eid'], 'Date_attended_center_' + row['instance']] = row['observation_date']
    
    def _add_physicalactivity_instances(self):
        data_pa = pd.read_csv(
            '/n/groups/patel/Alan/Aging/TimeSeries/series/PhysicalActivity/90001/features/PA_visit_date.csv')
        data_pa['eid'] = data_pa['eid'].astype(str)
        data_pa.set_index('eid', drop=False, inplace=True)
        data_pa.index.name = 'column_names'
        self.data_raw = self.data_raw.merge(data_pa, on=['eid'], how='outer')
        self.data_raw.set_index('eid', drop=False, inplace=True)
    
    def _compute_sex(self):
        # Use genetic sex when available
        self.data_raw['Sex_genetic'][self.data_raw['Sex_genetic'].isna()] = \
            self.data_raw['Sex'][self.data_raw['Sex_genetic'].isna()]
        self.data_raw.drop(['Sex'], axis=1, inplace=True)
        self.data_raw.rename(columns={'Sex_genetic': 'Sex'}, inplace=True)
        self.data_raw.dropna(subset=['Sex'], inplace=True)
    
    def _compute_age(self):
        # Recompute age with greater precision by leveraging the month of birth
        self.data_raw['Year_of_birth'] = self.data_raw['Year_of_birth'].astype(int)
        self.data_raw['Month_of_birth'] = self.data_raw['Month_of_birth'].astype(int)
        self.data_raw['Date_of_birth'] = self.data_raw.apply(
            lambda row: datetime(row.Year_of_birth, row.Month_of_birth, 15), axis=1)
        for i in self.instances:
            self.data_raw['Date_attended_center_' + i] = \
                self.data_raw['Date_attended_center_' + i].apply(
                    lambda x: pd.NaT if pd.isna(x) else datetime.strptime(x, '%Y-%m-%d'))
            self.data_raw['Age_' + i] = self.data_raw['Date_attended_center_' + i] - self.data_raw['Date_of_birth']
            self.data_raw['Age_' + i] = self.data_raw['Age_' + i].dt.days / 365.25
            self.data_raw.drop(['Date_attended_center_' + i], axis=1, inplace=True)
        self.data_raw.drop(['Year_of_birth', 'Month_of_birth', 'Date_of_birth'], axis=1, inplace=True)
        self.data_raw.dropna(how='all', subset=['Age_0', 'Age_1', 'Age_1.5', 'Age_1.51', 'Age_1.52', 'Age_1.53',
                                                'Age_1.54', 'Age_2', 'Age_3'], inplace=True)
    
    def _encode_ethnicity(self):
        # Fill NAs for ethnicity on instance 0 if available in other instances
        eids_missing_ethnicity = self.data_raw['eid'][self.data_raw['Ethnicity'].isna()]
        for eid in eids_missing_ethnicity:
            sample = self.data_raw.loc[eid, :]
            if not math.isnan(sample['Ethnicity_1']):
                self.data_raw.loc[eid, 'Ethnicity'] = self.data_raw.loc[eid, 'Ethnicity_1']
            elif not math.isnan(sample['Ethnicity_2']):
                self.data_raw.loc[eid, 'Ethnicity'] = self.data_raw.loc[eid, 'Ethnicity_2']
        self.data_raw.drop(['Ethnicity_1', 'Ethnicity_2'], axis=1, inplace=True)
        
        # One hot encode ethnicity
        dict_ethnicity_codes = {'1': 'Ethnicity.White', '1001': 'Ethnicity.British', '1002': 'Ethnicity.Irish',
                                '1003': 'Ethnicity.White_Other',
                                '2': 'Ethnicity.Mixed', '2001': 'Ethnicity.White_and_Black_Caribbean',
                                '2002': 'Ethnicity.White_and_Black_African',
                                '2003': 'Ethnicity.White_and_Asian', '2004': 'Ethnicity.Mixed_Other',
                                '3': 'Ethnicity.Asian', '3001': 'Ethnicity.Indian', '3002': 'Ethnicity.Pakistani',
                                '3003': 'Ethnicity.Bangladeshi', '3004': 'Ethnicity.Asian_Other',
                                '4': 'Ethnicity.Black', '4001': 'Ethnicity.Caribbean', '4002': 'Ethnicity.African',
                                '4003': 'Ethnicity.Black_Other',
                                '5': 'Ethnicity.Chinese',
                                '6': 'Ethnicity.Other_ethnicity',
                                '-1': 'Ethnicity.Do_not_know',
                                '-3': 'Ethnicity.Prefer_not_to_answer',
                                '-5': 'Ethnicity.NA'}
        self.data_raw['Ethnicity'] = self.data_raw['Ethnicity'].fillna(-5).astype(int).astype(str)
        ethnicities = pd.get_dummies(self.data_raw['Ethnicity'])
        self.data_raw.drop(['Ethnicity'], axis=1, inplace=True)
        ethnicities.rename(columns=dict_ethnicity_codes, inplace=True)
        ethnicities['Ethnicity.White'] = ethnicities['Ethnicity.White'] + ethnicities['Ethnicity.British'] + \
                                         ethnicities['Ethnicity.Irish'] + ethnicities['Ethnicity.White_Other']
        ethnicities['Ethnicity.Mixed'] = ethnicities['Ethnicity.Mixed'] + \
                                         ethnicities['Ethnicity.White_and_Black_Caribbean'] + \
                                         ethnicities['Ethnicity.White_and_Black_African'] + \
                                         ethnicities['Ethnicity.White_and_Asian'] + \
                                         ethnicities['Ethnicity.Mixed_Other']
        ethnicities['Ethnicity.Asian'] = ethnicities['Ethnicity.Asian'] + ethnicities['Ethnicity.Indian'] + \
                                         ethnicities['Ethnicity.Pakistani'] + ethnicities['Ethnicity.Bangladeshi'] + \
                                         ethnicities['Ethnicity.Asian_Other']
        ethnicities['Ethnicity.Black'] = ethnicities['Ethnicity.Black'] + ethnicities['Ethnicity.Caribbean'] + \
                                         ethnicities['Ethnicity.African'] + ethnicities['Ethnicity.Black_Other']
        ethnicities['Ethnicity.Other'] = ethnicities['Ethnicity.Other_ethnicity'] + \
                                         ethnicities['Ethnicity.Do_not_know'] + \
                                         ethnicities['Ethnicity.Prefer_not_to_answer'] + \
                                         ethnicities['Ethnicity.NA']
        self.data_raw = self.data_raw.join(ethnicities)
    
    def generate_data(self):
        # Preprocessing
        dict_UKB_fields_to_names = {'34-0.0': 'Year_of_birth', '52-0.0': 'Month_of_birth',
                                    '53-0.0': 'Date_attended_center_0', '53-1.0': 'Date_attended_center_1',
                                    '53-2.0': 'Date_attended_center_2', '53-3.0': 'Date_attended_center_3',
                                    '31-0.0': 'Sex', '22001-0.0': 'Sex_genetic', '21000-0.0': 'Ethnicity',
                                    '21000-1.0': 'Ethnicity_1', '21000-2.0': 'Ethnicity_2',
                                    '22414-2.0': 'Abdominal_images_quality'}
        self.data_raw = pd.read_csv('/n/groups/patel/uk_biobank/project_52887_41230/ukb41230.csv',
                                    usecols=['eid', '31-0.0', '22001-0.0', '21000-0.0', '21000-1.0', '21000-2.0',
                                             '34-0.0', '52-0.0', '53-0.0', '53-1.0', '53-2.0', '53-3.0', '22414-2.0'])
        
        # Formatting
        self.data_raw.rename(columns=dict_UKB_fields_to_names, inplace=True)
        self.data_raw['eid'] = self.data_raw['eid'].astype(str)
        self.data_raw.set_index('eid', drop=False, inplace=True)
        self.data_raw.index.name = 'column_names'
        self._add_outer_folds()
        self._impute_missing_ecg_instances()
        self._add_physicalactivity_instances()
        self._compute_sex()
        self._compute_age()
        self._encode_ethnicity()
        
        # Concatenate the data from the different instances
        self.data_features = None
        for i in self.instances:
            print('Preparing the samples for instance ' + i)
            df_i = self.data_raw[['eid', 'outer_fold', 'Age_' + i, 'Sex'] + self.ethnicities_vars +
                                 ['Abdominal_images_quality']].dropna(subset=['Age_' + i])
            print(str(len(df_i.index)) + ' samples found in instance ' + i)
            df_i.rename(columns={'Age_' + i: 'Age'}, inplace=True)
            df_i['instance'] = i
            df_i['id'] = df_i['eid'] + '_' + df_i['instance']
            df_i = df_i[self.id_vars + self.demographic_vars + ['Abdominal_images_quality']]
            if i != '2':
                df_i['Abdominal_images_quality'] = np.nan  # not defined for instance 3, not relevant for instances 0, 1
            if self.data_features is None:
                self.data_features = df_i
            else:
                self.data_features = self.data_features.append(df_i)
            print('The size of the full concatenated dataframe is now ' + str(len(self.data_features.index)))
        
        # Save age as a float32 instead of float64
        self.data_features['Age'] = np.float32(self.data_features['Age'])
        
        # Shuffle the rows before saving the dataframe
        self.data_features = self.data_features.sample(frac=1)
        
        # Generate dataframe for eids pipeline as opposed to instances pipeline
        self.data_features_eids = self.data_features[self.data_features.instance == '0']
        self.data_features_eids['instance'] = '*'
        self.data_features_eids['id'] = [ID.replace('_0', '_*') for ID in self.data_features_eids['id'].values]
    
    def save_data(self):
        self.data_features.to_csv(self.path_data + 'data-features_instances.csv', index=False)
        self.data_features_eids.to_csv(self.path_data + 'data-features_eids.csv', index=False)


class PreprocessingImagesIDs(Basics):
    
    def __init__(self):
        Basics.__init__(self)
        # Instances 2 and 3 datasets (most medical images, mostly medical images)
        self.instances23_eids = None
        self.HEART_EIDs = None
        self.heart_eids = None
        self.FOLDS_23_EIDS = None
    
    def _load_23_eids(self):
        data_features = pd.read_csv(self.path_data + 'data-features_instances.csv')
        images_eids = data_features['eid'][data_features['instance'].isin([2, 3])]
        self.images_eids = list(set(images_eids))
    
    def _load_heart_eids(self):
        # IDs already used in Heart videos
        HEART_EIDS = {}
        heart_eids = []
        for i in range(10):
            # Important: The i's data fold is used as *validation* fold for outer fold i.
            data_i = pd.read_csv(
                "/n/groups/patel/JbProst/Heart/Data/FoldsAugmented/data-features_Heart_20208_Augmented_Age_val_" + str(
                    i) + ".csv")
            HEART_EIDS[i] = list(set([int(str(ID)[:7]) for ID in data_i['eid']]))
            heart_eids = heart_eids + HEART_EIDS[i]
        self.HEART_EIDS = HEART_EIDS
        self.heart_eids = heart_eids
    
    def _split_23_eids_folds(self):
        self._load_23_eids()
        self._load_heart_eids()
        # List extra images ids, and split them between the different folds.
        extra_eids = [eid for eid in self.images_eids if eid not in self.heart_eids]
        random.shuffle(extra_eids)
        n_samples = len(extra_eids)
        n_samples_by_fold = n_samples / self.n_CV_outer_folds
        FOLDS_EXTRAEIDS = {}
        FOLDS_EIDS = {}
        for outer_fold in self.outer_folds:
            FOLDS_EXTRAEIDS[outer_fold] = \
                extra_eids[int((int(outer_fold)) * n_samples_by_fold):int((int(outer_fold) + 1) * n_samples_by_fold)]
            FOLDS_EIDS[outer_fold] = self.HEART_EIDS[int(outer_fold)] + FOLDS_EXTRAEIDS[outer_fold]
        self.FOLDS_23_EIDS = FOLDS_EIDS
    
    def _save_23_eids_folds(self):
        for outer_fold in self.outer_folds:
            with open(self.path_data + 'instances23_eids_' + outer_fold + '.csv', 'w', newline='') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                wr.writerow(self.FOLDS_23_EIDS[outer_fold])
    
    def generate_eids_splits(self):
        print("Generating eids split for organs on instances 2 and 3")
        self._split_23_eids_folds()
        self._save_23_eids_folds()


class PreprocessingFolds(Metrics):
    """
    Gather all the hyperparameters of the algorithm
    """
    def __init__(self, target, organ, regenerate_data):
        Metrics.__init__(self)
        self.target = target
        self.organ = organ
        self.list_ids_per_view_transformation = None
        
        # Check if these folds have already been generated
        if not regenerate_data:
            if len(glob.glob(self.path_data + 'data-features_' + organ + '_*_' + target + '_*.csv')) > 0:
                print("Error: The files already exist! Either change regenerate_data to True or delete the previous"
                      " version.")
                sys.exit(1)
        
        self.side_predictors = self.dict_side_predictors[target]
        self.variables_to_normalize = self.side_predictors
        if target in self.targets_regression:
            self.variables_to_normalize.append(target)
        self.dict_image_quality_col = {'Liver': 'Abdominal_images_quality'}
        self.dict_image_quality_col.update(
            dict.fromkeys(['Brain', 'Eyes', 'Arterial', 'Heart', 'Abdomen', 'Musculoskeletal', 'PhysicalActivity'],
                          None))
        self.image_quality_col = self.dict_image_quality_col[organ]
        self.views = self.dict_organs_to_views[organ]
        self.list_ids = None
        self.list_ids_per_view = {}
        self.data = None
        self.EIDS = None
        self.EIDS_per_view = {'train': {}, 'val': {}, 'test': {}}
        self.data_fold = None
    
    def _get_list_ids(self):
        self.list_ids_per_view_transformation = {}
        list_ids = []
        # if different views are available, take the union of the ids
        for view in self.views:
            self.list_ids_per_view_transformation[view] = {}
            for transformation in self.dict_organsviews_to_transformations[self.organ + '_' + view]:
                list_ids_transformation = []
                path = '../images/' + self.organ + '/' + view + '/' + transformation + '/'
                # for paired organs, take the unions of the ids available on the right and the left sides
                if self.organ + '_' + view in self.left_right_organs_views:
                    for side in ['right', 'left']:
                        list_ids_transformation += os.listdir(path + side + '/')
                    list_ids_transformation = np.unique(list_ids_transformation).tolist()
                else:
                    list_ids_transformation += os.listdir(path)
                self.list_ids_per_view_transformation[view][transformation] = \
                    [im.replace('.jpg', '') for im in list_ids_transformation]
                list_ids += self.list_ids_per_view_transformation[view][transformation]
        self.list_ids = np.unique(list_ids).tolist()
        self.list_ids.sort()
    
    def _filter_and_format_data(self):
        """
        Clean the data before it can be split between the rows
        """
        cols_data = self.id_vars + self.demographic_vars
        if self.image_quality_col is not None:
            cols_data.append(self.dict_image_quality_col[self.organ])
        data = pd.read_csv(self.path_data + 'data-features_instances.csv', usecols=cols_data)
        data.rename(columns={self.dict_image_quality_col[self.organ]: 'Data_quality'}, inplace=True)
        for col_name in self.id_vars:
            data[col_name] = data[col_name].astype(str)
        data.set_index('id', drop=False, inplace=True)
        if self.image_quality_col is not None:
            data = data[data['Data_quality'] != np.nan]
            data.drop('Data_quality', axis=1, inplace=True)
        # get rid of samples with NAs
        data.dropna(inplace=True)
        # list the samples' ids for which images are available
        data = data.loc[self.list_ids]
        self.data = data
    
    def _split_data(self):
        # Generate the data for each outer_fold
        for i, outer_fold in enumerate(self.outer_folds):
            of_val = outer_fold
            of_test = str((int(outer_fold) + 1) % len(self.outer_folds))
            DATA = {
                'train': self.data[~self.data['outer_fold'].isin([of_val, of_test])],
                'val': self.data[self.data['outer_fold'] == of_val],
                'test': self.data[self.data['outer_fold'] == of_test]
            }
            
            # Generate the data for the different views and transformations
            for view in self.views:
                for transformation in self.dict_organsviews_to_transformations[self.organ + '_' + view]:
                    print('Splitting data for view ' + view + ', and transformation ' + transformation)
                    DF = {}
                    for fold in self.folds:
                        idx = DATA[fold]['id'].isin(self.list_ids_per_view_transformation[view][transformation]).values
                        DF[fold] = DATA[fold].iloc[idx, :]
                    
                    # compute values for scaling of variables
                    normalizing_values = {}
                    for var in self.variables_to_normalize:
                        var_mean = DF['train'][var].mean()
                        if len(DF['train'][var].unique()) < 2:
                            print('Variable ' + var + ' has a single value in fold ' + outer_fold +
                                  '. Using 1 as std for normalization.')
                            var_std = 1
                        else:
                            var_std = DF['train'][var].std()
                        normalizing_values[var] = {'mean': var_mean, 'std': var_std}

                    # normalize the variables
                    for fold in self.folds:
                        for var in self.variables_to_normalize:
                            DF[fold][var + '_raw'] = DF[fold][var]
                            DF[fold][var] = (DF[fold][var] - normalizing_values[var]['mean']) \
                                             / normalizing_values[var]['std']

                        # report issue if NAs were detected (most likely comes from a sample whose id did not match)
                        n_mismatching_samples = DF[fold].isna().sum().max()
                        if n_mismatching_samples > 0:
                            print(DF[fold][DF[fold].isna().any(axis=1)])
                            print('/!\\ WARNING! ' + str(n_mismatching_samples) + ' ' + fold + ' images ids out of ' +
                                  str(len(DF[fold].index)) + ' did not match the dataframe!')
                        
                        # save the data
                        DF[fold].to_csv(self.path_data + 'data-features_' + self.organ + '_' + view + '_' +
                                         transformation + '_' + self.target + '_' + fold + '_' + outer_fold + '.csv',
                                         index=False)
                        print('For outer_fold ' + outer_fold + ', the ' + fold + ' fold has a sample size of ' +
                              str(len(DF[fold].index)))
    
    def generate_folds(self):
        self._get_list_ids()
        self._filter_and_format_data()
        self._split_data()


class PreprocessingSurvival(Basics):
    
    def __init__(self):
        Basics.__init__(self)
        self.data_raw = None
        self.data_features = None
        self.data_features_eids = None
        self.survival_vars = ['FollowUpTime', 'Death']
    
    def _preprocessing(self):
        usecols = ['eid', '40000-0.0', '34-0.0', '52-0.0', '53-0.0', '53-1.0', '53-2.0', '53-3.0']
        self.data_raw = pd.read_csv('/n/groups/patel/uk_biobank/project_52887_41230/ukb41230.csv', usecols=usecols)
        dict_UKB_fields_to_names = {'40000-0.0': 'FollowUpDate', '34-0.0': 'Year_of_birth', '52-0.0': 'Month_of_birth',
                                    '53-0.0': 'Date_attended_center_0', '53-1.0': 'Date_attended_center_1',
                                    '53-2.0': 'Date_attended_center_2', '53-3.0': 'Date_attended_center_3'}
        self.data_raw.rename(columns=dict_UKB_fields_to_names, inplace=True)
        self.data_raw['eid'] = self.data_raw['eid'].astype(str)
        self.data_raw.set_index('eid', drop=False, inplace=True)
        self.data_raw.index.name = 'column_names'
        # Format survival data
        self.data_raw['Death'] = ~self.data_raw['FollowUpDate'].isna()
        self.data_raw['FollowUpDate'][self.data_raw['FollowUpDate'].isna()] = '2020-04-27'
        self.data_raw['FollowUpDate'] = self.data_raw['FollowUpDate'].apply(
            lambda x: pd.NaT if pd.isna(x) else datetime.strptime(x, '%Y-%m-%d'))
        assert ('FollowUpDate.1' not in self.data_raw.columns)
    
    def _add_physicalactivity_instances(self):
        data_pa = pd.read_csv(
            '/n/groups/patel/Alan/Aging/TimeSeries/series/PhysicalActivity/90001/features/PA_visit_date.csv')
        data_pa['eid'] = data_pa['eid'].astype(str)
        data_pa.set_index('eid', drop=False, inplace=True)
        data_pa.index.name = 'column_names'
        self.data_raw = self.data_raw.merge(data_pa, on=['eid'], how='outer')
        self.data_raw.set_index('eid', drop=False, inplace=True)
    
    def _compute_age(self):
        # Recompute age with greater precision by leveraging the month of birth
        self.data_raw.dropna(subset=['Year_of_birth'], inplace=True)
        self.data_raw['Year_of_birth'] = self.data_raw['Year_of_birth'].astype(int)
        self.data_raw['Month_of_birth'] = self.data_raw['Month_of_birth'].astype(int)
        self.data_raw['Date_of_birth'] = self.data_raw.apply(
            lambda row: datetime(row.Year_of_birth, row.Month_of_birth, 15), axis=1)
        for i in self.instances:
            self.data_raw['Date_attended_center_' + i] = self.data_raw['Date_attended_center_' + i].apply(
                lambda x: pd.NaT if pd.isna(x) else datetime.strptime(x, '%Y-%m-%d'))
            self.data_raw['Age_' + i] = self.data_raw['Date_attended_center_' + i] - self.data_raw['Date_of_birth']
            self.data_raw['Age_' + i] = self.data_raw['Age_' + i].dt.days / 365.25
            self.data_raw['FollowUpTime_' + i] = self.data_raw['FollowUpDate'] - self.data_raw[
                'Date_attended_center_' + i]
            self.data_raw['FollowUpTime_' + i] = self.data_raw['FollowUpTime_' + i].dt.days / 365.25
            self.data_raw.drop(['Date_attended_center_' + i], axis=1, inplace=True)
        
        self.data_raw.drop(['Year_of_birth', 'Month_of_birth', 'Date_of_birth', 'FollowUpDate'], axis=1, inplace=True)
        self.data_raw.dropna(how='all', subset=['Age_0', 'Age_1', 'Age_1.5', 'Age_1.51', 'Age_1.52', 'Age_1.53',
                                                'Age_1.54', 'Age_2', 'Age_3'], inplace=True)
    
    def _concatenate_instances(self):
        self.data_features = None
        for i in self.instances:
            print('Preparing the samples for instance ' + i)
            df_i = self.data_raw.dropna(subset=['Age_' + i])
            print(str(len(df_i.index)) + ' samples found in instance ' + i)
            dict_names = {}
            features = ['Age', 'FollowUpTime']
            
            for feature in features:
                dict_names[feature + '_' + i] = feature
            self.dict_names = dict_names
            df_i.rename(columns=dict_names, inplace=True)
            df_i['instance'] = i
            df_i['id'] = df_i['eid'] + '_' + df_i['instance']
            df_i = df_i[['id', 'eid', 'instance'] + self.survival_vars]
            if self.data_features is None:
                self.data_features = df_i
            else:
                self.data_features = self.data_features.append(df_i)
            print('The size of the full concatenated dataframe is now ' + str(len(self.data_features.index)))
        
        # Add * instance for eids
        survival_eids = self.data_features[self.data_features['instance'] == '0']
        survival_eids['instance'] = '*'
        survival_eids['id'] = survival_eids['eid'] + '_' + survival_eids['instance']
        self.data_features = self.data_features.append(survival_eids)
    
    def generate_data(self):
        # Formatting
        self._preprocessing()
        self._add_physicalactivity_instances()
        self._compute_age()
        self._concatenate_instances()
        
        # save data
        self.data_features.to_csv('../data/data_survival.csv', index=False)


class MyImageDataGenerator(Basics, Sequence, ImageDataGenerator):
    
    def __init__(self, target=None, organ=None, view=None, data_features=None, n_samples_per_subepoch=None,
                 batch_size=None, training_mode=None, side_predictors=None, dir_images=None, images_width=None,
                 images_height=None, data_augmentation=False, data_augmentation_factor=None, seed=None):
        # Parameters
        Basics.__init__(self)
        self.target = target
        if target in self.targets_regression:
            self.labels = data_features[target]
        else:
            self.labels = data_features[target + '_raw']
        self.organ = organ
        self.view = view
        self.training_mode = training_mode
        self.data_features = data_features
        self.list_ids = data_features.index.values
        self.batch_size = batch_size
        # for paired organs, take twice fewer ids (two images for each id), and add organ_side as side predictor
        if organ + '_' + view in self.left_right_organs_views:
            self.data_features['organ_side'] = np.nan
            self.n_ids_batch = batch_size // 2
        else:
            self.n_ids_batch = batch_size
        if self.training_mode & (n_samples_per_subepoch is not None):  # during training, 1 epoch = number of samples
            self.steps = math.ceil(n_samples_per_subepoch / batch_size)
        else:  # during prediction and other tasks, an epoch is defined as all the samples being seen once and only once
            self.steps = math.ceil(len(self.list_ids) / self.n_ids_batch)
        # learning_rate_patience
        if n_samples_per_subepoch is not None:
            self.n_subepochs_per_epoch = math.ceil(len(self.data_features.index) / n_samples_per_subepoch)
        # initiate the indices and shuffle the ids
        self.shuffle = training_mode  # Only shuffle if the model is being trained. Otherwise no need.
        self.indices = np.arange(len(self.list_ids))
        self.idx_end = 0  # Keep track of last indice to permute indices accordingly at the end of epoch.
        if self.shuffle:
            np.random.shuffle(self.indices)
        # Input for side NN and CNN
        self.side_predictors = side_predictors
        self.dir_images = dir_images
        self.images_width = images_width
        self.images_height = images_height
        # Data augmentation
        self.data_augmentation = data_augmentation
        self.data_augmentation_factor = data_augmentation_factor
        self.seed = seed
        # Parameters for data augmentation: (rotation range, width shift range, height shift range, zoom range)
        self.augmentation_parameters = \
            pd.DataFrame(index=['Brain_MRI', 'Eyes_Fundus', 'Eyes_OCT', 'Arterial_Carotids', 'Heart_MRI',
                                'Abdomen_Liver', 'Abdomen_Pancreas', 'Musculoskeletal_Spine', 'Musculoskeletal_Hips',
                                'Musculoskeletal_Knees', 'Musculoskeletal_FullBody', 'PhysicalActivity_FullWeek',
                                'PhysicalActivity_Walking'],
                         columns=['rotation', 'width_shift', 'height_shift', 'zoom'])
        self.augmentation_parameters.loc['Brain_MRI', :] = [10, 0.05, 0.1, 0.0]
        self.augmentation_parameters.loc['Eyes_Fundus', :] = [20, 0.02, 0.02, 0]
        self.augmentation_parameters.loc['Eyes_OCT', :] = [30, 0.1, 0.2, 0]
        self.augmentation_parameters.loc[['Arterial_Carotids'], :] = [0, 0.2, 0.0, 0.0]
        self.augmentation_parameters.loc[['Heart_MRI', 'Abdomen_Liver', 'Abdomen_Pancreas',
                                          'Musculoskeletal_Spine'], :] = [10, 0.1, 0.1, 0.0]
        self.augmentation_parameters.loc[['Musculoskeletal_Hips', 'Musculoskeletal_Knees'], :] = [10, 0.1, 0.1, 0.1]
        self.augmentation_parameters.loc[['Musculoskeletal_FullBody'], :] = [10, 0.05, 0.02, 0.0]
        self.augmentation_parameters.loc[['PhysicalActivity_FullWeek'], :] = [0, 0, 0, 0.0]
        organ_view = organ + '_' + view
        ImageDataGenerator.__init__(self, rescale=1. / 255.,
                                    rotation_range=self.augmentation_parameters.loc[organ_view, 'rotation'],
                                    width_shift_range=self.augmentation_parameters.loc[organ_view, 'width_shift'],
                                    height_shift_range=self.augmentation_parameters.loc[organ_view, 'height_shift'],
                                    zoom_range=self.augmentation_parameters.loc[organ_view, 'zoom'])
    
    def __len__(self):
        return self.steps
    
    def on_epoch_end(self):
        _ = gc.collect()
        self.indices = np.concatenate([self.indices[self.idx_end:], self.indices[:self.idx_end]])
    
    def _generate_image(self, path_image):
        img = load_img(path_image, target_size=(self.images_width, self.images_height), color_mode='rgb')
        Xi = img_to_array(img)
        if hasattr(img, 'close'):
            img.close()
        if self.data_augmentation:
            params = self.get_random_transform(Xi.shape)
            Xi = self.apply_transform(Xi, params)
        Xi = self.standardize(Xi)
        return Xi
    
    def _data_generation(self, list_ids_batch):
        # initialize empty matrices
        n_samples_batch = min(len(list_ids_batch), self.batch_size)
        X = np.empty((n_samples_batch, self.images_width, self.images_height, 3)) * np.nan
        x = np.empty((n_samples_batch, len(self.side_predictors))) * np.nan
        y = np.empty((n_samples_batch, 1)) * np.nan
        # fill the matrices sample by sample
        for i, ID in enumerate(list_ids_batch):
            y[i] = self.labels[ID]
            x[i] = self.data_features.loc[ID, self.side_predictors]
            if self.organ + '_' + self.view in self.left_right_organs_views:
                if i % 2 == 0:
                    path = self.dir_images + 'right/'
                    x[i][-1] = 0
                else:
                    path = self.dir_images + 'left/'
                    x[i][-1] = 1
                if not os.path.exists(path + ID + '.jpg'):
                    path = path.replace('/right/', '/left/') if i % 2 == 0 else path.replace('/left/', '/right/')
                    x[i][-1] = 1 - x[i][-1]
            else:
                path = self.dir_images
            X[i, :, :, :] = self._generate_image(path_image=path + ID + '.jpg')
        return [X, x], y
    
    def __getitem__(self, index):
        # Select the indices
        idx_start = (index * self.n_ids_batch) % len(self.list_ids)
        idx_end = (((index + 1) * self.n_ids_batch) - 1) % len(self.list_ids) + 1
        if idx_start > idx_end:
            # If this happens outside of training, that is a mistake
            if not self.training_mode:
                print('\nERROR: Outside of training, every sample should only be predicted once!')
                sys.exit(1)
            # Select part of the indices from the end of the epoch
            indices = self.indices[idx_start:]
            # Generate a new set of indices
            # print('\nThe end of the data was reached within this batch, looping.')
            if self.shuffle:
                np.random.shuffle(self.indices)
            # Complete the batch with samples from the new indices
            indices = np.concatenate([indices, self.indices[:idx_end]])
        else:
            indices = self.indices[idx_start: idx_end]
            if idx_end == len(self.list_ids) & self.shuffle:
                # print('\nThe end of the data was reached. Shuffling for the next epoch.')
                np.random.shuffle(self.indices)
        # Keep track of last indice for end of subepoch
        self.idx_end = idx_end
        # Select the corresponding ids
        list_ids_batch = [self.list_ids[i] for i in indices]
        # For paired organs, two images (left, right eyes) are selected for each id.
        if self.organ + '_' + self.view in self.left_right_organs_views:
            list_ids_batch = [ID for ID in list_ids_batch for _ in ('right', 'left')]
        return self._data_generation(list_ids_batch)


class MyCSVLogger(Callback):
    
    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.csv_file = None
        if six.PY2:
            self.file_flags = 'b'
            self._open_args = {}
        else:
            self.file_flags = ''
            self._open_args = {'newline': '\n'}
        Callback.__init__(self)
    
    def on_train_begin(self, logs=None):
        if self.append:
            if file_io.file_exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            mode = 'a'
        else:
            mode = 'w'
        self.csv_file = io.open(self.filename, mode + self.file_flags, **self._open_args)
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, collections_abc.Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k
        
        if self.keys is None:
            self.keys = sorted(logs.keys())
        
        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])
        
        if not self.writer:
            
            class CustomDialect(csv.excel):
                delimiter = self.sep
            
            fieldnames = ['epoch', 'learning_rate'] + self.keys
            if six.PY2:
                fieldnames = [unicode(x) for x in fieldnames]
            
            self.writer = csv.DictWriter(
                self.csv_file,
                fieldnames=fieldnames,
                dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()
        
        row_dict = collections.OrderedDict({'epoch': epoch, 'learning_rate': eval(self.model.optimizer.lr)})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
    
    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, monitor='val_loss', baseline=-np.Inf, verbose=0, save_best_only=False,
                 save_weights_only=False, mode='auto', save_freq='epoch'):
        # Parameters
        ModelCheckpoint.__init__(self, filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only,
                                 save_weights_only=save_weights_only, mode=mode, save_freq=save_freq)
        if mode == 'min':
            self.monitor_op = np.less
            self.best = baseline
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = baseline
        else:
            print('Error. mode for metric must be either min or max')
            sys.exit(1)


class DeepLearning(Metrics):
    """
    Train models
    """
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, n_fc_layers=None,
                 n_fc_nodes=None, optimizer=None, learning_rate=None, weight_decay=None, dropout_rate=None,
                 data_augmentation_factor=None, debug_mode=False):
        # Initialization
        Metrics.__init__(self)
        tf.random.set_seed(self.seed)
        
        # Model's version
        self.target = target
        self.organ = organ
        self.view = view
        self.transformation = transformation
        self.architecture = architecture
        self.n_fc_layers = int(n_fc_layers)
        self.n_fc_nodes = int(n_fc_nodes)
        self.optimizer = optimizer
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.dropout_rate = float(dropout_rate)
        self.data_augmentation_factor = float(data_augmentation_factor)
        self.outer_fold = None
        self.version = target + '_' + organ + '_' + view + '_' + transformation + '_' + architecture + '_' + \
                       n_fc_layers + '_' + n_fc_nodes + '_' + optimizer + '_' + learning_rate + '_' + weight_decay + \
                       '_' + dropout_rate + '_' + data_augmentation_factor
        
        # NNet's architecture and weights
        self.side_predictors = self.dict_side_predictors[target]
        if self.organ + '_' + self.view in self.left_right_organs_views:
            self.side_predictors.append('organ_side')
        self.dict_final_activations = {'regression': 'linear', 'binary': 'sigmoid', 'multiclass': 'softmax',
                                       'saliency': 'linear'}
        self.path_load_weights = None
        self.keras_weights = None
        
        # Generators
        self.debug_mode = debug_mode
        self.debug_fraction = 0.005
        self.DATA_FEATURES = {}
        self.mode = None
        self.n_cpus = len(os.sched_getaffinity(0))
        self.dir_images = '../images/' + organ + '/' + view + '/' + transformation + '/'
        
        # define dictionary to fit the architecture's input size to the images sizes (take min (height, width))
        self.dict_organ_view_transformation_to_image_size = {
            'Eyes_Fundus_Raw': (316, 316),  # initial size (1388, 1388)
            'Eyes_OCT_Raw': (312, 320),  # initial size (500, 512)
            'Musculoskeletal_Spine_Sagittal': (466, 211),  # initial size (1513, 684)
            'Musculoskeletal_Spine_Coronal': (315, 313),  # initial size (724, 720)
            'Musculoskeletal_Hips_MRI': (329, 303),  # initial size (626, 680)
            'Musculoskeletal_Knees_MRI': (347, 286)  # initial size (851, 700)
        }
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Brain_MRI_SagittalRaw', 'Brain_MRI_SagittalReference', 'Brain_MRI_CoronalRaw',
                           'Brain_MRI_CoronalReference', 'Brain_MRI_TransverseRaw', 'Brain_MRI_TransverseReference'],
                          (316, 316)))  # initial size (88, 88)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Arterial_Carotids_Mixed', 'Arterial_Carotids_LongAxis', 'Arterial_Carotids_CIMT120',
                           'Arterial_Carotids_CIMT150', 'Arterial_Carotids_ShortAxis'],
                          (337, 291)))  # initial size (505, 436)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Heart_MRI_2chambersRaw', 'Heart_MRI_2chambersContrast', 'Heart_MRI_3chambersRaw',
                           'Heart_MRI_3chambersContrast', 'Heart_MRI_4chambersRaw', 'Heart_MRI_4chambersContrast'],
                          (316, 316)))  # initial size (200, 200)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Abdomen_Liver_Raw', 'Abdomen_Liver_Contrast'], (288, 364)))  # initial size (288, 364)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Abdomen_Pancreas_Raw', 'Abdomen_Pancreas_Contrast'], (288, 350)))  # initial size (288, 350)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['Musculoskeletal_FullBody_Figure', 'Musculoskeletal_FullBody_Skeleton',
                           'Musculoskeletal_FullBody_Flesh', 'Musculoskeletal_FullBody_Mixed'],
                          (541, 181)))  # initial size (811, 272)
        self.dict_organ_view_transformation_to_image_size.update(
            dict.fromkeys(['PhysicalActivity_FullWeek_GramianAngularField1minDifference',
                           'PhysicalActivity_FullWeek_GramianAngularField1minSummation',
                           'PhysicalActivity_FullWeek_MarkovTransitionField1min',
                           'PhysicalActivity_FullWeek_RecurrencePlots1min'],
                          (316, 316)))  # initial size (316, 316)
        self.dict_architecture_to_image_size = {'MobileNet': (224, 224), 'MobileNetV2': (224, 224),
                                                'NASNetMobile': (224, 224), 'NASNetLarge': (331, 331)}
        if self.architecture in ['MobileNet', 'MobileNetV2', 'NASNetMobile', 'NASNetLarge']:
            self.image_width, self.image_height = self.dict_architecture_to_image_size[architecture]
        else:
            self.image_width, self.image_height = \
                self.dict_organ_view_transformation_to_image_size[organ + '_' + view + '_' + transformation]
        
        # define dictionary of batch sizes to fit as many samples as the model's architecture allows
        self.dict_batch_sizes = {
            # Default, applies to all images with resized input ~100,000 pixels
            'Default': {'VGG16': 32, 'VGG19': 32, 'DenseNet121': 16, 'DenseNet169': 16, 'DenseNet201': 16,
                        'Xception': 32, 'InceptionV3': 32, 'InceptionResNetV2': 8, 'ResNet50': 32, 'ResNet101': 16,
                        'ResNet152': 16, 'ResNet50V2': 32, 'ResNet101V2': 16, 'ResNet152V2': 16, 'ResNeXt50': 4,
                        'ResNeXt101': 8, 'EfficientNetB7': 4,
                        'MobileNet': 128, 'MobileNetV2': 64, 'NASNetMobile': 64, 'NASNetLarge': 4}}
        # Define batch size
        if organ + '_' + view in self.dict_batch_sizes.keys():
            randoself.batch_size = self.dict_batch_sizes[organ + '_' + view][architecture]
        else:
            self.batch_size = self.dict_batch_sizes['Default'][architecture]
        # double the batch size for the teslaM40 cores that have bigger memory
        if len(GPUtil.getGPUs()) > 0:  # make sure GPUs are available (not truesometimes for debugging)
            if GPUtil.getGPUs()[0].memoryTotal > 20000:
                self.batch_size *= 2
        # Define number of ids per batch (twice fewer for paired organs, because left and right samples)
        self.n_ids_batch = self.batch_size
        if organ + '_' + view in self.left_right_organs_views:
            self.n_ids_batch //= 2
        
        # Define number of samples per subepoch
        if debug_mode:
            self.n_samples_per_subepoch = self.batch_size * 4
        else:
            self.n_samples_per_subepoch = 32768
        if organ + '_' + view in self.left_right_organs_views:
            self.n_samples_per_subepoch //= 2
        
        # dict to decide which field is used to generate the ids when several targets share the same ids
        self.dict_target_to_ids = dict.fromkeys(['Age', 'Sex'], 'Age')
        
        # Note: R-Squared and F1-Score are not available, because their batch based values are misleading.
        # For some reason, Sensitivity and Specificity are not available either. Might implement later.
        self.dict_losses_K = {'MSE': MeanSquaredError(name='MSE'),
                              'Binary-Crossentropy': BinaryCrossentropy(name='Binary-Crossentropy')}
        self.dict_metrics_K = {'R-Squared': RSquare(name='R-Squared', y_shape=(1,)),
                               'RMSE': RootMeanSquaredError(name='RMSE'),
                               'F1-Score': F1Score(name='F1-Score', num_classes=1, dtype=tf.float32),
                               'ROC-AUC': AUC(curve='ROC', name='ROC-AUC'),
                               'PR-AUC': AUC(curve='PR', name='PR-AUC'),
                               'Binary-Accuracy': BinaryAccuracy(name='Binary-Accuracy'),
                               'Precision': Precision(name='Precision'),
                               'Recall': Recall(name='Recall'),
                               'True-Positives': TruePositives(name='True-Positives'),
                               'False-Positives': FalsePositives(name='False-Positives'),
                               'False-Negatives': FalseNegatives(name='False-Negatives'),
                               'True-Negatives': TrueNegatives(name='True-Negatives')}
        
        # Metrics
        self.prediction_type = self.dict_prediction_types[target]
        self.loss_name = self.dict_losses_names[self.prediction_type]
        self.loss_function = self.dict_losses_K[self.loss_name]
        self.main_metric_name = self.dict_main_metrics_names_K[target]
        self.main_metric_mode = self.main_metrics_modes[self.main_metric_name]
        self.main_metric = self.dict_metrics_K[self.main_metric_name]
        self.metrics_names = [self.main_metric_name]
        self.metrics = [self.dict_metrics_K[metric_name] for metric_name in self.metrics_names]
        
        # Optimizers
        self.optimizers = {'Adam': Adam, 'RMSprop': RMSprop, 'Adadelta': Adadelta}
        
        # Model
        self.model = None
    
    @staticmethod
    def _append_ext(fn):
        return fn + ".jpg"
    
    def _load_data_features(self):
        for fold in self.folds:
            self.DATA_FEATURES[fold] = pd.read_csv(
                self.path_data + 'data-features_' + self.organ + '_' + self.view + '_' + self.transformation + '_' +
                self.dict_target_to_ids[self.target] + '_' + fold + '_' + self.outer_fold + '.csv')
            for col_name in self.id_vars:
                self.DATA_FEATURES[fold][col_name] = self.DATA_FEATURES[fold][col_name].astype(str)
            self.DATA_FEATURES[fold].set_index('id', drop=False, inplace=True)
    
    def _take_subset_to_debug(self):
        for fold in self.folds:
            # use +1 or +2 to test the leftovers pipeline
            leftovers_extra = {'train': 0, 'val': 1, 'test': 2}
            n_batches = 2
            n_limit_fold = leftovers_extra[fold] + self.batch_size * n_batches
            self.DATA_FEATURES[fold] = self.DATA_FEATURES[fold].iloc[:n_limit_fold, :]
    
    def _generate_generators(self, DATA_FEATURES):
        GENERATORS = {}
        for fold in self.folds:
            # do not generate a generator if there are no samples (can happen for leftovers generators)
            if fold not in DATA_FEATURES.keys():
                continue
            # parameters
            training_mode = True if self.mode == 'model_training' else False
            if (fold == 'train') & (self.mode == 'model_training') & \
                    (self.organ + '_' + self.view not in self.organsviews_not_to_augment):
                data_augmentation = True
            else:
                data_augmentation = False
            # define batch size for testing: data is split between a part that fits in batches, and leftovers
            if self.mode == 'model_testing':
                if self.organ + '_' + self.view in self.left_right_organs_views:
                    n_samples = len(DATA_FEATURES[fold].index) * 2
                else:
                    n_samples = len(DATA_FEATURES[fold].index)
                batch_size_fold = min(self.batch_size, n_samples)
            else:
                batch_size_fold = self.batch_size
            if (fold == 'train') & (self.mode == 'model_training'):
                n_samples_per_subepoch = self.n_samples_per_subepoch
            else:
                n_samples_per_subepoch = None
            # generator
            GENERATORS[fold] = \
                MyImageDataGenerator(target=self.target, organ=self.organ, view=self.view,
                                     data_features=DATA_FEATURES[fold], n_samples_per_subepoch=n_samples_per_subepoch,
                                     batch_size=batch_size_fold, training_mode=training_mode,
                                     side_predictors=self.side_predictors, dir_images=self.dir_images,
                                     images_width=self.image_width, images_height=self.image_height,
                                     data_augmentation=data_augmentation,
                                     data_augmentation_factor=self.data_augmentation_factor, seed=self.seed)
        return GENERATORS
    
    def _generate_class_weights(self):
        if self.dict_prediction_types[self.target] == 'binary':
            self.class_weights = {}
            counts = self.DATA_FEATURES['train'][self.target + '_raw'].value_counts()
            n_total = counts.sum()
            # weighting the samples for each class inversely proportional to their prevalence, with order of magnitude 1
            for i in counts.index.values:
                self.class_weights[i] = n_total / (counts.loc[i] * len(counts.index))
    
    def _generate_cnn(self):
        # define the arguments
        # take special initial weights for EfficientNetB7 (better)
        if (self.architecture == 'EfficientNetB7') & (self.keras_weights == 'imagenet'):
            w = 'noisy-student'
        else:
            w = self.keras_weights
        kwargs = {"include_top": False, "weights": w, "input_shape": (self.image_width, self.image_height, 3)}
        if self.architecture in ['ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2',
                                 'ResNeXt50', 'ResNeXt101']:
            import tensorflow.keras
            kwargs.update(
                {"backend": tensorflow.keras.backend, "layers": tensorflow.keras.layers,
                 "models": tensorflow.keras.models, "utils": tensorflow.keras.utils})
        
        # load the architecture builder
        if self.architecture == 'VGG16':
            from tensorflow.keras.applications.vgg16 import VGG16 as ModelBuilder
        elif self.architecture == 'VGG19':
            from tensorflow.keras.applications.vgg19 import VGG19 as ModelBuilder
        elif self.architecture == 'DenseNet121':
            from tensorflow.keras.applications.densenet import DenseNet121 as ModelBuilder
        elif self.architecture == 'DenseNet169':
            from tensorflow.keras.applications.densenet import DenseNet169 as ModelBuilder
        elif self.architecture == 'DenseNet201':
            from tensorflow.keras.applications.densenet import DenseNet201 as ModelBuilder
        elif self.architecture == 'Xception':
            from tensorflow.keras.applications.xception import Xception as ModelBuilder
        elif self.architecture == 'InceptionV3':
            from tensorflow.keras.applications.inception_v3 import InceptionV3 as ModelBuilder
        elif self.architecture == 'InceptionResNetV2':
            from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as ModelBuilder
        elif self.architecture == 'ResNet50':
            from keras_applications.resnet import ResNet50 as ModelBuilder
        elif self.architecture == 'ResNet101':
            from keras_applications.resnet import ResNet101 as ModelBuilder
        elif self.architecture == 'ResNet152':
            from keras_applications.resnet import ResNet152 as ModelBuilder
        elif self.architecture == 'ResNet50V2':
            from keras_applications.resnet_v2 import ResNet50V2 as ModelBuilder
        elif self.architecture == 'ResNet101V2':
            from keras_applications.resnet_v2 import ResNet101V2 as ModelBuilder
        elif self.architecture == 'ResNet152V2':
            from keras_applications.resnet_v2 import ResNet152V2 as ModelBuilder
        elif self.architecture == 'ResNeXt50':
            from keras_applications.resnext import ResNeXt50 as ModelBuilder
        elif self.architecture == 'ResNeXt101':
            from keras_applications.resnext import ResNeXt101 as ModelBuilder
        elif self.architecture == 'EfficientNetB7':
            from efficientnet.tfkeras import EfficientNetB7 as ModelBuilder
        # The following model have a fixed input size requirement
        elif self.architecture == 'NASNetMobile':
            from tensorflow.keras.applications.nasnet import NASNetMobile as ModelBuilder
        elif self.architecture == 'NASNetLarge':
            from tensorflow.keras.applications.nasnet import NASNetLarge as ModelBuilder
        elif self.architecture == 'MobileNet':
            from tensorflow.keras.applications.mobilenet import MobileNet as ModelBuilder
        elif self.architecture == 'MobileNetV2':
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as ModelBuilder
        else:
            print('Architecture does not exist.')
            sys.exit(1)
        
        # build the model's base
        cnn = ModelBuilder(**kwargs)
        x = cnn.output
        # complete the model's base
        if self.architecture in ['VGG16', 'VGG19']:
            x = Flatten()(x)
            x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = Dropout(self.dropout_rate)(x)
            x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            x = Dropout(self.dropout_rate)(x)
        else:
            x = GlobalAveragePooling2D()(x)
            if self.architecture == 'EfficientNetB7':
                x = Dropout(self.dropout_rate)(x)
        cnn_output = x
        return cnn.input, cnn_output
    
    def _generate_side_nn(self):
        side_nn = Sequential()
        side_nn.add(Dense(16, input_dim=len(self.side_predictors), activation="relu",
                          kernel_regularizer=regularizers.l2(self.weight_decay)))
        return side_nn.input, side_nn.output
    
    def _complete_architecture(self, cnn_input, cnn_output, side_nn_input, side_nn_output):
        x = concatenate([cnn_output, side_nn_output])
        x = Dropout(self.dropout_rate)(x)
        for n in [int(self.n_fc_nodes * (2 ** (2 * (self.n_fc_layers - 1 - i)))) for i in range(self.n_fc_layers)]:
            x = Dense(n, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay))(x)
            # scale the dropout proportionally to the number of nodes in a layer. No dropout for the last layers
            if n > 16:
                x = Dropout(self.dropout_rate * n / 1024)(x)
        predictions = Dense(1, activation=self.dict_final_activations[self.prediction_type],
                            kernel_regularizer=regularizers.l2(self.weight_decay))(x)
        self.model = Model(inputs=[cnn_input, side_nn_input], outputs=predictions)
    
    def _generate_architecture(self):
        cnn_input, cnn_output = self._generate_cnn()
        side_nn_input, side_nn_output = self._generate_side_nn()
        self._complete_architecture(cnn_input=cnn_input, cnn_output=cnn_output, side_nn_input=side_nn_input,
                                    side_nn_output=side_nn_output)
    
    def _load_model_weights(self):
        try:
            self.model.load_weights(self.path_load_weights)
        except (FileNotFoundError, TypeError):
            # load backup weights if the main weights are corrupted
            try:
                self.model.load_weights(self.path_load_weights.replace('model-weights', 'backup-model-weights'))
            except FileNotFoundError:
                print('Error. No file was found. imagenet weights should have been used. Bug somewhere.')
                sys.exit(1)
    
    @staticmethod
    def clean_exit():
        # exit
        print('\nDone.\n')
        print('Killing JOB PID with kill...')
        os.system('touch ../eo/' + os.environ['SLURM_JOBID'])
        os.system('kill ' + str(os.getpid()))
        time.sleep(60)
        print('Escalating to kill JOB PID with kill -9...')
        os.system('kill -9 ' + str(os.getpid()))
        time.sleep(60)
        print('Escalating to kill JOB ID')
        os.system('scancel ' + os.environ['SLURM_JOBID'])
        time.sleep(60)
        print('Everything failed to kill the job. Hanging there until hitting walltime...')


class Training(DeepLearning):
    """
    Train models
    """
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, n_fc_layers=None,
                 n_fc_nodes=None, optimizer=None, learning_rate=None, weight_decay=None, dropout_rate=None,
                 data_augmentation_factor=None, outer_fold=None, debug_mode=False, transfer_learning=None,
                 continue_training=True, display_full_metrics=True):
        # parameters
        DeepLearning.__init__(self, target, organ, view, transformation, architecture, n_fc_layers, n_fc_nodes,
                              optimizer, learning_rate, weight_decay, dropout_rate, data_augmentation_factor,
                              debug_mode)
        self.outer_fold = outer_fold
        self.version = self.version + '_' + str(outer_fold)
        # NNet's architecture's weights
        self.continue_training = continue_training
        self.transfer_learning = transfer_learning
        self.list_parameters_to_match = ['organ', 'transformation', 'view']
        # dict to decide in which order targets should be used when trying to transfer weight from a similar model
        self.dict_alternative_targets_for_transfer_learning = {'Age': ['Age', 'Sex'], 'Sex': ['Sex', 'Age']}
        
        # Generators
        self.folds = ['train', 'val']
        self.mode = 'model_training'
        self.class_weights = None
        self.GENERATORS = None
        
        # Metrics
        self.baseline_performance = None
        if display_full_metrics:
            self.metrics_names = self.dict_metrics_names_K[self.prediction_type]
        
        # Model
        self.path_load_weights = self.path_data + 'model-weights_' + self.version + '.h5'
        if debug_mode:
            self.path_save_weights = self.path_data + 'model-weights-debug.h5'
        else:
            self.path_save_weights = self.path_data + 'model-weights_' + self.version + '.h5'
        self.n_epochs_max = 100000
        self.callbacks = None
    
    # Load and preprocess the data, build the generators
    def data_preprocessing(self):
        self._load_data_features()
        if self.debug_mode:
            self._take_subset_to_debug()
        self._generate_class_weights()
        self.GENERATORS = self._generate_generators(self.DATA_FEATURES)
    
    # Determine which weights to load, if any.
    def _weights_for_transfer_learning(self):
        print('Looking for models to transfer weights from...')
        
        # define parameters
        parameters = self._version_to_parameters(self.version)
        
        # continue training if possible
        if self.continue_training and os.path.exists(self.path_load_weights):
            print('Loading the weights from the model\'s previous training iteration.')
            return
        
        # Initialize the weights using other the weights from other successful hyperparameters combinations
        if self.transfer_learning == 'hyperparameters':
            # Check if the same model with other hyperparameters have already been trained. Pick the best for transfer.
            params = self.version.split('_')
            params_tl_idx = \
                [i for i in range(len(names_model_parameters))
                 if any(names_model_parameters[i] == p for p in
                        ['optimizer', 'learning_rate', 'weight_decay', 'dropout_rate', 'data_augmentation_factor'])]
            for idx in params_tl_idx:
                params[idx] = '*'
            versions = '../eo/MI02_' + '_'.join(params) + '.out'
            files = glob.glob(versions)
            if self.main_metric_mode == 'min':
                best_perf = np.Inf
            else:
                best_perf = -np.Inf
            for file in files:
                hand = open(file, 'r')
                # find best last performance
                final_improvement_line = None
                baseline_performance_line = None
                for line in hand:
                    line = line.rstrip()
                    if re.search('Baseline validation ' + self.main_metric_name + ' = ', line):
                        baseline_performance_line = line
                    if re.search('val_' + self.main_metric_name + ' improved from', line):
                        final_improvement_line = line
                hand.close()
                if final_improvement_line is not None:
                    perf = float(final_improvement_line.split(' ')[7].replace(',', ''))
                elif baseline_performance_line is not None:
                    perf = float(baseline_performance_line.split(' ')[-1])
                else:
                    continue
                # Keep track of the file with the best performance
                if self.main_metric_mode == 'min':
                    update = perf < best_perf
                else:
                    update = perf > best_perf
                if update:
                    best_perf = perf
                    self.path_load_weights = \
                        file.replace('../eo/', self.path_data).replace('MI02', 'model-weights').replace('.out', '.h5')
            if best_perf not in [-np.Inf, np.Inf]:
                print('Transfering the weights from: ' + self.path_load_weights + ', with ' + self.main_metric_name +
                      ' = ' + str(best_perf))
                return
        
        # Initialize the weights based on models trained on different datasets, ranked by similarity
        if self.transfer_learning == 'datasets':
            while True:
                # print('Matching models for the following criterias:');
                # print(['architecture', 'target'] + list_parameters_to_match)
                # start by looking for models trained on the same target, then move to other targets
                for target_to_load in self.dict_alternative_targets_for_transfer_learning[parameters['target']]:
                    # print('Target used: ' + target_to_load)
                    parameters_to_match = parameters.copy()
                    parameters_to_match['target'] = target_to_load
                    # load the ranked performances table to select the best performing model among the similar
                    # models available
                    path_performances_to_load = self.path_data + 'PERFORMANCES_ranked_' + \
                                                parameters_to_match['target'] + '_' + 'val' + '.csv'
                    try:
                        Performances = pd.read_csv(path_performances_to_load)
                        Performances['organ'] = Performances['organ'].astype(str)
                    except FileNotFoundError:
                        # print("Could not load the file: " + path_performances_to_load)
                        break
                    # iteratively get rid of models that are not similar enough, based on the list
                    for parameter in ['architecture', 'target'] + self.list_parameters_to_match:
                        Performances = Performances[Performances[parameter] == parameters_to_match[parameter]]
                    # if at least one model is similar enough, load weights from the best of them
                    if len(Performances.index) != 0:
                        self.path_load_weights = self.path_data + 'model-weights_' + Performances['version'][0] + '.h5'
                        self.keras_weights = None
                        print('transfering the weights from: ' + self.path_load_weights)
                        return
                
                # if no similar model was found, try again after getting rid of the last selection criteria
                if len(self.list_parameters_to_match) == 0:
                    print('No model found for transfer learning.')
                    break
                self.list_parameters_to_match.pop()
        
        # Otherwise use imagenet weights to initialize
        print('Using imagenet weights.')
        # using string instead of None for path to not ge
        self.path_load_weights = None
        self.keras_weights = 'imagenet'
    
    def _compile_model(self):
        # if learning rate was reduced with success according to logger, start with this reduced learning rate
        if self.path_load_weights is not None:
            path_logger = self.path_load_weights.replace('model-weights', 'logger').replace('.h5', '.csv')
        else:
            path_logger = self.path_data + 'logger_' + self.version + '.csv'
        if os.path.exists(path_logger):
            try:
                logger = pd.read_csv(path_logger)
                best_log = \
                    logger[logger['val_' + self.main_metric_name] == logger['val_' + self.main_metric_name].max()]
                lr = best_log['learning_rate'].values[0]
            except pd.errors.EmptyDataError:
                os.remove(path_logger)
                lr = self.learning_rate
        else:
            lr = self.learning_rate
        self.model.compile(optimizer=self.optimizers[self.optimizer](lr=lr, clipnorm=1.0), loss=self.loss_function,
                           metrics=self.metrics)
    
    def _compute_baseline_performance(self):
        # calculate initial val_loss value
        if self.continue_training:
            idx_metric_name = ([self.loss_name] + self.metrics_names).index(self.main_metric_name)
            baseline_perfs = self.model.evaluate(self.GENERATORS['val'], steps=self.GENERATORS['val'].steps)
            self.baseline_performance = baseline_perfs[idx_metric_name]
        elif self.main_metric_mode == 'min':
            self.baseline_performance = np.Inf
        else:
            self.baseline_performance = -np.Inf
        print('Baseline validation ' + self.main_metric_name + ' = ' + str(self.baseline_performance))
    
    def _define_callbacks(self):
        if self.debug_mode:
            path_logger = self.path_data + 'logger-debug.csv'
            append = False
        else:
            path_logger = self.path_data + 'logger_' + self.version + '.csv'
            append = self.continue_training
        csv_logger = MyCSVLogger(path_logger, separator=',', append=append)
        model_checkpoint_backup = MyModelCheckpoint(self.path_save_weights.replace('model-weights',
                                                                                   'backup-model-weights'),
                                                    monitor='val_' + self.main_metric.name,
                                                    baseline=self.baseline_performance, verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode=self.main_metric_mode,
                                                    save_freq='epoch')
        model_checkpoint = MyModelCheckpoint(self.path_save_weights,
                                             monitor='val_' + self.main_metric.name, baseline=self.baseline_performance,
                                             verbose=1, save_best_only=True, save_weights_only=True,
                                             mode=self.main_metric_mode, save_freq='epoch')
        patience_reduce_lr = min(7, 3 * self.GENERATORS['train'].n_subepochs_per_epoch)
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=patience_reduce_lr, verbose=1,
                                                 mode='min', min_delta=0, cooldown=0, min_lr=0)
        early_stopping = EarlyStopping(monitor='val_' + self.main_metric.name, min_delta=0, patience=15, verbose=0,
                                       mode=self.main_metric_mode, baseline=self.baseline_performance,
                                       restore_best_weights=True)
        self.callbacks = [csv_logger, model_checkpoint_backup, model_checkpoint, early_stopping, reduce_lr_on_plateau]
    
    def build_model(self):
        self._weights_for_transfer_learning()
        self._generate_architecture()
        # Load weights if possible
        try:
            load_weights = True if os.path.exists(self.path_load_weights) else False
        except TypeError:
            load_weights = False
        if load_weights:
            self._load_model_weights()
        else:
            # save transferred weights as default, in case no better weights are found
            self.model.save_weights(self.path_save_weights.replace('model-weights', 'backup-model-weights'))
            self.model.save_weights(self.path_save_weights)
        self._compile_model()
        self._compute_baseline_performance()
        self._define_callbacks()
    
    def train_model(self):
        # garbage collector
        _ = gc.collect()
        # use more verbose when debugging
        verbose = 1 if self.debug_mode else 2
        
        # train the model
        self.model.fit(self.GENERATORS['train'], steps_per_epoch=self.GENERATORS['train'].steps,
                       validation_data=self.GENERATORS['val'], validation_steps=self.GENERATORS['val'].steps,
                       shuffle=False, use_multiprocessing=False, workers=self.n_cpus, epochs=self.n_epochs_max,
                       class_weight=self.class_weights, callbacks=self.callbacks, verbose=verbose)


class PredictionsGenerate(DeepLearning):
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, n_fc_layers=None,
                 n_fc_nodes=None, optimizer=None, learning_rate=None, weight_decay=None, dropout_rate=None,
                 data_augmentation_factor=None, outer_fold=None, debug_mode=False):
        # Initialize parameters
        DeepLearning.__init__(self, target, organ, view, transformation, architecture, n_fc_layers, n_fc_nodes,
                              optimizer, learning_rate, weight_decay, dropout_rate, data_augmentation_factor,
                              debug_mode)
        self.outer_fold = outer_fold
        self.mode = 'model_testing'
        # Define dictionaries attributes for data, generators and predictions
        self.DATA_FEATURES_BATCH = {}
        self.DATA_FEATURES_LEFTOVERS = {}
        self.GENERATORS_BATCH = None
        self.GENERATORS_LEFTOVERS = None
        self.PREDICTIONS = {}
    
    def _split_batch_leftovers(self):
        # split the samples into two groups: what can fit into the batch size, and the leftovers.
        for fold in self.folds:
            n_leftovers = len(self.DATA_FEATURES[fold].index) % self.n_ids_batch
            if n_leftovers > 0:
                self.DATA_FEATURES_BATCH[fold] = self.DATA_FEATURES[fold].iloc[:-n_leftovers]
                self.DATA_FEATURES_LEFTOVERS[fold] = self.DATA_FEATURES[fold].tail(n_leftovers)
            else:
                self.DATA_FEATURES_BATCH[fold] = self.DATA_FEATURES[fold]  # special case for syntax if no leftovers
                if fold in self.DATA_FEATURES_LEFTOVERS.keys():
                    del self.DATA_FEATURES_LEFTOVERS[fold]
    
    def _generate_outerfold_predictions(self):
        # prepare unscaling
        if self.target in self.targets_regression:
            mean_train = self.DATA_FEATURES['train'][self.target + '_raw'].mean()
            std_train = self.DATA_FEATURES['train'][self.target + '_raw'].std()
        else:
            mean_train, std_train = None, None
        # Generate predictions
        for fold in self.folds:
            print('Predicting samples from fold ' + fold + '.')
            print(str(len(self.DATA_FEATURES[fold].index)) + ' samples to predict.')
            print('Predicting batches: ' + str(len(self.DATA_FEATURES_BATCH[fold].index)) + ' samples.')
            pred_batch = self.model.predict(self.GENERATORS_BATCH[fold], steps=self.GENERATORS_BATCH[fold].steps,
                                            verbose=1)
            if fold in self.GENERATORS_LEFTOVERS.keys():
                print('Predicting leftovers: ' + str(len(self.DATA_FEATURES_LEFTOVERS[fold].index)) + ' samples.')
                pred_leftovers = self.model.predict(self.GENERATORS_LEFTOVERS[fold],
                                                    steps=self.GENERATORS_LEFTOVERS[fold].steps, verbose=1)
                pred_full = np.concatenate((pred_batch, pred_leftovers)).squeeze()
            else:
                pred_full = pred_batch.squeeze()
            print('Predicted a total of ' + str(len(pred_full)) + ' samples.')
            # take the average between left and right predictions for paired organs
            if self.organ + '_' + self.view in self.left_right_organs_views:
                pred_full = np.mean(pred_full.reshape(-1, 2), axis=1)
            # unscale predictions
            if self.target in self.targets_regression:
                pred_full = pred_full * std_train + mean_train
            # format the dataframe
            self.DATA_FEATURES[fold]['pred'] = pred_full
            self.PREDICTIONS[fold] = self.DATA_FEATURES[fold]
            self.PREDICTIONS[fold]['id'] = [ID.replace('.jpg', '') for ID in self.PREDICTIONS[fold]['id']]
    
    def _generate_predictions(self):
        self.path_load_weights = self.path_data + 'model-weights_' + self.version + '_' + self.outer_fold + '.h5'
        self._load_data_features()
        if self.debug_mode:
            self._take_subset_to_debug()
        self._load_model_weights()
        self._split_batch_leftovers()
        # generate the generators
        self.GENERATORS_BATCH = self._generate_generators(DATA_FEATURES=self.DATA_FEATURES_BATCH)
        if self.DATA_FEATURES_LEFTOVERS is not None:
            self.GENERATORS_LEFTOVERS = self._generate_generators(DATA_FEATURES=self.DATA_FEATURES_LEFTOVERS)
        self._generate_outerfold_predictions()
    
    def _format_predictions(self):
        for fold in self.folds:
            perf_fun = self.dict_metrics_sklearn[self.dict_main_metrics_names[self.target]]
            perf = perf_fun(self.PREDICTIONS[fold][self.target + '_raw'], self.PREDICTIONS[fold]['pred'])
            print('The ' + fold + ' performance is: ' + str(perf))
            # format the predictions
            self.PREDICTIONS[fold].index.name = 'column_names'
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold][['id', 'outer_fold', 'pred']]
    
    def generate_predictions(self):
        self._generate_architecture()
        self._generate_predictions()
        self._format_predictions()
    
    def save_predictions(self):
        for fold in self.folds:
            self.PREDICTIONS[fold].to_csv(self.path_data + 'Predictions_instances_' + self.version + '_' + fold + '_'
                                          + self.outer_fold + '.csv', index=False)


class PredictionsConcatenate(Basics):
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, n_fc_layers=None,
                 n_fc_nodes=None, optimizer=None, learning_rate=None, weight_decay=None, dropout_rate=None,
                 data_augmentation_factor=None):
        # Initialize parameters
        Basics.__init__(self)
        self.version = target + '_' + organ + '_' + view + '_' + transformation + '_' + architecture + '_' + \
                       n_fc_layers + '_' + n_fc_nodes + '_' + optimizer + '_' + learning_rate + '_' + weight_decay + \
                       '_' + dropout_rate + '_' + data_augmentation_factor
        # Define dictionaries attributes for data, generators and predictions
        self.PREDICTIONS = {}
    
    def concatenate_predictions(self):
        for fold in self.folds:
            for outer_fold in self.outer_folds:
                Predictions_fold = pd.read_csv(self.path_data + 'Predictions_instances_' + self.version + '_' + fold +
                                               '_' + outer_fold + '.csv')
                if fold in self.PREDICTIONS.keys():
                    self.PREDICTIONS[fold] = pd.concat([self.PREDICTIONS[fold], Predictions_fold])
                else:
                    self.PREDICTIONS[fold] = Predictions_fold
    
    def save_predictions(self):
        for fold in self.folds:
            self.PREDICTIONS[fold].to_csv(self.path_data + 'Predictions_instances_' + self.version + '_' + fold +
                                          '.csv', index=False)


class PredictionsMerge(Basics):
    
    def __init__(self, target=None, fold=None):
        
        Basics.__init__(self)
        
        # Define dictionaries attributes for data, generators and predictions
        self.target = target
        self.fold = fold
        self.data_features = None
        self.list_models = None
        self.Predictions_df_previous = None
        self.Predictions_df = None
    
    def _load_data_features(self):
        self.data_features = pd.read_csv(self.path_data + 'data-features_instances.csv',
                                         usecols=self.id_vars + self.demographic_vars)
        for var in self.id_vars:
            self.data_features[var] = self.data_features[var].astype(str)
        self.data_features.set_index('id', drop=False, inplace=True)
        self.data_features.index.name = 'column_names'
    
    def _preprocess_data_features(self):
        # For the training set, each sample is predicted n_CV_outer_folds times, so prepare a larger dataframe
        if self.fold == 'train':
            df_all_folds = None
            for outer_fold in self.outer_folds:
                df_fold = self.data_features.copy()
                df_all_folds = df_fold if outer_fold == self.outer_folds[0] else df_all_folds.append(df_fold)
            self.data_features = df_all_folds
    
    def _load_previous_merged_predictions(self):
        if os.path.exists(self.path_data + 'PREDICTIONS_withoutEnsembles_instances_' + self.target + '_' + self.fold +
                          '.csv'):
            self.Predictions_df_previous = pd.read_csv(self.path_data + 'PREDICTIONS_withoutEnsembles_instances_' +
                                                       self.target + '_' + self.fold + '.csv')
            self.Predictions_df_previous.drop(columns=['eid', 'instance'] + self.demographic_vars, inplace=True)
    
    def _list_models(self):
        # generate list of predictions that will be integrated in the Predictions dataframe
        self.list_models = glob.glob(self.path_data + 'Predictions_instances_' + self.target + '_*_' + self.fold +
                                     '.csv')
        # get rid of ensemble models and models already merged
        self.list_models = [model for model in self.list_models if ('*' not in model)]
        if self.Predictions_df_previous is not None:
            self.list_models = \
                [model for model in self.list_models
                 if ('pred_' + '_'.join(model.split('_')[2:-1]) not in self.Predictions_df_previous.columns)]
        self.list_models.sort()
    
    def preprocessing(self):
        self._load_data_features()
        self._preprocess_data_features()
        self._load_previous_merged_predictions()
        self._list_models()
    
    def merge_predictions(self):
        # merge the predictions
        print('There are ' + str(len(self.list_models)) + ' models to merge.')
        i = 0
        # define subgroups to accelerate merging process
        list_subgroups = list(set(['_'.join(model.split('_')[3:7]) for model in self.list_models]))
        for subgroup in list_subgroups:
            print('Merging models from the subgroup ' + subgroup)
            models_subgroup = [model for model in self.list_models if subgroup in model]
            Predictions_subgroup = None
            # merge the models one by one
            for file_name in models_subgroup:
                i += 1
                version = '_'.join(file_name.split('_')[2:-1])
                if self.Predictions_df_previous is not None and \
                        'pred_' + version in self.Predictions_df_previous.columns:
                    print('The model ' + version + ' has already been merged before.')
                else:
                    print('Merging the ' + str(i) + 'th model: ' + version)
                    # load csv and format the predictions
                    prediction = pd.read_csv(self.path_data + file_name)
                    print('raw prediction\'s shape: ' + str(prediction.shape))
                    for var in ['id', 'outer_fold']:
                        prediction[var] = prediction[var].apply(str)
                    prediction.rename(columns={'pred': 'pred_' + version}, inplace=True)
                    # merge data frames
                    if Predictions_subgroup is None:
                        Predictions_subgroup = prediction
                    elif self.fold == 'train':
                        Predictions_subgroup = Predictions_subgroup.merge(prediction, how='outer',
                                                                          on=['id', 'outer_fold'])
                    else:
                        prediction.drop(['outer_fold'], axis=1, inplace=True)
                        # not supported for panda version > 0.23.4 for now
                        Predictions_subgroup = Predictions_subgroup.merge(prediction, how='outer', on=['id'])
            
            # merge group predictions data frames
            if self.fold != 'train':
                Predictions_subgroup.drop(['outer_fold'], axis=1, inplace=True)
            if Predictions_subgroup is not None:
                if self.Predictions_df is None:
                    self.Predictions_df = Predictions_subgroup
                elif self.fold == 'train':
                    self.Predictions_df = self.Predictions_df.merge(Predictions_subgroup, how='outer',
                                                                    on=['id', 'outer_fold'])
                else:
                    # not supported for panda version > 0.23.4 for now
                    self.Predictions_df = self.Predictions_df.merge(Predictions_subgroup, how='outer', on=['id'])
                print('Predictions_df\'s shape: ' + str(self.Predictions_df.shape))
                # garbage collector
                gc.collect()
        
        # Merge with the previously merged predictions
        if (self.Predictions_df_previous is not None) & (self.Predictions_df is not None):
            if self.fold == 'train':
                self.Predictions_df = self.Predictions_df_previous.merge(self.Predictions_df, how='outer',
                                                                         on=['id', 'outer_fold'])
            else:
                self.Predictions_df.drop(columns=['outer_fold'], inplace=True)
                # not supported for panda version > 0.23.4 for now
                self.Predictions_df = self.Predictions_df_previous.merge(self.Predictions_df, how='outer', on=['id'])
            self.Predictions_df_previous = None
        elif self.Predictions_df is None:
            print('No new models to merge. Exiting.')
            print('Done.')
            sys.exit(0)
        
        # Reorder the columns alphabetically
        pred_versions = [col for col in self.Predictions_df.columns if 'pred_' in col]
        pred_versions.sort()
        id_cols = ['id', 'outer_fold'] if self.fold == 'train' else ['id']
        self.Predictions_df = self.Predictions_df[id_cols + pred_versions]
    
    def postprocessing(self):
        # get rid of useless rows in data_features before merging to keep the memory requirements as low as possible
        self.data_features = self.data_features[self.data_features['id'].isin(self.Predictions_df['id'].values)]
        # merge data_features and predictions
        if self.fold == 'train':
            print('Starting to merge a massive dataframe')
            self.Predictions_df = self.data_features.merge(self.Predictions_df, how='outer', on=['id', 'outer_fold'])
        else:
            # not supported for panda version > 0.23.4 for now
            self.Predictions_df = self.data_features.merge(self.Predictions_df, how='outer', on=['id'])
        print('Merging done')
        
        # remove rows for which no prediction is available (should be none)
        subset_cols = [col for col in self.Predictions_df.columns if 'pred_' in col]
        self.Predictions_df.dropna(subset=subset_cols, how='all', inplace=True)
        
        # Displaying the R2s
        versions = [col.replace('pred_', '') for col in self.Predictions_df.columns if 'pred_' in col]
        r2s = []
        for version in versions:
            df = self.Predictions_df[[self.target, 'pred_' + version]].dropna()
            r2s.append(r2_score(df[self.target], df['pred_' + version]))
        R2S = pd.DataFrame({'version': versions, 'R2': r2s})
        R2S.sort_values(by='R2', ascending=False, inplace=True)
        print('R2 for each model: ')
        print(R2S)
    
    def save_merged_predictions(self):
        print('Writing the merged predictions...')
        self.Predictions_df.to_csv(self.path_data + 'PREDICTIONS_withoutEnsembles_instances_' + self.target + '_' +
                                   self.fold + '.csv', index=False)


class PredictionsEids(Basics):
    
    def __init__(self, target=None, fold=None, debug_mode=None):
        Basics.__init__(self)
        
        # Define dictionaries attributes for data, generators and predictions
        self.target = target
        self.fold = fold
        self.debug_mode = debug_mode
        self.Predictions = None
        self.Predictions_chunk = None
        self.pred_versions = None
        self.res_versions = None
        self.target_0s = None
        self.Predictions_eids = None
        self.Predictions_eids_previous = None
        self.pred_versions_previous = None
    
    def preprocessing(self):
        # Load predictions
        self.Predictions = pd.read_csv(
            self.path_data + 'PREDICTIONS_withoutEnsembles_instances_' + self.target + '_' + self.fold + '.csv')
        self.Predictions.drop(columns=['id'], inplace=True)
        self.Predictions['eid'] = self.Predictions['eid'].astype(str)
        self.Predictions.index.name = 'column_names'
        self.pred_versions = [col for col in self.Predictions.columns.values if 'pred_' in col]
        
        # Prepare target values on instance 0 as a reference
        target_0s = pd.read_csv(self.path_data + 'data-features_eids.csv', usecols=['eid', self.target])
        target_0s['eid'] = target_0s['eid'].astype(str)
        target_0s.set_index('eid', inplace=True)
        target_0s = target_0s[self.target]
        target_0s.name = 'target_0'
        target_0s = target_0s[self.Predictions['eid'].unique()]
        self.Predictions = self.Predictions.merge(target_0s, on='eid')
        
        # Compute biological ages reported to target_0
        for pred in self.pred_versions:
            # Compute the biais of the predictions as a function of age
            print('Generating residuals for model ' + pred.replace('pred_', ''))
            df_model = self.Predictions[['Age', pred]]
            df_model.dropna(inplace=True)
            if (len(df_model.index)) > 0:
                age = df_model.loc[:, ['Age']]
                res = df_model['Age'] - df_model[pred]
                regr = LinearRegression()
                regr.fit(age, res)
                self.Predictions[pred.replace('pred_', 'correction_')] = regr.predict(self.Predictions[['Age']])
                # Take the residuals bias into account when "translating" the prediction to instance 0
                correction = self.Predictions['target_0'] - self.Predictions[self.target] + \
                             regr.predict(self.Predictions[['Age']]) - regr.predict(self.Predictions[['target_0']])
                self.Predictions[pred] = self.Predictions[pred] + correction
        self.Predictions[self.target] = self.Predictions['target_0']
        self.Predictions.drop(columns=['target_0'], inplace=True)
        self.Predictions.index.name = 'column_names'
    
    def processing(self):
        if self.fold == 'train':
            # Prepare template to which each model will be appended
            Predictions = self.Predictions[['eid'] + self.demographic_vars]
            Predictions = Predictions.groupby('eid', as_index=True).mean()
            Predictions.index.name = 'column_names'
            Predictions['eid'] = Predictions.index.values
            Predictions['instance'] = '*'
            Predictions['id'] = Predictions['eid'] + '_*'
            self.Predictions_eids = Predictions.copy()
            self.Predictions_eids['outer_fold'] = -1
            for i in range(self.n_CV_outer_folds):
                Predictions_i = Predictions.copy()
                Predictions_i['outer_fold'] = i
                self.Predictions_eids = self.Predictions_eids.append(Predictions_i)
            
            # Append each model one by one because the folds are different
            print(str(len(self.pred_versions)) + ' models to compute.')
            for pred_version in self.pred_versions:
                if pred_version in self.pred_versions_previous:
                    print(pred_version.replace('pred_', '') + ' had already been computed.')
                else:
                    print("Computing results for version " + pred_version.replace('pred_', ''))
                    Predictions_version = self.Predictions[['eid', pred_version, 'outer_fold']]
                    # Use placeholder for NaN in outer_folds
                    Predictions_version['outer_fold'][Predictions_version['outer_fold'].isna()] = -1
                    Predictions_version_eids = Predictions_version.groupby(['eid', 'outer_fold'], as_index=False).mean()
                    self.Predictions_eids = self.Predictions_eids.merge(Predictions_version_eids,
                                                                        on=['eid', 'outer_fold'], how='outer')
                    self.Predictions_eids[of_version] = self.Predictions_eids['outer_fold']
                    self.Predictions_eids[of_version][self.Predictions_eids[of_version] == -1] = np.nan
                    del Predictions_version
                    _ = gc.collect
            self.Predictions_eids.drop(columns=['outer_fold'], inplace=True)
        else:
            self.Predictions_eids = self.Predictions.groupby('eid').mean()
            self.Predictions_eids['eid'] = self.Predictions_eids.index.values
            self.Predictions_eids['instance'] = '*'
            self.Predictions_eids['id'] = self.Predictions_eids['eid'].astype(str) + '_' + \
                                          self.Predictions_eids['instance']
        
        # Re-order the columns
        self.Predictions_eids = self.Predictions_eids[self.id_vars + self.demographic_vars + self.pred_versions]
    
    def postprocessing(self):
        # Displaying the R2s
        versions = [col.replace('pred_', '') for col in self.Predictions_eids.columns if 'pred_' in col]
        r2s = []
        for version in versions:
            df = self.Predictions_eids[[self.target, 'pred_' + version]].dropna()
            r2s.append(r2_score(df[self.target], df['pred_' + version]))
        R2S = pd.DataFrame({'version': versions, 'R2': r2s})
        R2S.sort_values(by='R2', ascending=False, inplace=True)
        print('R2 for each model: ')
        print(R2S)
    
    def _generate_single_model_predictions(self):
        for pred_version in self.pred_versions:
            path_save = \
                self.path_data + 'Predictions_eids_' + '_'.join(pred_version.split('_')[1:]) + '_' + self.fold + '.csv'
            # Generate only if does not exist already.
            if not os.path.exists(path_save):
                Predictions_version = self.Predictions_eids[['id', 'outer_fold', pred_version]]
                Predictions_version.rename(columns={pred_version: 'pred'}, inplace=True)
                Predictions_version.dropna(subset=['pred'], inplace=True)
                Predictions_version.to_csv(path_save, index=False)
    
    def save_predictions(self):
        self.Predictions_eids.to_csv(self.path_data + 'PREDICTIONS_withoutEnsembles_eids_' + self.target + '_' +
                                     self.fold + '.csv', index=False)
        # Generate and save files for every single model
        self._generate_single_model_predictions()


class PerformancesGenerate(Metrics):
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, architecture=None, n_fc_layers=None,
                 n_fc_nodes=None, optimizer=None, learning_rate=None, weight_decay=None, dropout_rate=None,
                 data_augmentation_factor=None, fold=None, pred_type=None, debug_mode=False):
        
        Metrics.__init__(self)
        
        self.target = target
        self.organ = organ
        self.view = view
        self.transformation = transformation
        self.architecture = architecture
        self.n_fc_layers = n_fc_layers
        self.n_fc_nodes = n_fc_nodes
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.data_augmentation_factor = data_augmentation_factor
        self.fold = fold
        self.pred_type = pred_type
        if debug_mode:
            self.n_bootstrap_iterations = 3
        else:
            self.n_bootstrap_iterations = 1000
        self.version = target + '_' + organ + '_' + view + '_' + transformation + '_' + architecture + '_' + \
                       n_fc_layers + '_' + n_fc_nodes + '_' + optimizer + '_' + learning_rate + '_' + weight_decay + \
                       '_' + dropout_rate + '_' + data_augmentation_factor
        self.names_metrics = self.dict_metrics_names[self.dict_prediction_types[target]]
        self.data_features = None
        self.Predictions = None
        self.PERFORMANCES = None
    
    def _preprocess_data_features_predictions_for_performances(self):
        # load dataset
        data_features = pd.read_csv(self.path_data + 'data-features_' + self.pred_type + '.csv',
                                    usecols=['id', 'Sex', 'Age'])
        # format data_features to extract y
        data_features.rename(columns={self.target: 'y'}, inplace=True)
        data_features = data_features[['id', 'y']]
        data_features['id'] = data_features['id'].astype(str)
        data_features['id'] = data_features['id']
        data_features.set_index('id', drop=False, inplace=True)
        data_features.index.name = 'columns_names'
        self.data_features = data_features
    
    def _preprocess_predictions_for_performances(self):
        Predictions = pd.read_csv(self.path_data + 'Predictions_' + self.pred_type + '_' + self.version + '_' +
                                  self.fold + '.csv')
        Predictions['id'] = Predictions['id'].astype(str)
        self.Predictions = Predictions.merge(self.data_features, how='inner', on=['id'])
    
    # Initialize performances dataframes and compute sample sizes
    def _initiate_empty_performances_df(self):
        # Define an empty performances dataframe to store the performances computed
        row_names = ['all'] + self.outer_folds
        col_names_sample_sizes = ['N']
        if self.target in self.targets_binary:
            col_names_sample_sizes.extend(['N_0', 'N_1'])
        col_names = ['outer_fold'] + col_names_sample_sizes
        col_names.extend(self.names_metrics)
        performances = np.empty((len(row_names), len(col_names),))
        performances.fill(np.nan)
        performances = pd.DataFrame(performances)
        performances.index = row_names
        performances.columns = col_names
        performances['outer_fold'] = row_names
        # Convert float to int for sample sizes and some metrics.
        for col_name in col_names_sample_sizes:
            # need recent version of pandas to use type below. Otherwise nan cannot be int
            performances[col_name] = performances[col_name].astype('Int64')
        
        # compute sample sizes for the data frame
        performances.loc['all', 'N'] = len(self.Predictions.index)
        if self.target in self.targets_binary:
            performances.loc['all', 'N_0'] = len(self.Predictions.loc[self.Predictions['y'] == 0].index)
            performances.loc['all', 'N_1'] = len(self.Predictions.loc[self.Predictions['y'] == 1].index)
        for outer_fold in self.outer_folds:
            performances.loc[outer_fold, 'N'] = len(
                self.Predictions.loc[self.Predictions['outer_fold'] == int(outer_fold)].index)
            if self.target in self.targets_binary:
                performances.loc[outer_fold, 'N_0'] = len(
                    self.Predictions.loc[
                        (self.Predictions['outer_fold'] == int(outer_fold)) & (self.Predictions['y'] == 0)].index)
                performances.loc[outer_fold, 'N_1'] = len(
                    self.Predictions.loc[
                        (self.Predictions['outer_fold'] == int(outer_fold)) & (self.Predictions['y'] == 1)].index)
        
        # initialize the dataframes
        self.PERFORMANCES = {}
        for mode in self.modes:
            self.PERFORMANCES[mode] = performances.copy()
        
        # Convert float to int for sample sizes and some metrics.
        for col_name in self.PERFORMANCES[''].columns.values:
            if any(metric in col_name for metric in self.metrics_displayed_in_int):
                # need recent version of pandas to use type below. Otherwise nan cannot be int
                self.PERFORMANCES[''][col_name] = self.PERFORMANCES[''][col_name].astype('Int64')
    
    def preprocessing(self):
        self._preprocess_data_features_predictions_for_performances()
        self._preprocess_predictions_for_performances()
        self._initiate_empty_performances_df()
    
    # Fill the columns for this model, outer_fold by outer_fold
    def compute_performances(self):
        
        # fill it outer_fold by outer_fold
        for outer_fold in ['all'] + self.outer_folds:
            print('Calculating the performances for the outer fold ' + outer_fold)
            # Generate a subdataframe from the predictions table for each outerfold
            if outer_fold == 'all':
                predictions_fold = self.Predictions.copy()
            else:
                predictions_fold = self.Predictions.loc[self.Predictions['outer_fold'] == int(outer_fold), :]
            
            # if no samples are available for this fold, fill columns with nans
            if len(predictions_fold.index) == 0:
                print('NO SAMPLES AVAILABLE FOR MODEL ' + self.version + ' IN OUTER_FOLD ' + outer_fold)
            else:
                # For binary classification, generate class prediction
                if self.target in self.targets_binary:
                    predictions_fold_class = predictions_fold.copy()
                    predictions_fold_class['pred'] = predictions_fold_class['pred'].round()
                else:
                    predictions_fold_class = None
                
                # Fill the Performances dataframe metric by metric
                for name_metric in self.names_metrics:
                    # print('Calculating the performance using the metric ' + name_metric)
                    if name_metric in self.metrics_needing_classpred:
                        predictions_metric = predictions_fold_class
                    else:
                        predictions_metric = predictions_fold
                    metric_function = self.dict_metrics_sklearn[name_metric]
                    self.PERFORMANCES[''].loc[outer_fold, name_metric] = metric_function(predictions_metric['y'],
                                                                                         predictions_metric['pred'])
                    self.PERFORMANCES['_sd'].loc[outer_fold, name_metric] = \
                        self._bootstrap(predictions_metric, metric_function)[1]
                    self.PERFORMANCES['_str'].loc[outer_fold, name_metric] = "{:.3f}".format(
                        self.PERFORMANCES[''].loc[outer_fold, name_metric]) + '+-' + "{:.3f}".format(
                        self.PERFORMANCES['_sd'].loc[outer_fold, name_metric])
        
        # calculate the fold sd (variance between the metrics values obtained on the different folds)
        folds_sd = self.PERFORMANCES[''].iloc[1:, :].std(axis=0)
        for name_metric in self.names_metrics:
            self.PERFORMANCES['_str'].loc['all', name_metric] = "{:.3f}".format(
                self.PERFORMANCES[''].loc['all', name_metric]) + '+-' + "{:.3f}".format(
                folds_sd[name_metric]) + '+-' + "{:.3f}".format(self.PERFORMANCES['_sd'].loc['all', name_metric])
        
        # print the performances
        print('Performances for model ' + self.version + ': ')
        print(self.PERFORMANCES['_str'])
    
    def save_performances(self):
        for mode in self.modes:
            path_save = self.path_data + 'Performances_' + self.pred_type + '_' + self.version + '_' + self.fold + \
                        mode + '.csv'
            self.PERFORMANCES[mode].to_csv(path_save, index=False)


class PerformancesMerge(Metrics):
    
    def __init__(self, target=None, fold=None, pred_type=None, ensemble_models=None):
        
        # Parameters
        Metrics.__init__(self)
        self.target = target
        self.fold = fold
        self.pred_type = pred_type
        self.ensemble_models = self.convert_string_to_boolean(ensemble_models)
        self.names_metrics = self.dict_metrics_names[self.dict_prediction_types[target]]
        self.main_metric_name = self.dict_main_metrics_names[target]
        # list the models that need to be merged
        self.list_models = glob.glob(self.path_data + 'Performances_' + pred_type + '_' + target + '_*_' + fold +
                                     '_str.csv')
        # get rid of ensemble models
        if self.ensemble_models:
            self.list_models = [model for model in self.list_models if '*' in model]
        else:
            self.list_models = [model for model in self.list_models if '*' not in model]
        self.Performances = None
        self.Performances_alphabetical = None
        self.Performances_ranked = None
    
    def _initiate_empty_performances_summary_df(self):
        # Define the columns of the Performances dataframe
        # columns for sample sizes
        names_sample_sizes = ['N']
        if self.target in self.targets_binary:
            names_sample_sizes.extend(['N_0', 'N_1'])
        
        # columns for metrics
        names_metrics = self.dict_metrics_names[self.dict_prediction_types[self.target]]
        # for normal folds, keep track of metric and bootstrapped metric's sd
        names_metrics_with_sd = []
        for name_metric in names_metrics:
            names_metrics_with_sd.extend([name_metric, name_metric + '_sd', name_metric + '_str'])
        
        # for the 'all' fold, also keep track of the 'folds_sd' (metric's sd calculated using the folds' results)
        names_metrics_with_folds_sd_and_sd = []
        for name_metric in names_metrics:
            names_metrics_with_folds_sd_and_sd.extend(
                [name_metric, name_metric + '_folds_sd', name_metric + '_sd', name_metric + '_str'])
        
        # merge all the columns together. First description of the model, then sample sizes and metrics for each fold
        names_col_Performances = ['version'] + self.names_model_parameters
        # special outer fold 'all'
        names_col_Performances.extend(
            ['_'.join([name, 'all']) for name in names_sample_sizes + names_metrics_with_folds_sd_and_sd])
        # other outer_folds
        for outer_fold in self.outer_folds:
            names_col_Performances.extend(
                ['_'.join([name, outer_fold]) for name in names_sample_sizes + names_metrics_with_sd])
        
        # Generate the empty Performance table from the rows and columns.
        Performances = np.empty((len(self.list_models), len(names_col_Performances),))
        Performances.fill(np.nan)
        Performances = pd.DataFrame(Performances)
        Performances.columns = names_col_Performances
        # Format the types of the columns
        for colname in Performances.columns.values:
            if (colname in self.names_model_parameters) | ('_str' in colname):
                col_type = str
            else:
                col_type = float
            Performances[colname] = Performances[colname].astype(col_type)
        self.Performances = Performances
    
    def merge_performances(self):
        # define parameters
        names_metrics = self.dict_metrics_names[self.dict_prediction_types[self.target]]
        
        # initiate dataframe
        self._initiate_empty_performances_summary_df()
        
        # Fill the Performance table row by row
        for i, model in enumerate(self.list_models):
            # load the performances subdataframe
            PERFORMANCES = {}
            for mode in self.modes:
                PERFORMANCES[mode] = pd.read_csv(model.replace('_str', mode))
                PERFORMANCES[mode].set_index('outer_fold', drop=False, inplace=True)
            
            # Fill the columns corresponding to the model's parameters
            version = '_'.join(model.split('_')[2:-2])
            parameters = self._version_to_parameters(version)
            
            # fill the columns for model parameters
            self.Performances['version'][i] = version
            for parameter_name in self.names_model_parameters:
                self.Performances[parameter_name][i] = parameters[parameter_name]
            
            # Fill the columns for this model, outer_fold by outer_fold
            for outer_fold in ['all'] + self.outer_folds:
                # Generate a subdataframe from the predictions table for each outerfold
                
                # Fill sample size columns
                self.Performances['N_' + outer_fold][i] = PERFORMANCES[''].loc[outer_fold, 'N']
                
                # For binary classification, calculate sample sizes for each class and generate class prediction
                if self.target in self.targets_binary:
                    self.Performances['N_0_' + outer_fold][i] = PERFORMANCES[''].loc[outer_fold, 'N_0']
                    self.Performances['N_1_' + outer_fold][i] = PERFORMANCES[''].loc[outer_fold, 'N_1']
                
                # Fill the Performances dataframe metric by metric
                for name_metric in names_metrics:
                    for mode in self.modes:
                        self.Performances[name_metric + mode + '_' + outer_fold][i] = PERFORMANCES[mode].loc[
                            outer_fold, name_metric]
                
                # calculate the fold sd (variance between the metrics values obtained on the different folds)
                folds_sd = PERFORMANCES[''].iloc[1:, :].std(axis=0)
                for name_metric in names_metrics:
                    self.Performances[name_metric + '_folds_sd_all'] = folds_sd[name_metric]
        
        # Convert float to int for sample sizes and some metrics.
        for name_col in self.Performances.columns.values:
            cond1 = name_col.startswith('N_')
            cond2 = any(metric in name_col for metric in self.metrics_displayed_in_int)
            cond3 = '_sd' not in name_col
            cond4 = '_str' not in name_col
            if cond1 | cond2 & cond3 & cond4:
                self.Performances[name_col] = self.Performances[name_col].astype('Int64')
                # need recent version of pandas to use this type. Otherwise nan cannot be int
        
        # For ensemble models, merge the new performances with the previously computed performances
        if self.ensemble_models:
            Performances_withoutEnsembles = pd.read_csv(self.path_data + 'PERFORMANCES_tuned_alphabetical_' +
                                                        self.pred_type + '_' + self.target + '_' + self.fold + '.csv')
            self.Performances = Performances_withoutEnsembles.append(self.Performances)
            # reorder the columns (weird: automatic alphabetical re-ordering happened when append was called for 'val')
            self.Performances = self.Performances[Performances_withoutEnsembles.columns]
        
        # Ranking, printing and saving
        self.Performances_alphabetical = self.Performances.sort_values(by='version')
        cols_to_print = ['version', self.main_metric_name + '_str_all']
        print('Performances of the models ranked by models\'names:')
        print(self.Performances_alphabetical[cols_to_print])
        sort_by = self.dict_main_metrics_names[self.target] + '_all'
        sort_ascending = self.main_metrics_modes[self.dict_main_metrics_names[self.target]] == 'min'
        self.Performances_ranked = self.Performances.sort_values(by=sort_by, ascending=sort_ascending)
        print('Performances of the models ranked by the performance on the main metric on all the samples:')
        print(self.Performances_ranked[cols_to_print])
    
    def save_performances(self):
        name_extension = 'withEnsembles' if self.ensemble_models else 'withoutEnsembles'
        path = self.path_data + 'PERFORMANCES_' + name_extension + '_alphabetical_' + self.pred_type + '_' + \
               self.target + '_' + self.fold + '.csv'
        self.Performances_alphabetical.to_csv(path, index=False)
        self.Performances_ranked.to_csv(path.replace('_alphabetical_', '_ranked_'), index=False)


class PerformancesTuning(Metrics):
    
    def __init__(self, target=None, pred_type=None):
        
        Metrics.__init__(self)
        self.target = target
        self.pred_type = pred_type
        self.PERFORMANCES = {}
        self.PREDICTIONS = {}
        self.Performances = None
        self.models = None
        self.folds = ['val', 'test']
    
    def load_data(self):
        for fold in self.folds:
            path = self.path_data + 'PERFORMANCES_withoutEnsembles_ranked_' + self.pred_type + '_' + self.target + \
                   '_' + fold + '.csv'
            self.PERFORMANCES[fold] = pd.read_csv(path).set_index('version', drop=False)
            self.PERFORMANCES[fold]['organ'] = self.PERFORMANCES[fold]['organ'].astype(str)
            self.PERFORMANCES[fold].index.name = 'columns_names'
            self.PREDICTIONS[fold] = pd.read_csv(path.replace('PERFORMANCES', 'PREDICTIONS').replace('_ranked', ''))
    
    def preprocess_data(self):
        # Get list of distinct models without taking into account hyperparameters tuning
        self.Performances = self.PERFORMANCES['val']
        self.Performances['model'] = self.Performances['organ'] + '_' + self.Performances['view'] + '_' + \
                                     self.Performances['transformation'] + '_' + self.Performances['architecture']
        self.models = self.Performances['model'].unique()
    
    def select_models(self):
        main_metric_name = self.dict_main_metrics_names[self.target]
        main_metric_mode = self.main_metrics_modes[main_metric_name]
        Perf_col_name = main_metric_name + '_all'
        for model in self.models:
            Performances_model = self.Performances[self.Performances['model'] == model]
            Performances_model.sort_values([Perf_col_name, 'n_fc_layers', 'n_fc_nodes', 'learning_rate', 'dropout_rate',
                                            'weight_decay', 'data_augmentation_factor'],
                                           ascending=[main_metric_mode == 'min', True, True, False, False, False,
                                                      False], inplace=True)
            best_version = Performances_model['version'][
                Performances_model[Perf_col_name] == Performances_model[Perf_col_name].max()].values[0]
            versions_to_drop = [version for version in Performances_model['version'].values if
                                not version == best_version]
            # define columns from predictions to drop
            cols_to_drop = ['pred_' + version for version in versions_to_drop] + ['outer_fold_' + version for version in
                                                                                  versions_to_drop]
            for fold in self.folds:
                self.PERFORMANCES[fold].drop(versions_to_drop, inplace=True)
                self.PREDICTIONS[fold].drop(cols_to_drop, axis=1, inplace=True)
        
        # drop 'model' column
        self.Performances.drop(['model'], axis=1, inplace=True)
        
        # Display results
        for fold in self.folds:
            print('The tuned ' + fold + ' performances are:')
            print(self.PERFORMANCES[fold])
    
    def save_data(self):
        # Save the files
        for fold in self.folds:
            path_pred = self.path_data + 'PREDICTIONS_tuned_' + self.pred_type + '_' + self.target + '_' + fold + \
                        '.csv'
            path_perf = self.path_data + 'PERFORMANCES_tuned_ranked_' + self.pred_type + '_' + self.target + '_' + \
                        fold + '.csv'
            self.PREDICTIONS[fold].to_csv(path_pred, index=False)
            self.PERFORMANCES[fold].to_csv(path_perf, index=False)
            Performances_alphabetical = self.PERFORMANCES[fold].sort_values(by='version')
            Performances_alphabetical.to_csv(path_perf.replace('ranked', 'alphabetical'), index=False)


class PerformancesSurvival(Metrics):
    
    def __init__(self, target=None, fold=None, pred_type=None, debug_mode=None):
        Metrics.__init__(self)
        
        self.target = target
        self.fold = fold
        self.pred_type = pred_type
        if debug_mode:
            self.n_bootstrap_iterations = 3
        else:
            self.n_bootstrap_iterations = 1000
        self.PERFORMANCES = None
        self.Survival = None
        self.SURV = None
    
    def _bootstrap_c_index(self, data):
        results = []
        for i in range(self.n_bootstrap_iterations):
            data_i = resample(data, replace=True, n_samples=len(data.index))
            if len(data_i['Death'].unique()) == 2:
                try:
                    results.append(concordance_index(data_i['Age'], -data_i['pred'], data_i['Death']))
                except: #TODO remove
                    print('WEIRD, should not happen! Printing the df')
                    print(data_i)
                    self.data_i_debug = data_i
                    break
        if len(results) > 0:
            results_mean = np.mean(results)
            results_std = np.std(results)
        else:
            results_mean = np.nan
            results_std = np.nan
        return results_mean, results_std
    
    def load_data(self):
        # Load and preprocess PERFORMANCES
        self.PERFORMANCES = pd.read_csv(self.path_data + 'PERFORMANCES_withEnsembles_alphabetical_' +
                                        self.pred_type + '_' + self.target + '_' + self.fold + '.csv')
        self.PERFORMANCES.set_index('version', drop=False, inplace=True)
        self.PERFORMANCES.index.name = 'index'
        for inner_fold in ['all'] + [str(i) for i in range(10)]:
            for metric in ['C-Index', 'C-Index-difference']:
                for mode in self.modes:
                    self.PERFORMANCES[metric + mode + '_' + inner_fold] = np.nan
        
        Residuals = pd.read_csv(
            self.path_data + 'RESIDUALS_' + self.pred_type + '_' + self.target + '_' + self.fold + '.csv')
        Survival = pd.read_csv(self.path_data + 'data_survival.csv')
        self.Survival = pd.merge(Survival[['id', 'FollowUpTime', 'Death']], Residuals, on='id')
        data_folds = pd.read_csv(self.path_data + 'data-features_eids.csv', usecols=['eid', 'outer_fold'])
        self.SURV = {}
        for i in range(10):
            self.SURV[i] = \
                self.Survival[self.Survival['eid'].isin(data_folds['eid'][data_folds['outer_fold'] == i].values)]
    
    def compute_c_index_and_save_data(self):
        models = [col.replace('res_' + self.target, self.target) for col in self.Survival.columns if 'res_' in col]
        for k, model in enumerate(models):
            print(model) #TODO remove
            if k % 30 == 0:
                print('Computing CI for the ' + str(k) + 'th model out of ' + str(len(models)) + ' models.')
            # Load Performances dataframes
            PERFS = {}
            for mode in self.modes:
                PERFS[mode] = pd.read_csv('../data/Performances_' + self.pred_type + '_' + model + '_' + self.fold +
                                          mode + '.csv')
                PERFS[mode].set_index('outer_fold', drop=False, inplace=True)
                PERFS[mode]['C-Index'] = np.nan
                PERFS[mode]['C-Index-difference'] = np.nan
            df_model = self.Survival[['FollowUpTime', 'Death', 'Age', 'res_' + model]].dropna()
            df_model.rename(columns={'res_' + model: 'pred'}, inplace=True)
            # Compute CI over all samples
            if len(df_model['Death'].unique()) == 2:
                ci_model = concordance_index(df_model['FollowUpTime'], -(df_model['Age'] - df_model['pred']),
                                             df_model['Death'])
                ci_age = concordance_index(df_model['FollowUpTime'], -df_model['Age'], df_model['Death'])
                ci_diff = ci_model - ci_age
                PERFS[''].loc['all', 'C-Index'] = ci_model
                PERFS[''].loc['all', 'C-Index-difference'] = ci_diff
                self.PERFORMANCES.loc[model, 'C-Index_all'] = ci_model
                self.PERFORMANCES.loc[model, 'C-Index-difference_all'] = ci_diff
                _, ci_sd = self._bootstrap_c_index(df_model)
                PERFS['_sd'].loc['all', 'C-Index'] = ci_sd
                PERFS['_sd'].loc['all', 'C-Index-difference'] = ci_sd
                self.PERFORMANCES.loc[model, 'C-Index_sd_all'] = ci_sd
                self.PERFORMANCES.loc[model, 'C-Index-difference_sd_all'] = ci_sd
            # Compute CI over each fold
            for i in range(10):
                df_model_i = self.SURV[i][['FollowUpTime', 'Death', 'Age', 'res_' + model]].dropna()
                df_model_i.rename(columns={'res_' + model: 'pred'}, inplace=True)
                if len(df_model_i['Death'].unique()) == 2:
                    ci_model_i = concordance_index(df_model_i['FollowUpTime'],
                                                   -(df_model_i['Age'] - df_model_i['pred']),
                                                   df_model_i['Death'])
                    ci_age_i = concordance_index(df_model_i['FollowUpTime'], -df_model_i['Age'], df_model_i['Death'])
                    ci_diff_i = ci_model_i - ci_age_i
                    PERFS[''].loc[str(i), 'C-Index'] = ci_model_i
                    PERFS[''].loc[str(i), 'C-Index-difference'] = ci_diff_i
                    self.PERFORMANCES.loc[model, 'C-Index_' + str(i)] = ci_model_i
                    self.PERFORMANCES.loc[model, 'C-Index-difference_' + str(i)] = ci_diff_i
                    _, ci_i_sd = self._bootstrap_c_index(df_model_i)
                    PERFS['_sd'].loc[str(i), 'C-Index'] = ci_i_sd
                    PERFS['_sd'].loc[str(i), 'C-Index-difference'] = ci_i_sd
                    self.PERFORMANCES.loc[model, 'C-Index_sd_' + str(i)] = ci_i_sd
                    self.PERFORMANCES.loc[model, 'C-Index-difference_sd_' + str(i)] = ci_i_sd
            # Compute sd using all folds
            ci_str = round(PERFS[''][['C-Index', 'C-Index-difference']], 3).astype(str) + '+-' + \
                     round(PERFS['_sd'][['C-Index', 'C-Index-difference']], 3).astype(str)
            PERFS['_str'][['C-Index', 'C-Index-difference']] = ci_str
            for col in ['C-Index', 'C-Index-difference']:
                cols = [col + '_str_' + str(i) for i in range(10)]
                # Fill model's performance matrix
                ci_std_lst = PERFS['_str'].loc['all', col].split('+-')
                ci_std_lst.insert(1, str(round(PERFS[''][col].iloc[1:].std(), 3)))
                ci_std_str = '+-'.join(ci_std_lst)
                PERFS['_str'].loc['all', col] = ci_std_str
                # Fill global performances matrix
                self.PERFORMANCES.loc[model, cols] = ci_str[col].values[1:]
                self.PERFORMANCES.loc[model, col + '_str_all'] = ci_std_str
            # Save new performances
            for mode in self.modes:
                PERFS[mode].to_csv('../data/Performances_' + self.pred_type + '_withCI_' + model + '_' + self.fold +
                                   mode + '.csv')
        
        # Save PERFORMANCES dataframes
        self.PERFORMANCES.to_csv(self.path_data + 'PERFORMANCES_withCI_withEnsembles_alphabetical_' +
                                 self.pred_type + '_' + self.target + '_' + self.fold + '.csv', index=False)
        
        # Ranking, printing and saving
        # Sort by alphabetical order
        Performances_alphabetical = self.PERFORMANCES.sort_values(by='version')
        Performances_alphabetical.to_csv(self.path_data + 'PERFORMANCES_withEnsembles_withCI_alphabetical_' +
                                         self.pred_type + '_' + self.target + '_' + self.fold + '.csv', index=False)
        # Sort by C-Index difference, to print
        cols_to_print = ['version', 'C-Index-difference_str_all']
        Performances_ranked = self.PERFORMANCES.sort_values(by='C-Index-difference_all', ascending=False)
        print('Performances of the models ranked by C-Index difference with C-Index based on age only,'
              ' on all the samples:')
        print(Performances_ranked[cols_to_print])
        # Sort by main metric, to save
        sort_by = self.dict_main_metrics_names[self.target] + '_all'
        sort_ascending = self.main_metrics_modes[self.dict_main_metrics_names[self.target]] == 'min'
        Performances_ranked = self.PERFORMANCES.sort_values(by=sort_by, ascending=sort_ascending)
        Performances_ranked.to_csv(self.path_data + 'PERFORMANCES_withEnsembles_withCI_withEnsembles_ranked_' +
                                        self.pred_type + '_' + self.target + '_' + self.fold + '.csv', index=False)
        # Save with ensembles
        models_nonensembles = [idx for idx in Performances_alphabetical.index if '*' not in idx]
        path_save = self.path_data + 'PERFORMANCES_withoutEnsembles_withCI_alphabetical_' + self.pred_type + '_' + \
                    self.target + '_' + self.fold + '.csv'
        Performances_alphabetical.loc[models_nonensembles, :].to_csv(path_save, index=False)
        Performances_ranked.loc[models_nonensembles, :].to_csv(path_save.replace('alphabetical', 'ranked'))


# This class was coded by Samuel Diai.
class InnerCV:
    def __init__(self, models, inner_splits, n_iter):
        self.inner_splits = inner_splits
        self.n_iter = n_iter
        if isinstance(models, str):
            models = [models]
        self.models = models
    
    @staticmethod
    def get_model(model_name, params):
        if model_name == 'ElasticNet':
            return ElasticNet(max_iter=2000, **params)
        elif model_name == 'RandomForest':
            return RandomForestRegressor(**params)
        elif model_name == 'GradientBoosting':
            return GradientBoostingRegressor(**params)
        elif model_name == 'Xgboost':
            return XGBRegressor(**params)
        elif model_name == 'LightGbm':
            return LGBMRegressor(**params)
        elif model_name == 'NeuralNetwork':
            return MLPRegressor(solver='adam',
                                activation='relu',
                                hidden_layer_sizes=(128, 64, 32),
                                batch_size=1000,
                                early_stopping=True, **params)
    
    @staticmethod
    def get_hyper_distribution(model_name):
        
        if model_name == 'ElasticNet':
            return {
                'alpha': hp.loguniform('alpha', low=np.log(0.01), high=np.log(10)),
                'l1_ratio': hp.uniform('l1_ratio', low=0.01, high=0.99)
            }
        elif model_name == 'RandomForest':
            return {
                'n_estimators': hp.randint('n_estimators', upper=300) + 150,
                'max_features': hp.choice('max_features', ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
                'max_depth': hp.choice('max_depth', [None, 10, 8, 6])
            }
        elif model_name == 'GradientBoosting':
            return {
                'n_estimators': hp.randint('n_estimators', upper=300) + 150,
                'max_features': hp.choice('max_features', ['auto', 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
                'learning_rate': hp.uniform('learning_rate', low=0.01, high=0.3),
                'max_depth': hp.randint('max_depth', 10) + 5
            }
        elif model_name == 'Xgboost':
            return {
                'colsample_bytree': hp.uniform('colsample_bytree', low=0.2, high=0.7),
                'gamma': hp.uniform('gamma', low=0.1, high=0.5),
                'learning_rate': hp.uniform('learning_rate', low=0.02, high=0.2),
                'max_depth': hp.randint('max_depth', 10) + 5,
                'n_estimators': hp.randint('n_estimators', 300) + 150,
                'subsample': hp.uniform('subsample', 0.2, 0.8)
            }
        elif model_name == 'LightGbm':
            return {
                'num_leaves': hp.randint('num_leaves', 40) + 5,
                'min_child_samples': hp.randint('min_child_samples', 400) + 100,
                'min_child_weight': hp.choice('min_child_weight', [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]),
                'subsample': hp.uniform('subsample', low=0.2, high=0.8),
                'colsample_bytree': hp.uniform('colsample_bytree', low=0.4, high=0.6),
                'reg_alpha': hp.choice('reg_alpha', [0, 1e-1, 1, 2, 5, 7, 10, 50, 100]),
                'reg_lambda': hp.choice('reg_lambda', [0, 1e-1, 1, 5, 10, 20, 50, 100]),
                'n_estimators': hp.randint('n_estimators', 300) + 150
            }
        elif model_name == 'NeuralNetwork':
            return {
                'learning_rate_init': hp.loguniform('learning_rate_init', low=np.log(5e-5), high=np.log(2e-2)),
                'alpha': hp.uniform('alpha', low=1e-6, high=1e3)
            }
    
    def create_folds(self, X, y):
        """
        X columns : eid + features except target
        y columns : eid + target
        """
        X_eid = X.drop_duplicates('eid')
        y_eid = y.drop_duplicates('eid')
        eids = X_eid.eid
        
        # Kfold on the eid, then regroup all ids
        inner_cv = KFold(n_splits=self.inner_splits, shuffle=False, random_state=0)
        list_test_folds = [elem[1] for elem in inner_cv.split(X_eid, y_eid)]
        list_test_folds_eid = [eids[elem].values for elem in list_test_folds]
        list_test_folds_id = [X.index[X.eid.isin(list_test_folds_eid[elem])].values for elem in
                              range(len(list_test_folds_eid))]
        return list_test_folds_id
    
    def optimize_hyperparameters(self, X, y, scoring):
        """
        input X  : dataframe with features + eid
        input y : dataframe with target + eid
        """
        if 'instance' in X.columns:
            X = X.drop(columns=['instance'])
        if 'instance' in y.columns:
            y = y.drop(columns=['instance'])
        list_test_folds_id = self.create_folds(X, y)
        X = X.drop(columns=['eid'])
        y = y.drop(columns=['eid'])
        
        # Create custom Splits
        list_test_folds_id_index = [np.array([X.index.get_loc(elem) for elem in list_test_folds_id[fold_num]])
                                    for fold_num in range(len(list_test_folds_id))]
        test_folds = np.zeros(len(X), dtype='int')
        for fold_count in range(len(list_test_folds_id)):
            test_folds[list_test_folds_id_index[fold_count]] = fold_count
        inner_cv = PredefinedSplit(test_fold=test_folds)
        
        list_best_params = {}
        list_best_score = {}
        objective, model_name = None, None
        for model_name in self.models:
            def objective(hyperparameters):
                estimator_ = self.get_model(model_name, hyperparameters)
                pipeline = Pipeline([('scaler', StandardScaler()), ('estimator', estimator_)])
                scores = cross_validate(pipeline, X.values, y, scoring=scoring, cv=inner_cv, n_jobs=self.inner_splits)
                return {'status': STATUS_OK, 'loss': -scores['test_score'].mean(),
                        'attachments': {'split_test_scores_and_params': (scores['test_score'], hyperparameters)}}
            space = self.get_hyper_distribution(model_name)
            trials = Trials()
            best = fmin(objective, space, algo=tpe.suggest, max_evals=self.n_iter, trials=trials)
            best_params = space_eval(space, best)
            list_best_params[model_name] = best_params
            list_best_score[model_name] = - min(trials.losses())
        
        # Recover best between all models :
        best_model = max(list_best_score.keys(), key=(lambda k: list_best_score[k]))
        best_model_hyp = list_best_params[best_model]
        
        # Recreate best estim :
        estim = self.get_model(best_model, best_model_hyp)
        pipeline_best = Pipeline([('scaler', StandardScaler()), ('estimator', estim)])
        pipeline_best.fit(X.values, y)
        return pipeline_best


# Useful for EnsemblesPredictions. This function needs to be global to allow pool to pickle it.
def compute_ensemble_folds(ensemble_inputs):
    if len(ensemble_inputs[1]) < 100:
        print('Small sample size:' + str(len(ensemble_inputs[1])))
        n_inner_splits = 5
    else:
        n_inner_splits = 10
    # Can use different models: models=['ElasticNet', 'LightGBM', 'NeuralNetwork']
    cv = InnerCV(models=['ElasticNet'], inner_splits=n_inner_splits, n_iter=30)
    model = cv.optimize_hyperparameters(ensemble_inputs[0], ensemble_inputs[1], scoring='r2')
    return model


class EnsemblesPredictions(Metrics):
    
    def __init__(self, target=None, pred_type=None, regenerate_models=False):
        # Parameters
        Metrics.__init__(self)
        self.target = target
        self.pred_type = pred_type
        self.folds = ['val', 'test']
        self.regenerate_models = regenerate_models
        self.ensembles_performance_cutoff_percent = 0.5
        self.parameters = {'target': target, 'organ': '*', 'view': '*', 'transformation': '*', 'architecture': '*',
                           'n_fc_layers': '*', 'n_fc_nodes': '*', 'optimizer': '*', 'learning_rate': '*',
                           'weight_decay': '*', 'dropout_rate': '*', 'data_augmentation_factor': '*'}
        self.version = self._parameters_to_version(self.parameters)
        self.main_metric_name = self.dict_main_metrics_names[target]
        self.init_perf = -np.Inf if self.main_metrics_modes[self.main_metric_name] == 'max' else np.Inf
        path_perf = self.path_data + 'PERFORMANCES_tuned_ranked_' + pred_type + '_' + target + '_val.csv'
        self.Performances = pd.read_csv(path_perf).set_index('version', drop=False)
        self.Performances['organ'] = self.Performances['organ'].astype(str)
        self.list_ensemble_levels = ['transformation', 'view', 'organ']
        self.PREDICTIONS = {}
        self.weights_by_category = None
        self.weights_by_ensembles = None
        self.N_ensemble_CV_split = 10
        self.instancesS = {'instances': ['01', '1.5x', '23'], 'eids': ['*']}
        self.instances_names_to_numbers = {'01': ['0', '1'], '1.5x': ['1.5', '1.51', '1.52', '1.53', '1.54'],
                                           '23': ['2', '3'], '*': ['*']}
        self.INSTANCES_DATASETS = {
            '01': ['Eyes', 'Hearing', 'Lungs', 'Arterial', 'Musculoskeletal', 'Biochemistry', 'ImmuneSystem'],
            '1.5x': ['PhysicalActivity'],
            '23': ['Brain', 'Arterial', 'Heart', 'Abdomen', 'Musculoskeletal'],
            '*': ['Brain', 'Eyes', 'Hearing', 'Lungs', 'Arterial', 'Heart', 'Abdomen', 'Musculoskeletal',
                  'PhysicalActivity', 'ImmuneSystem']
        }
    
    # Get rid of columns and rows for the versions for which all samples as NANs
    @staticmethod
    def _drop_na_pred_versions(PREDS, Performances):
        # Select the versions for which only NAs are available
        pred_versions = [col for col in PREDS['val'].columns.values if 'pred_' in col]
        to_drop = []
        for pv in pred_versions:
            for fold in PREDS.keys():
                if PREDS[fold][pv].notna().sum() == 0:
                    to_drop.append(pv)
                    break
        
        # Drop the corresponding columns from preds, and rows from performances
        index_to_drop = [p.replace('pred_', '') for p in to_drop if '*' not in p]
        for fold in PREDS.keys():
            PREDS[fold].drop(to_drop, axis=1, inplace=True)
        return Performances.drop(index_to_drop)
    
    def load_data(self):
        for fold in self.folds:
            self.PREDICTIONS[fold] = pd.read_csv(
                self.path_data + 'PREDICTIONS_tuned_' + self.pred_type + '_' + self.target + '_' + fold + '.csv')
    
    def _build_single_ensemble(self, PREDICTIONS, version):
        # Drop columns that are exclusively NaNs
        all_nan = PREDICTIONS['val'].isna().all() | PREDICTIONS['test'].isna().all()
        non_nan_cols = all_nan[~all_nan.values].index
        for fold in self.folds:
            PREDICTIONS[fold] = PREDICTIONS[fold][non_nan_cols]
        Predictions = PREDICTIONS['val']
        # Select the columns for the model
        ensemble_preds_cols = [col for col in Predictions.columns.values if
                                    bool(re.compile('pred_' + version).match(col))]
        
        # If only one model in the ensemble, just copy the column. Otherwise build the ensemble model
        if len(ensemble_preds_cols) == 1:
            for fold in self.folds:
                PREDICTIONS[fold]['pred_' + version] = PREDICTIONS[fold][ensemble_preds_cols[0]]
        else:
            # Initiate the dictionaries
            PREDICTIONS_OUTERFOLDS = {}
            ENSEMBLE_INPUTS = {}
            for outer_fold in self.outer_folds:
                # take the subset of the rows that correspond to the outer_fold
                PREDICTIONS_OUTERFOLDS[outer_fold] = {}
                XS_outer_fold = {}
                YS_outer_fold = {}
                dict_fold_to_outer_folds = {
                    'val': [float(outer_fold)],
                    'test': [(float(outer_fold) + 1) % self.n_CV_outer_folds],
                    'train': [float(of) for of in self.outer_folds
                              if float(of) not in [float(outer_fold), (float(outer_fold) + 1) % self.n_CV_outer_folds]]
                }
                for fold in self.folds:
                    PREDICTIONS_OUTERFOLDS[outer_fold][fold] = \
                        PREDICTIONS[fold][PREDICTIONS[fold]['outer_fold'].isin(dict_fold_to_outer_folds[fold])]
                    PREDICTIONS_OUTERFOLDS[outer_fold][fold] = PREDICTIONS_OUTERFOLDS[outer_fold][fold][
                        ['id', 'eid', 'instance', self.target] + ensemble_preds_cols].dropna()
                    X = PREDICTIONS_OUTERFOLDS[outer_fold][fold][['id', 'eid', 'instance'] + ensemble_preds_cols]
                    X.set_index('id', inplace=True)
                    XS_outer_fold[fold] = X
                    y = PREDICTIONS_OUTERFOLDS[outer_fold][fold][['id', 'eid', self.target]]
                    y.set_index('id', inplace=True)
                    YS_outer_fold[fold] = y
                ENSEMBLE_INPUTS[outer_fold] = [XS_outer_fold['val'], YS_outer_fold['val']]
            
            # Build ensemble model using ElasticNet and/or LightGBM, Neural Network.
            PREDICTIONS_ENSEMBLE = {}
            pool = Pool(self.N_ensemble_CV_split)
            MODELS = pool.map(compute_ensemble_folds, list(ENSEMBLE_INPUTS.values()))
            pool.close()
            pool.join()
            
            # Concatenate all outer folds
            for outer_fold in self.outer_folds:
                for fold in self.folds:
                    X = PREDICTIONS_OUTERFOLDS[outer_fold][fold][ensemble_preds_cols]
                    PREDICTIONS_OUTERFOLDS[outer_fold][fold]['pred_' + version] = MODELS[int(outer_fold)].predict(X)
                    PREDICTIONS_OUTERFOLDS[outer_fold][fold]['outer_fold'] = float(outer_fold)
                    df_outer_fold = PREDICTIONS_OUTERFOLDS[outer_fold][fold][['id', 'outer_fold',
                                                                              'pred_' + version]]
                    # Initiate, or append if some previous outerfolds have already been concatenated
                    if fold not in PREDICTIONS_ENSEMBLE.keys():
                        PREDICTIONS_ENSEMBLE[fold] = df_outer_fold
                    else:
                        PREDICTIONS_ENSEMBLE[fold] = PREDICTIONS_ENSEMBLE[fold].append(df_outer_fold)
    
            # Add the ensemble predictions to the dataframe
            for fold in self.folds:
                if fold == 'train':
                    PREDICTIONS[fold] = PREDICTIONS[fold].merge(PREDICTIONS_ENSEMBLE[fold], how='outer',
                                                                on=['id', 'outer_fold'])
                else:
                    PREDICTIONS_ENSEMBLE[fold].drop('outer_fold', axis=1, inplace=True)
                    PREDICTIONS[fold] = PREDICTIONS[fold].merge(PREDICTIONS_ENSEMBLE[fold], how='outer', on=['id'])
    
    def _build_single_ensemble_wrapper(self, version, ensemble_level):
        print('Building the ensemble model ' + version)
        pred_version = 'pred_' + version
        # Evaluate if the ensemble model should be built
        # 1 - separately on instance 0-1, 1.5 and 2-3 (for ensemble at the top level, since overlap between models is 0)
        # 2 - piece by piece on each outer_fold
        # 1-Compute instances 0-1, 1.5 and 2-3 separately
        if ensemble_level == 'organ':
            for fold in self.folds:
                self.PREDICTIONS[fold][pred_version] = np.nan
                # Add an ensemble for each instances (01, 1.5x, and 23)
                if self.pred_type == 'instances':
                    for instances_names in self.instancesS[self.pred_type]:
                        pv = 'pred_' + version.split('_')[0] + '_*instances' + instances_names + '_' + \
                             '_'.join(version.split('_')[2:])
                        self.PREDICTIONS[fold][pv] = np.nan
            
            for instances_names in self.instancesS[self.pred_type]:
                print('Building final ensemble model for samples in the instances: ' + instances_names)
                # Take subset of rows and columns
                instances = self.instances_names_to_numbers[instances_names]
                instances_datasets = self.INSTANCES_DATASETS[instances_names]
                versions = \
                    [col.replace('pred_', '') for col in self.PREDICTIONS['val'].columns if 'pred_' in col]
                instances_versions = [version for version in versions
                                      if any(dataset in version for dataset in instances_datasets)]
                cols_to_keep = self.id_vars + self.demographic_vars + \
                               ['pred_' + version for version in instances_versions]
                PREDICTIONS = {}
                for fold in self.folds:
                    PREDICTIONS[fold] = self.PREDICTIONS[fold][self.PREDICTIONS[fold].instance.isin(instances)]
                    PREDICTIONS[fold] = PREDICTIONS[fold][cols_to_keep]
                self._build_single_ensemble(PREDICTIONS, version)

                # Print a quick performance estimation for each instance(s)
                df_model = PREDICTIONS['test'][[self.target, 'pred_' + version]].dropna()
                print(instances_names)
                print(self.main_metric_name + ' for instance(s) ' + instances_names + ': ' +
                      str(r2_score(df_model[self.target], df_model['pred_' + version])))
                print('The sample size is ' + str(len(df_model.index)) + '.')
                
                # Add the predictions to the dataframe, chunck by chunk, instances by instances
                for fold in self.folds:
                    self.PREDICTIONS[fold][pred_version][self.PREDICTIONS[fold].instance.isin(instances)] = \
                        PREDICTIONS[fold][pred_version].values
                    # Add an ensemble for the instance(s) only
                    if self.pred_type == 'instances':
                        pv = 'pred_' + version.split('_')[0] + '_*instances' + instances_names + '_' + \
                             '_'.join(version.split('_')[2:])
                        self.PREDICTIONS[fold][pv][self.PREDICTIONS[fold].instance.isin(instances)] = \
                            PREDICTIONS[fold][pred_version].values
            
            # Add three extra ensemble models for eids, to allow larger sample sizes for GWAS purposes
            if self.pred_type == 'eids':
                for instances_names in ['01', '1.5x', '23']:
                    print('Building final sub-ensemble model for samples in the instances: ' + instances_names)
                    # Keep only relevant columns
                    instances_datasets = self.INSTANCES_DATASETS[instances_names]
                    versions = \
                        [col.replace('pred_', '') for col in self.PREDICTIONS['val'].columns if 'pred_' in col]
                    instances_versions = [version for version in versions
                                          if any(dataset in version for dataset in instances_datasets)]
                    cols_to_keep = self.id_vars + self.demographic_vars + \
                                   ['pred_' + version for version in instances_versions]
                    PREDICTIONS = {}
                    for fold in self.folds:
                        PREDICTIONS[fold] = self.PREDICTIONS[fold].copy()
                        PREDICTIONS[fold] = PREDICTIONS[fold][cols_to_keep]
                    self._build_single_ensemble(PREDICTIONS, version)
                    
                    # Print a quick performance estimation for each instance(s)
                    df_model = PREDICTIONS['test'][[self.target, 'pred_' + version]].dropna()
                    print(instances_names)
                    print(self.main_metric_name + ' for instance(s) ' + instances_names + ': ' +
                          str(r2_score(df_model[self.target], df_model['pred_' + version])))
                    print('The sample size is ' + str(len(df_model.index)) + '.')
                    
                    # Add the predictions to the dataframe
                    pv = 'pred_' + version.split('_')[0] + '_*instances' + instances_names + '_' + \
                         '_'.join(version.split('_')[2:])
                    for fold in self.folds:
                        self.PREDICTIONS[fold][pv] = PREDICTIONS[fold][pred_version].values
        
        # 2-Compute fold by fold
        else:
            self._build_single_ensemble(self.PREDICTIONS, version)
        
        # build and save a dataset for this specific ensemble model
        for fold in self.folds:
            df_single_ensemble = self.PREDICTIONS[fold][['id', 'outer_fold', pred_version]]
            df_single_ensemble.rename(columns={pred_version: 'pred'}, inplace=True)
            df_single_ensemble.dropna(inplace=True, subset=['pred'])
            df_single_ensemble.to_csv(self.path_data + 'Predictions_' + self.pred_type + '_' + version + '_' + fold +
                                      '.csv', index=False)
            # Add extra ensembles at organ level
            if ensemble_level == 'organ':
                for instances_names in ['01', '1.5x', '23']:
                    pv = 'pred_' + version.split('_')[0] + '_*instances' + instances_names + '_' + \
                         '_'.join(version.split('_')[2:])
                    version_instances = version.split('_')[0] + '_*instances' + instances_names + '_' + \
                         '_'.join(version.split('_')[2:])
                    df_single_ensemble = self.PREDICTIONS[fold][['id', 'outer_fold', pv]]
                    df_single_ensemble.rename(columns={pv: 'pred'}, inplace=True)
                    df_single_ensemble.dropna(inplace=True, subset=['pred'])
                    df_single_ensemble.to_csv(self.path_data + 'Predictions_' + self.pred_type + '_' +
                                              version_instances + '_' + fold + '.csv', index=False)
    
    def _recursive_ensemble_builder(self, Performances_grandparent, parameters_parent, version_parent,
                                    list_ensemble_levels_parent):
        # Compute the ensemble models for the children first, so that they can be used for the parent model
        Performances_parent = Performances_grandparent[
            Performances_grandparent['version'].isin(
                fnmatch.filter(Performances_grandparent['version'], version_parent))]
        # if the last ensemble level has not been reached, go down one level and create a branch for each child.
        # Otherwise the leaf has been reached
        if len(list_ensemble_levels_parent) > 0:
            list_ensemble_levels_child = list_ensemble_levels_parent.copy()
            ensemble_level = list_ensemble_levels_child.pop()
            list_children = Performances_parent[ensemble_level].unique()
            for child in list_children:
                parameters_child = parameters_parent.copy()
                parameters_child[ensemble_level] = child
                version_child = self._parameters_to_version(parameters_child)
                # recursive call to the function
                self._recursive_ensemble_builder(Performances_parent, parameters_child, version_child,
                                                 list_ensemble_levels_child)
        else:
            ensemble_level = None
        
        # compute the ensemble model for the parent
        # Check if ensemble model has already been computed. If it has, load the predictions. If it has not, compute it.
        if not self.regenerate_models and \
                os.path.exists(self.path_data + 'Predictions_' + self.pred_type + '_' + version_parent + '_test.csv'):
            print('The model ' + version_parent + ' has already been computed. Loading it...')
            for fold in self.folds:
                df_single_ensemble = pd.read_csv(self.path_data + 'Predictions_' + self.pred_type + '_' +
                                                 version_parent + '_' + fold + '.csv')
                df_single_ensemble.rename(columns={'pred': 'pred_' + version_parent}, inplace=True)
                # Add the ensemble predictions to the dataframe
                if fold == 'train':
                    self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(df_single_ensemble, how='outer',
                                                                          on=['id', 'outer_fold'])
                else:
                    df_single_ensemble.drop(columns=['outer_fold'], inplace=True)
                    self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(df_single_ensemble, how='outer', on=['id'])
                # Add the extra ensemble models at the 'organ' level
                if ensemble_level == 'organ':
                    if self.pred_type == 'instances':
                        instances = self.instancesS[self.pred_type]
                    else:
                        instances = ['01', '23']
                    for instances_names in instances:
                        pv = 'pred_' + version_parent.split('_')[0] + '_*instances' + instances_names + '_' + \
                             '_'.join(version_parent.split('_')[2:])
                        version_instances = version_parent.split('_')[0] + '_*instances' + instances_names + '_' + \
                                            '_'.join(version_parent.split('_')[2:])
                        df_single_ensemble = pd.read_csv(self.path_data + 'Predictions_' + self.pred_type + '_' +
                                                         version_instances + '_' + fold + '.csv')
                        df_single_ensemble.rename(columns={'pred': pv}, inplace=True)
                        if fold == 'train':
                            self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(df_single_ensemble, how='outer',
                                                                                  on=['id', 'outer_fold'])
                        else:
                            df_single_ensemble.drop(columns=['outer_fold'], inplace=True)
                            self.PREDICTIONS[fold] = self.PREDICTIONS[fold].merge(df_single_ensemble, how='outer',
                                                                                  on=['id'])
        else:
            self._build_single_ensemble_wrapper(version_parent, ensemble_level)
        
        # Print a quick performance estimation
        df_model = self.PREDICTIONS['test'][[self.target, 'pred_' + version_parent]].dropna()
        print(self.main_metric_name + ': ' + str(r2_score(df_model[self.target], df_model['pred_' + version_parent])))
        print('The sample size is ' + str(len(df_model.index)) + '.')
    
    def generate_ensemble_predictions(self):
        self._recursive_ensemble_builder(self.Performances, self.parameters, self.version, self.list_ensemble_levels)
        
        # Reorder the columns alphabetically
        for fold in self.folds:
            pred_versions = [col for col in self.PREDICTIONS[fold].columns if 'pred_' in col]
            pred_versions.sort()
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold][self.id_vars + self.demographic_vars + pred_versions]
        
        # Displaying the R2s
        for fold in self.folds:
            versions = [col.replace('pred_', '') for col in self.PREDICTIONS[fold].columns if 'pred_' in col]
            r2s = []
            for version in versions:
                df = self.PREDICTIONS[fold][[self.target, 'pred_' + version]].dropna()
                r2s.append(r2_score(df[self.target], df['pred_' + version]))
            R2S = pd.DataFrame({'version': versions, 'R2': r2s})
            R2S.sort_values(by='R2', ascending=False, inplace=True)
            print(fold + ' R2s for each model: ')
            print(R2S)
    
    def save_predictions(self):
        for fold in self.folds:
            self.PREDICTIONS[fold].to_csv(self.path_data + 'PREDICTIONS_withEnsembles_' + self.pred_type + '_' +
                                          self.target + '_' + fold + '.csv', index=False)


class ResidualsGenerate(Basics):
    
    def __init__(self, target=None, fold=None, pred_type=None, debug_mode=False):
        # Parameters
        Basics.__init__(self)
        self.target = target
        self.fold = fold
        self.pred_type = pred_type
        self.debug_mode = debug_mode
        self.Residuals = pd.read_csv(self.path_data + 'PREDICTIONS_withEnsembles_' + pred_type + '_' + target + '_' +
                                     fold + '.csv')
        self.list_models = [col_name.replace('pred_', '') for col_name in self.Residuals.columns.values
                            if 'pred_' in col_name]
    
    def generate_residuals(self):
        list_models = [col_name.replace('pred_', '') for col_name in self.Residuals.columns.values
                       if 'pred_' in col_name]
        for model in list_models:
            print('Generating residuals for model ' + model)
            df_model = self.Residuals[['Age', 'pred_' + model]]
            no_na_indices = [not b for b in df_model['pred_' + model].isna()]
            df_model.dropna(inplace=True)
            if (len(df_model.index)) > 0:
                age = df_model.loc[:, ['Age']]
                res = df_model['Age'] - df_model['pred_' + model]
                regr = LinearRegression()
                regr.fit(age, res)
                res_correction = regr.predict(age)
                res_corrected = res - res_correction
                self.Residuals.loc[no_na_indices, 'pred_' + model] = res_corrected
                # debug plot
                if self.debug_mode:
                    print('Bias for the residuals ' + model, regr.coef_)
                    plt.scatter(age, res)
                    plt.scatter(age, res_corrected)
                    regr2 = LinearRegression()
                    regr2.fit(age, res_corrected)
                    print('Coefficients after: \n', regr2.coef_)
        self.Residuals.rename(columns=lambda x: x.replace('pred_', 'res_'), inplace=True)
    
    def save_residuals(self):
        self.Residuals.to_csv(self.path_data + 'RESIDUALS_' + self.pred_type + '_' + self.target + '_' + self.fold +
                              '.csv', index=False)


class ResidualsCorrelations(Basics):
    
    def __init__(self, target=None, fold=None, pred_type=None, debug_mode=False):
        Basics.__init__(self)
        self.target = target
        self.fold = fold
        self.pred_type = pred_type
        self.debug_mode = debug_mode
        if debug_mode:
            self.n_bootstrap_iterations_correlations = 10
        else:
            self.n_bootstrap_iterations_correlations = 1000
        self.Residuals = None
        self.CORRELATIONS = {}
        self.Correlation_sample_sizes = None
    
    def preprocessing(self):
        # load data
        Residuals = pd.read_csv(self.path_data + 'RESIDUALS_' + self.pred_type + '_' + self.target + '_' + self.fold +
                                '.csv')
        # Format the dataframe
        Residuals_only = Residuals[[col_name for col_name in Residuals.columns.values if 'res_' in col_name]]
        Residuals_only.rename(columns=lambda x: x.replace('res_' + self.target + '_', ''), inplace=True)
        # Reorder the columns to make the correlation matrix more readable
        # Need to temporarily rename '?' because its ranking differs from the '*' and ',' characters
        Residuals_only.columns = [col_name.replace('?', ',placeholder') for col_name in Residuals_only.columns.values]
        Residuals_only = Residuals_only.reindex(sorted(Residuals_only.columns), axis=1)
        Residuals_only.columns = [col_name.replace(',placeholder', '?') for col_name in Residuals_only.columns.values]
        self.Residuals = Residuals_only
    
    def _bootstrap_correlations(self):
        names = self.Residuals.columns.values
        results = []
        for i in range(self.n_bootstrap_iterations_correlations):
            if (i + 1) % 100 == 0:
                print('Bootstrap iteration ' + str(i + 1) + ' out of ' + str(self.n_bootstrap_iterations_correlations))
            data_i = resample(self.Residuals, replace=True, n_samples=len(self.Residuals.index))
            results.append(np.array(data_i.corr()))
        results = np.array(results)
        RESULTS = {}
        for op in ['mean', 'std']:
            results_op = pd.DataFrame(getattr(np, op)(results, axis=0))
            results_op.index = names
            results_op.columns = names
            RESULTS[op] = results_op
        self.CORRELATIONS['_sd'] = RESULTS['std']
    
    def generate_correlations(self):
        # Generate the correlation matrix
        self.CORRELATIONS[''] = self.Residuals.corr()
        # Gerate the std by bootstrapping
        self._bootstrap_correlations()
        # Merge both as a dataframe of strings
        self.CORRELATIONS['_str'] = self.CORRELATIONS[''].round(3).applymap(str) \
                                    + '+-' + self.CORRELATIONS['_sd'].round(3).applymap(str)
        # Print correlations
        print(self.CORRELATIONS[''])
        
        # Generate correlation sample sizes
        self.Residuals[~self.Residuals.isna()] = 1
        self.Residuals[self.Residuals.isna()] = 0
        self.Correlation_sample_sizes = self.Residuals.transpose() @ self.Residuals
    
    def save_correlations(self):
        self.Correlation_sample_sizes.to_csv(self.path_data + 'ResidualsCorrelations_samplesizes_' + self.pred_type +
                                             '_' + self.target + '_' + self.fold + '.csv', index=True)
        for mode in self.modes:
            self.CORRELATIONS[mode].to_csv(self.path_data + 'ResidualsCorrelations' + mode + '_' + self.pred_type +
                                           '_' + self.target + '_' + self.fold + '.csv', index=True)


class SelectBest(Metrics):
    
    def __init__(self, target=None, pred_type=None):
        Metrics.__init__(self)
        self.target = target
        self.pred_type = pred_type
        self.folds = ['test']
        self.organs_with_suborgans = {'Brain': ['Cognitive', 'MRI'], 'Eyes': ['All', 'Fundus', 'OCT'],
                                      'Arterial': ['PulseWaveAnalysis', 'Carotids'],
                                      'Heart': ['ECG', 'MRI'], 'Abdomen': ['Liver', 'Pancreas'],
                                      'Musculoskeletal': ['Spine', 'Hips', 'Knees', 'FullBody', 'Scalars'],
                                      'Biochemistry': ['Urine', 'Blood']}
        self.organs = []
        self.best_models = []
        self.PREDICTIONS = {}
        self.RESIDUALS = {}
        self.PERFORMANCES = {}
        self.CORRELATIONS = {}
        self.CORRELATIONS_SAMPLESIZES = {}
    
    def _load_data(self):
        for fold in self.folds:
            path_pred = self.path_data + 'PREDICTIONS_withEnsembles_' + self.pred_type + '_' + self.target + '_' + \
                        fold + '.csv'
            path_res = self.path_data + 'RESIDUALS_' + self.pred_type + '_' + self.target + '_' + fold + '.csv'
            path_perf = self.path_data + 'PERFORMANCES_withEnsembles_withCI_ranked_' + self.pred_type + '_' + \
                        self.target + '_' + fold + '.csv'
            path_corr = self.path_data + 'ResidualsCorrelations_str_' + self.pred_type + '_' + self.target + '_' + \
                        fold + '.csv'
            self.PREDICTIONS[fold] = pd.read_csv(path_pred)
            self.RESIDUALS[fold] = pd.read_csv(path_res)
            self.PERFORMANCES[fold] = pd.read_csv(path_perf)
            self.PERFORMANCES[fold].set_index('version', drop=False, inplace=True)
            self.CORRELATIONS_SAMPLESIZES[fold] = pd.read_csv(self.path_data + 'ResidualsCorrelations_samplesizes_' +
                                                              self.pred_type + '_' + self.target + '_' + fold + '.csv',
                                                              index_col=0)
            self.CORRELATIONS[fold] = {}
            for mode in self.modes:
                self.CORRELATIONS[fold][mode] = pd.read_csv(path_corr.replace('_str', mode), index_col=0)
    
    def _select_versions(self):
        # Load val performances
        path_perf = self.path_data + 'PERFORMANCES_withEnsembles_withCI_ranked_' + self.pred_type + '_' + \
                    self.target + '_test.csv'
        Performances = pd.read_csv(path_perf)
        Performances.set_index('version', drop=False, inplace=True)
        list_organs = Performances['organ'].unique()
        list_organs.sort()
        for organ in list_organs:
            print('Selecting best model for ' + organ)
            Perf_organ = Performances[Performances['organ'] == organ]
            self.organs.append(organ)
            self.best_models.append(Perf_organ['version'].values[0])
            if organ in self.organs_with_suborgans.keys():
                for view in self.organs_with_suborgans[organ]:
                    print('Selecting best model for ' + organ + view)
                    Perf_organview = Performances[(Performances['organ'] == organ) & (Performances['view'] == view)]
                    self.organs.append(organ + view)
                    self.best_models.append(Perf_organview['version'].values[0])
    
    def _take_subsets(self):
        base_cols = self.id_vars + self.demographic_vars
        best_models_pred = ['pred_' + model for model in self.best_models]
        best_models_res = ['res_' + model for model in self.best_models]
        best_models_corr = ['_'.join(model.split('_')[1:]) for model in self.best_models]
        for fold in self.folds:
            self.PREDICTIONS[fold] = self.PREDICTIONS[fold].loc[:, base_cols + best_models_pred]
            self.PREDICTIONS[fold].columns = base_cols + self.organs
            self.RESIDUALS[fold] = self.RESIDUALS[fold].loc[:, base_cols + best_models_res]
            self.RESIDUALS[fold].columns = base_cols + self.organs
            self.PERFORMANCES[fold] = self.PERFORMANCES[fold].loc[self.best_models, :]
            self.PERFORMANCES[fold].index = self.organs
            self.CORRELATIONS_SAMPLESIZES[fold] = \
                self.CORRELATIONS_SAMPLESIZES[fold].loc[best_models_corr, best_models_corr]
            self.CORRELATIONS_SAMPLESIZES[fold].index = self.organs
            self.CORRELATIONS_SAMPLESIZES[fold].columns = self.organs
            for mode in self.modes:
                self.CORRELATIONS[fold][mode] = self.CORRELATIONS[fold][mode].loc[best_models_corr, best_models_corr]
                self.CORRELATIONS[fold][mode].index = self.organs
                self.CORRELATIONS[fold][mode].columns = self.organs
    
    def select_models(self):
        self._load_data()
        self._select_versions()
        self._take_subsets()
    
    def save_data(self):
        for fold in self.folds:
            path_pred = self.path_data + 'PREDICTIONS_bestmodels_' + self.pred_type + '_' + self.target + '_' + fold \
                        + '.csv'
            path_res = self.path_data + 'RESIDUALS_bestmodels_' + self.pred_type + '_' + self.target + '_' + fold + \
                       '.csv'
            path_corr = self.path_data + 'ResidualsCorrelations_bestmodels_str_' + self.pred_type + '_' + self.target \
                        + '_' + fold + '.csv'
            path_perf = self.path_data + 'PERFORMANCES_bestmodels_ranked_' + self.pred_type + '_' + self.target + '_' \
                        + fold + '.csv'
            self.PREDICTIONS[fold].to_csv(path_pred, index=False)
            self.RESIDUALS[fold].to_csv(path_res, index=False)
            self.PERFORMANCES[fold].sort_values(by=self.dict_main_metrics_names[self.target] + '_all', ascending=False,
                                                inplace=True)
            self.PERFORMANCES[fold].to_csv(path_perf, index=False)
            Performances_alphabetical = self.PERFORMANCES[fold].sort_values(by='version')
            Performances_alphabetical.to_csv(path_perf.replace('ranked', 'alphabetical'), index=False)
            for mode in self.modes:
                self.CORRELATIONS[fold][mode].to_csv(path_corr.replace('_str', mode), index=True)


class SelectCorrelationsNAs(Basics):
    
    def __init__(self, target=None):
        Basics.__init__(self)
        self.target = target
        self.folds = ['test']
        self.CORRELATIONS = {'*': {'': {}, '_sd': {}, '_str': {}}}
    
    def load_data(self):
        for models_type in self.models_types:
            self.CORRELATIONS[models_type] = {}
            for pred_type in ['instances', 'eids', '*']:
                self.CORRELATIONS[models_type][pred_type] = {}
                for mode in self.modes:
                    self.CORRELATIONS[models_type][pred_type][mode] = {}
                    for fold in self.folds:
                        if pred_type == '*':
                            self.CORRELATIONS[models_type][pred_type][mode][fold] = \
                                pd.read_csv(self.path_data + 'ResidualsCorrelations' + models_type + mode +
                                            '_instances_' + self.target + '_' + fold + '.csv', index_col=0)
                        else:
                            self.CORRELATIONS[models_type][pred_type][mode][fold] = \
                                pd.read_csv(self.path_data + 'ResidualsCorrelations' + models_type + mode + '_' +
                                            pred_type + '_' + self.target + '_' + fold + '.csv', index_col=0)
    
    def fill_na(self):
        # Dectect NAs in the instances correlation matrix
        for models_type in self.models_types:
            NAs_mask = self.CORRELATIONS[models_type]['instances']['']['test'].isna()
            for mode in self.modes:
                for fold in self.folds:
                    self.CORRELATIONS[models_type]['*'][mode][fold] = \
                        self.CORRELATIONS[models_type]['instances'][mode][fold].copy()
                    self.CORRELATIONS[models_type]['*'][mode][fold][NAs_mask] = \
                        self.CORRELATIONS[models_type]['eids'][mode][fold][NAs_mask]
    
    def save_correlations(self):
        for models_type in self.models_types:
            for mode in self.modes:
                for fold in self.folds:
                    self.CORRELATIONS[models_type]['*'][mode][fold].to_csv(self.path_data + 'ResidualsCorrelations' +
                                                                           models_type + mode + '_*_' + self.target +
                                                                           '_' + fold + '.csv', index=True)


class CorrelationsAverages:
    
    def __init__(self):
        self.Performances = pd.read_csv("../data/PERFORMANCES_withEnsembles_ranked_eids_Age_test.csv")
        self.Correlations = pd.read_csv("../data/ResidualsCorrelations_eids_Age_test.csv", index_col=0)
    
    def _melt_correlation_matrix(self, models):
        models = ['_'.join(c.split('_')[1:]) for c in models]
        Corrs = self.Correlations.loc[models, models]
        Corrs = Corrs.where(np.triu(np.ones(Corrs.shape), 1).astype(np.bool))
        Corrs = Corrs.stack().reset_index()
        Corrs.columns = ['Row', 'Column', 'Correlation']
        return Corrs
    
    @staticmethod
    def _split_version(row):
        names = ['organ', 'view', 'transformation', 'architecture', 'n_fc_layers', 'n_fc_nodes', 'optimizer',
                 'learning_rate', 'weight_decay', 'dropout_rate', 'data_augmentation_factor']
        row_names = ['row_' + name for name in names]
        col_names = ['col_' + name for name in names]
        row_params = row['Row'].split('_')
        col_params = row['Column'].split('_')
        new_row = pd.Series(row_params + col_params + [row['Correlation']])
        new_row.index = row_names + col_names + ['Correlation']
        return new_row
    
    @staticmethod
    def _compute_stats(data, title):
        m = data['Correlation'].mean()
        s = data['Correlation'].std()
        n = len(data.index)
        print('Correlation between ' + title + ': ' + str(round(m, 3)) + '+-' + str(round(s, 3)) + ', n_pairs=' +
              str(n))
    
    @staticmethod
    def _generate_pairs(ls):
        pairs = []
        for i in range(len(ls)):
            for j in range((i + 1), len(ls)):
                pairs.append((ls[i], ls[j]))
        return pairs
    
    @staticmethod
    def _extract_pair(Corrs, pair, level):
        extracted = Corrs[((Corrs['row_' + level] == pair[0]) & (Corrs['col_' + level] == pair[1])) |
                          ((Corrs['row_' + level] == pair[1]) & (Corrs['col_' + level] == pair[0]))]
        return extracted
    
    def _extract_pairs(self, Corrs, pairs, level):
        extracted = None
        for pair in pairs:
            extracted_pair = self._extract_pair(Corrs, pair, level)
            if extracted is None:
                extracted = extracted_pair
            else:
                extracted = extracted.append(extracted_pair)
        return extracted
    
    def correlations_all(self):
        Corrs = self._melt_correlation_matrix(self.Performances['version'].values)
        self._compute_stats(Corrs, 'All models')
    
    def correlations_dimensions(self):
        Perf = self.Performances[(self.Performances['view'] == '*') &
                                 ~(self.Performances['organ'].isin(['*', '*instances01', '*instances1.5x',
                                                                    '*instances23']))]
        Corrs = self._melt_correlation_matrix(Perf['version'].values)
        self._compute_stats(Corrs, 'Main Dimensions')
    
    def correlations_subdimensions(self):
        # Subdimensions
        dict_dims_to_subdims = {
            'Brain': ['Cognitive', 'MRI'],
            'Eyes': ['OCT', 'Fundus', 'IntraocularPressure', 'Acuity', 'Autorefraction'],
            'Arterial': ['PulseWaveAnalysis', 'Carotids'],
            'Heart': ['ECG', 'MRI'],
            'Abdomen': ['Liver', 'Pancreas'],
            'Musculoskeletal': ['Spine', 'Hips', 'Knees', 'FullBody', 'Scalars'],
            'PhysicalActivity': ['FullWeek', 'Walking'],
            'Biochemistry': ['Urine', 'Blood']
        }
        Corrs_subdim = None
        for dim in dict_dims_to_subdims.keys():
            models = ['Age_' + dim + '_' + subdim + '_*' * 9 for subdim in dict_dims_to_subdims[dim]]
            Corrs_dim = self._melt_correlation_matrix(models)
            self._compute_stats(Corrs_dim, dim + ' subdimensions')
            if Corrs_subdim is None:
                Corrs_subdim = Corrs_dim
            else:
                Corrs_subdim = Corrs_subdim.append(Corrs_dim)
        
        # Compute the average over the subdimensions
        self._compute_stats(Corrs_subdim, 'Subdimensions')
    
    def correlations_subsubdimensions(self):
        # Only select the ensemble models at the architecture level,
        Perf_ss = self.Performances[self.Performances['architecture'] == '*']
        
        # Brain - Cognitive
        Perf = Perf_ss[(Perf_ss['organ'] == 'Brain') & (Perf_ss['view'] == 'Cognitive')]
        # Remove ensemble model and all scalars
        Perf = Perf[~Perf['transformation'].isin(['*', 'AllScalars'])]
        Corrs_bc = self._melt_correlation_matrix(Perf['version'].values)
        self._compute_stats(Corrs_bc, 'Brain cognitive sub-subdimensions')
        
        # Musculoskeletal - Scalars
        Perf = Perf_ss[(Perf_ss['organ'] == 'Musculoskeletal') & (Perf_ss['view'] == 'Scalars')]
        Perf = Perf[~Perf['transformation'].isin(['*', 'AllScalars'])]
        Corrs_ms = self._melt_correlation_matrix(Perf['version'].values)
        self._compute_stats(Corrs_ms, 'Musculoskeletal - Scalars sub-subdimensions')
        
        # Average over subsubdimensions
        Corrs_subsubdimensions = Corrs_bc.append(Corrs_ms)
        self._compute_stats(Corrs_subsubdimensions, 'Sub-subdimensions')
    
    def correlations_views(self):
        # Variables
        dict_dim_to_view = {
            'Brain_MRI': [['SagittalReference', 'CoronalReference', 'TransverseReference', 'dMRIWeightedMeans',
                           'SubcorticalVolumes', 'GreyMatterVolumes'],
                          ['SagittalRaw', 'CoronalRaw', 'TransverseRaw']],
            'Arterial_PulseWaveAnalysis': [['Scalars', 'TimeSeries']],
            'Arterial_Carotids': [['Scalars', 'LongAxis', 'CIMT120', 'CIMT150', 'ShortAxis']],
            'Heart_ECG': [['Scalars', 'TimeSeries']],
            'Heart_MRI': [['2chambersRaw', '3chambersRaw', '4chambersRaw'],
                          ['2chambersContrast', '3chambersContrast', '4chambersContrast']],
            'Musculoskeletal_Spine': [['Sagittal', 'Coronal']],
            'Musculoskeletal_FullBody': [['Figure', 'Flesh']],
            'PhysicalActivity_FullWeek': [
                ['Scalars', 'Acceleration', 'TimeSeriesFeatures', 'GramianAngularField1minDifference',
                 'GramianAngularField1minSummation', 'MarkovTransitionField1min',
                 'RecurrencePlots1min']]
        }
        
        Corrs_views = None
        for dim in dict_dim_to_view.keys():
            Corrs_dims = None
            for i, views in enumerate(dict_dim_to_view[dim]):
                models = ['Age_' + dim + '_' + view + '_*' * 8 for view in dict_dim_to_view[dim][i]]
                Corrs_dim = self._melt_correlation_matrix(models)
                if Corrs_dims is None:
                    Corrs_dims = Corrs_dim
                else:
                    Corrs_dims = Corrs_dims.append(Corrs_dim)
            self._compute_stats(Corrs_dims, dim + ' views')
            
            if Corrs_views is None:
                Corrs_views = Corrs_dims
            else:
                Corrs_views = Corrs_views.append(Corrs_dims)
        
        # Compute the average over the views
        self._compute_stats(Corrs_views, 'Views')
    
    def correlations_transformations(self):
        # Raw vs. Contrast (Heart MRI, Abdomen Liver, Abdomen Pancreas), Raw vs. Reference (Brain MRI),
        # Figure vs Skeleton (Musculoskeltal FullBody)
        # Filter out the models that are ensembles at the architecture level
        models_to_keep = [model for model in self.Correlations.index.values if model.split('_')[3] != '*']
        # Select only the models that are Heart MRI, Abdomen, or Brain MRI
        models_to_keep = [model for model in models_to_keep if
                          ((model.split('_')[0] == 'Abdomen') & (model.split('_')[1] in ['Liver', 'Pancreas'])) |
                          ((model.split('_')[0] == 'Brain') & (model.split('_')[1] == 'MRI')) |
                          ((model.split('_')[0] == 'Heart') & (model.split('_')[1] == 'MRI')) |
                          ((model.split('_')[0] == 'Musculoskeletal') & (model.split('_')[1] == 'FullBody') &
                           (model.split('_')[2] in ['Figure', 'Skeleton']))]
        # Select only the models that have the relevant preprocessing/transformations
        models_to_keep = [model for model in models_to_keep if model.split('_')[2] in
                          ['Raw', 'Contrast', '2chambersRaw', '2chambersContrast', '3chambersRaw', '3chambersContrast',
                           '4chambersRaw', '4chambersContrast', 'SagittalRaw', 'SagittalReference', 'CoronalRaw',
                           'CoronalReference', 'TransverseRaw', 'TransverseReference', 'Figure', 'Skeleton']]
        # Select the corresponding rows and columns
        Corrs = self.Correlations.loc[models_to_keep, models_to_keep]
        # Melt correlation matrix to dataframe
        Corrs = Corrs.where(np.triu(np.ones(Corrs.shape), 1).astype(np.bool))
        Corrs = Corrs.stack().reset_index()
        Corrs.columns = ['Row', 'Column', 'Correlation']
        Corrs = Corrs.apply(self._split_version, axis=1)
        # Only keep the models that have the same organ, view and architecture
        Corrs = Corrs[(Corrs['row_organ'] == Corrs['col_organ']) & (Corrs['row_view'] == Corrs['col_view']) &
                      (Corrs['row_architecture'] == Corrs['col_architecture'])]
        
        # Define preprocessing pairs
        dict_preprocessing = {
            'Raw-Reference': [('SagittalRaw', 'SagittalReference'), ('CoronalRaw', 'CoronalReference'),
                              ('TransverseRaw', 'TransverseReference')],
            'Raw-Contrast': [('Raw', 'Contrast'), ('2chambersRaw', '2chambersContrast'),
                             ('3chambersRaw', '3chambersContrast'), ('4chambersRaw', '4chambersContrast')],
            'Figure-Skeleton': [('Figure', 'Skeleton')]
        }
        
        # Compute average correlation between each pair of transformations
        Corrs_transformations = None
        for comparison in dict_preprocessing.keys():
            Corrs_comp = self._extract_pairs(Corrs, dict_preprocessing[comparison], 'transformation')
            print(comparison)
            print(Corrs_comp)
            self._compute_stats(Corrs_comp, comparison)
            if Corrs_transformations is None:
                Corrs_transformations = Corrs_comp
            else:
                Corrs_transformations = Corrs_transformations.append(Corrs_comp)
        
        # Compute average correlation between transformations
        self._compute_stats(Corrs_transformations, 'Transformations')
    
    def correlations_algorithms(self):
        # Variables
        algorithms_scalars = ['ElasticNet', 'LightGBM', 'NeuralNetwork']
        algorithms_images = ['InceptionV3', 'InceptionResNetV2']
        
        # Filter out the ensemble models (at the level of the algorithm)
        models_to_keep = [model for model in self.Correlations.index.values if model.split('_')[3] != '*']
        Corrs = self.Correlations.loc[models_to_keep, models_to_keep]
        # Melt correlation matrix to dataframe
        Corrs = Corrs.where(np.triu(np.ones(Corrs.shape), 1).astype(np.bool))
        Corrs = Corrs.stack().reset_index()
        Corrs.columns = ['Row', 'Column', 'Correlation']
        Corrs = Corrs.apply(self._split_version, axis=1)
        # Select the rows for which everything is identical aside from the dataset
        for name in ['organ', 'view', 'transformation']:
            Corrs = Corrs[Corrs['row_' + name] == Corrs['col_' + name]]
        
        # Compute average correlation between algorithms
        self._compute_stats(Corrs, 'Algorithms')
        algorithms_pairs = self._generate_pairs(algorithms_scalars) + self._generate_pairs(algorithms_images)
        # Compute average correlation between each algorithm pair
        for pair in algorithms_pairs:
            Corrs_pair = self._extract_pair(Corrs, pair, 'architecture')
            self._compute_stats(Corrs_pair, pair[0] + ' and ' + pair[1])


class AttentionMaps(DeepLearning):
    
    def __init__(self, target=None, organ=None, view=None, transformation=None, debug_mode=False):
        # Partial initialization with placeholders to get access to parameters and functions
        DeepLearning.__init__(self, 'Age', 'Abdomen', 'Liver', 'Raw', 'InceptionResNetV2', '1', '1024', 'Adam',
                              '0.0001', '0.1', '0.5', '1.0', False)
        # Parameters
        self.target = target
        self.organ = organ
        self.view = view
        self.transformation = transformation
        self.version = None
        self.leftright = True if self.organ + '_' + self.view in self.left_right_organs_views else False
        self.parameters = None
        self.image_width = None
        self.image_height = None
        self.batch_size = None
        self.N_samples_attentionmaps = 10  # needs to be > 1 for the script to work
        if debug_mode:
            self.N_samples_attentionmaps = 2
        self.dir_images = '../images/' + organ + '/' + view + '/' + transformation + '/'
        self.prediction_type = self.dict_prediction_types[target]
        self.Residuals = None
        self.df_to_plot = None
        self.df_outer_fold = None
        self.class_mode = None
        self.image = None
        self.generator = None
        self.dict_architecture_to_last_conv_layer_name = \
            {'VGG16': 'block5_conv3', 'VGG19': 'block5_conv4', 'MobileNet': 'conv_pw_13_relu',
             'MobileNetV2': 'out_relu', 'DenseNet121': 'relu', 'DenseNet169': 'relu', 'DenseNet201': 'relu',
             'NASNetMobile': 'activation_1136', 'NASNetLarge': 'activation_1396', 'Xception': 'block14_sepconv2_act',
             'InceptionV3': 'mixed10', 'InceptionResNetV2': 'conv_7b_ac', 'EfficientNetB7': 'top_activation'}
        self.last_conv_layer = None
        self.organs_views_transformations_images = \
            ['Brain_MRI_SagittalRaw', 'Brain_MRI_SagittalReference', 'Brain_MRI_CoronalRaw',
             'Brain_MRI_CoronalReference', 'Brain_MRI_TransverseRaw', 'Brain_MRI_TransverseReference',
             'Eyes_Fundus_Raw', 'Eyes_OCT_Raw', 'Arterial_Carotids_Mixed', 'Arterial_Carotids_LongAxis',
             'Arterial_Carotids_CIMT120', 'Arterial_Carotids_CIMT150', 'Arterial_Carotids_ShortAxis',
             'Heart_MRI_2chambersRaw', 'Heart_MRI_2chambersContrast', 'Heart_MRI_3chambersRaw',
             'Heart_MRI_3chambersContrast', 'Heart_MRI_4chambersRaw', 'Heart_MRI_4chambersContrast',
             'Abdomen_Liver_Raw', 'Abdomen_Liver_Contrast', 'Abdomen_Pancreas_Raw', 'Abdomen_Pancreas_Contrast',
             'Musculoskeletal_Spine_Sagittal', 'Musculoskeletal_Spine_Coronal', 'Musculoskeletal_Hips_MRI',
             'Musculoskeletal_Knees_MRI', 'Musculoskeletal_FullBody_Mixed', 'Musculoskeletal_FullBody_Figure',
             'Musculoskeletal_FullBody_Skeleton', 'Musculoskeletal_FullBody_Flesh',
             'PhysicalActivity_FullWeek_GramianAngularField1minDifference',
             'PhysicalActivity_FullWeek_GramianAngularField1minSummation',
             'PhysicalActivity_FullWeek_MarkovTransitionField1min', 'PhysicalActivity_FullWeek_RecurrencePlots1min']
    
    def _select_best_model(self):
        # Pick the best model based on the performances
        path_perf = self.path_data + 'PERFORMANCES_withoutEnsembles_ranked_instances_' + self.target + '_test.csv'
        Performances = pd.read_csv(path_perf).set_index('version', drop=False)
        Performances = Performances[(Performances['organ'] == self.organ)
                                    & (Performances['view'] == self.view)
                                    & (Performances['transformation'] == self.transformation)]
        self.version = Performances['version'].values[0]
        del Performances
        # other parameters
        self.parameters = self._version_to_parameters(self.version)
        if self.organ + '_' + self.view + '_' + self.transformation in self.organs_views_transformations_images:
            DeepLearning.__init__(self, self.parameters['target'], self.parameters['organ'], self.parameters['view'],
                                  self.parameters['transformation'], self.parameters['architecture'],
                                  self.parameters['n_fc_layers'], self.parameters['n_fc_nodes'],
                                  self.parameters['optimizer'], self.parameters['learning_rate'],
                                  self.parameters['weight_decay'], self.parameters['dropout_rate'],
                                  self.parameters['data_augmentation_factor'], False)
    
    def _format_residuals(self):
        # Format the residuals
        Residuals_full = pd.read_csv(self.path_data + 'RESIDUALS_instances_' + self.target + '_test.csv')
        Residuals = Residuals_full[['id', 'outer_fold'] + self.demographic_vars + ['res_' + self.version]]
        del Residuals_full
        Residuals.dropna(inplace=True)
        Residuals.rename(columns={'res_' + self.version: 'res'}, inplace=True)
        Residuals.set_index('id', drop=False, inplace=True)
        Residuals['outer_fold'] = Residuals['outer_fold'].astype(int).astype(str)
        Residuals['res_abs'] = Residuals['res'].abs()
        self.Residuals = Residuals
    
    def _select_representative_samples(self):
        # Select with samples to plot
        print('Selecting representative samples...')
        df_to_plot = None
        
        # Sex
        dict_sexes_to_values = {'Male': 1, 'Female': 0}
        for sex in ['Male', 'Female']:
            print('Sex: ' + sex)
            Residuals_sex = self.Residuals[self.Residuals['Sex'] == dict_sexes_to_values[sex]]
            Residuals_sex['sex'] = sex
            
            # Age category
            for age_category in ['young', 'middle', 'old']:
                print('Age category: ' + age_category)
                if age_category == 'young':
                    Residuals_age = Residuals_sex[Residuals_sex['Age'] <= Residuals_sex['Age'].min() + 10]
                elif age_category == 'middle':
                    Residuals_age = Residuals_sex[(Residuals_sex['Age'] - Residuals_sex['Age'].median()).abs() < 5]
                else:
                    Residuals_age = Residuals_sex[Residuals_sex['Age'] >= Residuals_sex['Age'].max() - 10]
                Residuals_age['age_category'] = age_category
                
                # Aging rate
                for aging_rate in ['accelerated', 'normal', 'decelerated']:
                    print('Aging rate: ' + aging_rate)
                    Residuals_ar = Residuals_age
                    if aging_rate == 'accelerated':
                        Residuals_ar.sort_values(by='res', ascending=True, inplace=True)
                    elif aging_rate == 'decelerated':
                        Residuals_ar.sort_values(by='res', ascending=False, inplace=True)
                    else:
                        Residuals_ar.sort_values(by='res_abs', ascending=True, inplace=True)
                    Residuals_ar['aging_rate'] = aging_rate
                    Residuals_ar = Residuals_ar.iloc[:self.N_samples_attentionmaps, ]
                    Residuals_ar['sample'] = range(len(Residuals_ar.index))
                    if df_to_plot is None:
                        df_to_plot = Residuals_ar
                    else:
                        df_to_plot = df_to_plot.append(Residuals_ar)
        
        # Postprocessing
        df_to_plot['Biological_Age'] = df_to_plot['Age'] - df_to_plot['res']
        activations_path = '../figures/Attention_Maps/' + self.target + '/' + self.organ + '/' + self.view + '/' + \
                           self.transformation + '/' + df_to_plot['sex'] + '/' + df_to_plot['age_category'] + '/' + \
                           df_to_plot['aging_rate']
        file_names = '/imagetypeplaceholder_' + self.target + '_' + self.organ + '_' + self.view + '_' + \
                     self.transformation + '_' + df_to_plot['sex'] + '_' + df_to_plot['age_category'] + '_' + \
                     df_to_plot['aging_rate'] + '_' + df_to_plot['sample'].astype(str)
        if self.leftright:
            activations_path += '/sideplaceholder'
            file_names += '_sideplaceholder'
        df_to_plot['save_title'] = activations_path + file_names
        path_save = self.path_data + 'AttentionMaps-samples_' + self.target + '_' + self.organ + '_' + self.view + \
                    '_' + self.transformation + '.csv'
        df_to_plot.to_csv(path_save, index=False)
        self.df_to_plot = df_to_plot
    
    def preprocessing(self):
        self._select_best_model()
        self._format_residuals()
        self._select_representative_samples()
    
    def _preprocess_for_outer_fold(self, outer_fold):
        self.df_outer_fold = self.df_to_plot[self.df_to_plot['outer_fold'] == outer_fold]
        self.n_images = len(self.df_outer_fold.index)
        if self.leftright:
            self.n_images *= 2
        
        # Generate the data generator(s)
        self.n_images_batch = self.n_images // self.batch_size * self.batch_size
        self.n_samples_batch = self.n_images_batch // 2 if self.leftright else self.n_images_batch
        self.df_batch = self.df_outer_fold.iloc[:self.n_samples_batch, :]
        if self.n_images_batch > 0:
            self.generator_batch = \
                MyImageDataGenerator(target=self.target, organ=self.organ, view=self.view,
                                     data_features=self.df_batch, n_samples_per_subepoch=None,
                                     batch_size=self.batch_size, training_mode=False,
                                     side_predictors=self.side_predictors, dir_images=self.dir_images,
                                     images_width=self.image_width, images_height=self.image_height,
                                     data_augmentation=False, data_augmentation_factor=None, seed=self.seed)
        else:
            self.generator_batch = None
        self.n_samples_leftovers = self.n_images % self.batch_size
        self.df_leftovers = self.df_outer_fold.iloc[self.n_samples_batch:, :]
        if self.n_samples_leftovers > 0:
            self.generator_leftovers = \
                MyImageDataGenerator(target=self.target, organ=self.organ, view=self.view,
                                     data_features=self.df_leftovers, n_samples_per_subepoch=None,
                                     batch_size=self.n_samples_leftovers, training_mode=False,
                                     side_predictors=self.side_predictors, dir_images=self.dir_images,
                                     images_width=self.image_width, images_height=self.image_height,
                                     data_augmentation=False, data_augmentation_factor=None, seed=self.seed)
        else:
            self.generator_leftovers = None
        
        # load the weights for the fold (for test images in fold i, load the corresponding model: (i-1)%N_CV_folds
        outer_fold_model = str((int(outer_fold) - 1) % self.n_CV_outer_folds)
        self.model.load_weights(self.path_data + 'model-weights_' + self.version + '_' + outer_fold_model + '.h5')
    
    @staticmethod
    def _process_saliency(saliency):
        saliency *= 255 / np.max(np.abs(saliency))
        saliency = saliency.astype(int)
        r_ch = saliency.copy()
        r_ch[r_ch < 0] = 0
        b_ch = -saliency.copy()
        b_ch[b_ch < 0] = 0
        g_ch = saliency.copy() * 0
        a_ch = np.maximum(b_ch, r_ch)
        saliency = np.dstack((r_ch, g_ch, b_ch, a_ch))
        return saliency
    
    @staticmethod
    def _process_gradcam(gradcam):
        # rescale to 0-255
        gradcam = np.maximum(gradcam, 0) / np.max(gradcam)
        gradcam = np.uint8(255 * gradcam)
        # Convert to rgb
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_gradcam = jet_colors[gradcam]
        jet_gradcam = array_to_img(jet_gradcam)
        jet_gradcam = jet_gradcam.resize((gradcam.shape[1], gradcam.shape[0]))
        jet_gradcam = img_to_array(jet_gradcam)
        return jet_gradcam
    
    def _generate_maps_for_one_batch(self, df, Xs, y):
        # Generate saliency
        saliencies = get_gradients_of_activations(self.model, Xs, y, layer_name='input_1')['input_1'].sum(axis=3)
        # Generate gradam
        weights = get_gradients_of_activations(self.model, Xs, y, layer_name=self.last_conv_layer,
                                               )[self.last_conv_layer]
        weights = weights.mean(axis=(1, 2))
        weights /= np.abs(weights.max()) + 1e-7  # for numerical stability
        activations = get_activations(self.model, Xs, layer_name=self.last_conv_layer)[self.last_conv_layer]
        # We must take the absolute value because for Grad-RAM, unlike for Grad-Cam, we care both about + and - effects
        gradcams = np.abs(np.einsum('il,ijkl->ijk', weights, activations))
        zoom_factor = [1] + list(np.array(Xs[0].shape[1:3]) / np.array(gradcams.shape[1:]))
        gradcams = zoom(gradcams, zoom_factor)
        
        # Save single images and filters
        for j in range(len(y)):
            # select sample
            if self.leftright:
                idx = j // 2
                side = 'right' if j % 2 == 0 else 'left'
            else:
                idx = j
                side = None
            path = df['save_title'].values[idx]
            ID = df['id'].values[idx]
            # create directory tree if necessary
            if self.leftright:
                path = path.replace('sideplaceholder', side)
            path_dir = '/'.join(path.split('/')[:-1])
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            # Save raw image
            # Compute path to test if images existed in first place
            path_image = '../images/' + self.organ + '/' + self.view + '/' + self.transformation + '/'
            if self.leftright:
                path_image += side + '/'
            path_image += ID + '.jpg'
            if not os.path.exists(path_image):
                print('No image found at ' + path_image + ', skipping.')
                continue
            img = load_img(path_image, target_size=(saliencies.shape[1], saliencies.shape[2]))
            img.save(path.replace('imagetypeplaceholder', 'RawImage') + '.jpg')
            # Save saliency
            saliency = saliencies[j, :, :]
            saliency = self._process_saliency(saliency)
            np.save(path.replace('imagetypeplaceholder', 'Saliency') + '.npy', saliency)
            # Save gradcam
            gradcam = gradcams[j, :, :]
            gradcam = self._process_gradcam(gradcam)
            np.save(path.replace('imagetypeplaceholder', 'Gradcam') + '.npy', gradcam)
    
    def generate_filters(self):
        if self.organ + '_' + self.view + '_' + self.transformation in self.organs_views_transformations_images:
            self._generate_architecture()
            self.model.compile(optimizer=self.optimizers[self.optimizer](lr=self.learning_rate, clipnorm=1.0),
                               loss=self.loss_function, metrics=self.metrics)
            self.last_conv_layer = self.dict_architecture_to_last_conv_layer_name[self.parameters['architecture']]
            for outer_fold in self.outer_folds:
                print('Generate attention maps for outer_fold ' + outer_fold)
                gc.collect()
                self._preprocess_for_outer_fold(outer_fold)
                n_samples_per_batch = self.batch_size // 2 if self.leftright else self.batch_size
                for i in range(self.n_images // self.batch_size):
                    print('Generating maps for batch ' + str(i))
                    Xs, y = self.generator_batch.__getitem__(i)
                    df = self.df_batch.iloc[n_samples_per_batch * i: n_samples_per_batch * (i + 1), :]
                    self._generate_maps_for_one_batch(df, Xs, y)
                if self.n_samples_leftovers > 0:
                    print('Generating maps for leftovers')
                    Xs, y = self.generator_leftovers.__getitem__(0)
                    self._generate_maps_for_one_batch(self.df_leftovers, Xs, y)


class GWASPreprocessing(Basics):
    
    def __init__(self, target=None):
        Basics.__init__(self)
        self.target = target
        self.fam = None
        self.Residuals = None
        self.covars = None
        self.data = None
        self.list_organs = None
        self.IIDs_organs = {}
        self.IIDs_organ_pairs = {}
    
    def _generate_fam_file(self):
        fam = pd.read_csv('/n/groups/patel/uk_biobank/project_52887_genetics/ukb52887_cal_chr1_v2_s488264.fam',
                          header=None, sep=' ')
        fam.columns = ['FID', 'IID', 'father', 'mother', 'Sex', 'phenotype']
        fam['phenotype'] = 1
        fam.to_csv(self.path_data + 'GWAS.fam', index=False, header=False, sep=' ')
        fam.to_csv(self.path_data + 'GWAS_exhaustive_placeholder.tab', index=False, sep='\t')
        self.fam = fam
    
    def _preprocess_residuals(self):
        # Load residuals
        Residuals = pd.read_csv(self.path_data + 'RESIDUALS_bestmodels_eids_' + self.target + '_test.csv')
        Residuals['id'] = Residuals['eid']
        Residuals.rename(columns={'id': 'FID', 'eid': 'IID'}, inplace=True)
        Residuals = Residuals[Residuals['Ethnicity.White'] == 1]
        cols_to_drop = ['instance', 'outer_fold', 'Sex'] + \
                       [col for col in Residuals.columns.values if 'Ethnicity.' in col]
        Residuals.drop(columns=cols_to_drop, inplace=True)
        self.Residuals = Residuals
        self.list_organs = [col for col in self.Residuals.columns.values if col not in ['FID', 'IID', 'Age']]
    
    def _preprocess_covars(self):
        # Load covars
        covar_cols = ['eid', '22001-0.0', '21000-0.0', '54-0.0', '22000-0.0'] + ['22009-0.' + str(i) for i in
                                                                                 range(1, 41)]
        covars = pd.read_csv('/n/groups/patel/uk_biobank/project_52887_41230/ukb41230.csv', usecols=covar_cols)
        dict_rename = {'eid': 'IID', '22001-0.0': 'Sex', '21000-0.0': 'Ethnicity', '54-0.0': 'Assessment_center',
                       '22000-0.0': 'Genotyping_batch'}
        for i in range(1, 41):
            dict_rename.update(dict.fromkeys(['22009-0.' + str(i)], 'PC' + str(i)))
        covars.rename(columns=dict_rename, inplace=True)
        covars.dropna(inplace=True)
        covars['Sex'][covars['Sex'] == 0] = 2
        covars['Sex'] = covars['Sex'].astype(int)
        # remove non whites samples as suggested in BOLT-LMM_v2.3.4_manual.pdf p18
        covars = covars[covars['Ethnicity'].isin([1, 1001, 1002, 1003])]
        self.covars = covars
    
    def _merge_main_data(self):
        # Merge both dataframes
        self.data = self.covars.merge(self.Residuals, on=['IID'])
        reordered_cols = ['FID', 'IID', 'Assessment_center', 'Genotyping_batch', 'Age', 'Sex', 'Ethnicity'] + \
                         ['PC' + str(i) for i in range(1, 41)] + self.list_organs
        self.data = self.data[reordered_cols]
        print('Preparing data for heritabilities')
        for organ in self.list_organs:
            print('Preparing data for ' + organ)
            data_organ = self.data.copy()
            cols_to_drop = [organ2 for organ2 in self.list_organs if organ2 != organ]
            data_organ.drop(columns=cols_to_drop, inplace=True)
            data_organ.dropna(inplace=True)
            data_organ.to_csv(self.path_data + 'GWAS_data_' + self.target + '_' + organ + '.tab', index=False,
                              sep='\t')
            self.IIDs_organs[organ] = data_organ['IID'].values
    
    def _preprocessing_genetic_correlations(self):
        print('Preparing data for genetic correlations')
        organs_pairs = pd.DataFrame(columns=['organ1', 'organ2'])
        for counter, organ1 in enumerate(self.list_organs):
            for organ2 in self.list_organs[(counter + 1):]:
                print('Preparing data for the organ pair ' + organ1 + ' and ' + organ2)
                # Generate GWAS dataframe
                organs_pairs = organs_pairs.append({'organ1': organ1, 'organ2': organ2}, ignore_index=True)
                data_organ_pair = self.data.copy()
                cols_to_drop = [organ3 for organ3 in self.list_organs if organ3 not in [organ1, organ2]]
                data_organ_pair.drop(columns=cols_to_drop, inplace=True)
                data_organ_pair.dropna(inplace=True)
                data_organ_pair.to_csv(self.path_data + 'GWAS_data_' + self.target + '_' + organ1 + '_' + organ2 +
                                       '.tab', index=False, sep='\t')
                self.IIDs_organ_pairs[organ1 + '_' + organ2] = data_organ_pair['IID'].values
                
        organs_pairs.to_csv(self.path_data + 'GWAS_genetic_correlations_pairs_' + self.target + '.csv', header=False,
                            index=False)
    
    def _list_removed(self):
        # samples to remove for each organ
        print('Listing samples to remove for each organ')
        for organ in self.list_organs:
            print('Preparing samples to remove for organ ' + organ)
            remove_organ = self.fam[['FID', 'IID']].copy()
            remove_organ = remove_organ[-remove_organ['IID'].isin(self.IIDs_organs[organ])]
            remove_organ.to_csv(self.path_data + 'GWAS_remove_' + self.target + '_' + organ + '.tab', index=False,
                                header=False, sep=' ')
        
        # samples to remove for each organ pair
        print('Listing samples to remove for each organ pair')
        for counter, organ1 in enumerate(self.list_organs):
            for organ2 in self.list_organs[(counter + 1):]:
                print('Preparing samples to remove for organ pair ' + organ1 + ' and ' + organ2)
                remove_organ_pair = self.fam[['FID', 'IID']].copy()
                remove_organ_pair = \
                    remove_organ_pair[-remove_organ_pair['IID'].isin(self.IIDs_organ_pairs[organ1 + '_' + organ2])]
                remove_organ_pair.to_csv(self.path_data + 'GWAS_remove_' + self.target + '_' + organ1 + '_' + organ2 +
                                         '.tab', index=False, header=False, sep=' ')
    
    def compute_gwas_inputs(self):
        self._generate_fam_file()
        self._preprocess_residuals()
        self._preprocess_covars()
        self._merge_main_data()
        self._preprocessing_genetic_correlations()
        self._list_removed()


class GWASPostprocessing(Basics):
    
    def __init__(self, target=None):
        Basics.__init__(self)
        self.target = target
        self.organ = None
        self.GWAS = None
        self.FDR_correction = 5e-8
    
    def _processing(self):
        self.GWAS = pd.read_csv(self.path_data + 'GWAS_' + self.target + '_' + self.organ + '_X.stats', sep='\t')
        GWAS_autosome = pd.read_csv(self.path_data + 'GWAS_' + self.target + '_' + self.organ + '_autosome.stats',
                                    sep='\t')
        self.GWAS[self.GWAS['CHR'] != 23] = GWAS_autosome
        self.GWAS_hits = self.GWAS[self.GWAS['P_BOLT_LMM_INF'] < self.FDR_correction]
    
    def _save_data(self):
        self.GWAS.to_csv(self.path_data + 'GWAS_' + self.target + '_' + self.organ + '.csv', index=False)
        self.GWAS_hits.to_csv(self.path_data + 'GWAS_hits_' + self.target + '_' + self.organ + '.csv', index=False)
    
    def _merge_all_hits(self):
        print('Merging all the GWAS results into a model called All...')
        # Summarize all the significant SNPs
        files = [file for file in glob.glob(self.path_data + 'GWAS_hits*')
                 if ('All' not in file) & ('_withGenes' not in file)]
        All_hits = None
        print(files)
        for file in files:
            print(file)
            hits_organ = pd.read_csv(file)[
                ['SNP', 'CHR', 'BP', 'GENPOS', 'ALLELE1', 'ALLELE0', 'A1FREQ', 'F_MISS', 'CHISQ_LINREG',
                 'P_LINREG', 'BETA', 'SE', 'CHISQ_BOLT_LMM_INF', 'P_BOLT_LMM_INF']]
            hits_organ['organ'] = '.'.join(file.split('_')[-1].split('.')[:-1])
            if All_hits is None:
                All_hits = hits_organ
            else:
                All_hits = pd.concat([All_hits, hits_organ])
        All_hits.sort_values(by=['CHR', 'BP'], inplace=True)
        All_hits.to_csv(self.path_data + 'GWAS_hits_' + self.target + '_All.csv', index=False)
    
    def processing_all_organs(self):
        if not os.path.exists('../figures/GWAS/'):
            os.makedirs('../figures/GWAS/')
        for organ in self.organs_XWAS:
            if os.path.exists(self.path_data + 'GWAS_' + self.target + '_' + organ + '_X.stats') & \
                    os.path.exists(self.path_data + 'GWAS_' + self.target + '_' + organ + '_autosome.stats'):
                print('Processing data for organ ' + organ)
                self.organ = organ
                self._processing()
                self._save_data()
        self._merge_all_hits()
        
    @staticmethod
    def _grep(pattern, path):
        for line in open(path, 'r'):
            if line.find(pattern) > -1:
                return True
        return False
    
    @staticmethod
    def _melt_correlation_matrix(Correlations, models):
        Corrs = Correlations.loc[models, models]
        Corrs = Corrs.where(np.triu(np.ones(Corrs.shape), 1).astype(np.bool))
        Corrs = Corrs.stack().reset_index()
        Corrs.columns = ['Row', 'Column', 'Correlation']
        return Corrs
    
    @staticmethod
    def _compute_stats(data, title):
        m = data['Correlation'].mean()
        s = data['Correlation'].std()
        n = len(data.index)
        print('Correlation between ' + title + ': ' + str(round(m, 3)) + '+-' + str(round(s, 3)) +
              ', n_pairs=' + str(n))
    
    def parse_heritability_scores(self):
        # Generate empty dataframe
        Heritabilities = np.empty((len(self.organs_XWAS), 3,))
        Heritabilities.fill(np.nan)
        Heritabilities = pd.DataFrame(Heritabilities)
        Heritabilities.index = self.organs_XWAS
        Heritabilities.columns = ['Organ', 'h2', 'h2_sd']
        # Fill the dataframe
        for organ in self.organs_XWAS:
            path = '../eo/MI09C_reml_' + self.target + '_' + organ + '_X.out'
            if os.path.exists(path) and self._grep("h2g", path):
                for line in open('../eo/MI09C_reml_' + self.target + '_' + organ + '_X.out', 'r'):
                    if line.find('h2g (1,1): ') > -1:
                        h2 = float(line.split()[2])
                        h2_sd = float(line.split()[-1][1:-2])
                        Heritabilities.loc[organ, :] = [organ, h2, h2_sd]
        # Print and save results
        print('Heritabilities:')
        print(Heritabilities)
        Heritabilities.to_csv(self.path_data + 'GWAS_heritabilities_' + self.target + '.csv', index=False)
    
    def parse_genetic_correlations(self):
        # Generate empty dataframe
        Genetic_correlations = np.empty((len(self.organs_XWAS), len(self.organs_XWAS),))
        Genetic_correlations.fill(np.nan)
        Genetic_correlations = pd.DataFrame(Genetic_correlations)
        Genetic_correlations.index = self.organs_XWAS
        Genetic_correlations.columns = self.organs_XWAS
        Genetic_correlations_sd = Genetic_correlations.copy()
        Genetic_correlations_str = Genetic_correlations.copy()
        # Fill the dataframe
        for counter, organ1 in enumerate(self.organs_XWAS):
            for organ2 in self.organs_XWAS[(counter + 1):]:
                if os.path.exists('../eo/MI09D_' + self.target + '_' + organ1 + '_' + organ2 + '.out'):
                    for line in open('../eo/MI09D_' + self.target + '_' + organ1 + '_' + organ2 + '.out', 'r'):
                        if line.find('gen corr (1,2):') > -1:
                            corr = float(line.split()[3])
                            corr_sd = float(line.split()[-1][1:-2])
                            corr_str = "{:.3f}".format(corr) + '+-' + "{:.3f}".format(corr_sd)
                            Genetic_correlations.loc[organ1, organ2] = corr
                            Genetic_correlations.loc[organ2, organ1] = corr
                            Genetic_correlations_sd.loc[organ1, organ2] = corr_sd
                            Genetic_correlations_sd.loc[organ2, organ1] = corr_sd
                            Genetic_correlations_str.loc[organ1, organ2] = corr_str
                            Genetic_correlations_str.loc[organ2, organ1] = corr_str
        # Print and save the results
        print('Genetic correlations:')
        print(Genetic_correlations)
        Genetic_correlations.to_csv(self.path_data + 'GWAS_correlations_' + self.target + '.csv')
        Genetic_correlations_sd.to_csv(self.path_data + 'GWAS_correlations_sd_' + self.target + '.csv')
        Genetic_correlations_str.to_csv(self.path_data + 'GWAS_correlations_str_' + self.target + '.csv')
        
        # Save sample size for the GWAS correlations
        Correlations_sample_sizes = Genetic_correlations.copy()
        Correlations_sample_sizes = Correlations_sample_sizes * np.NaN
        dimensions = Correlations_sample_sizes.columns.values
        for i1, dim1 in enumerate(dimensions):
            for i2, dim2 in enumerate(dimensions[i1:]):
                # Find the sample size
                path = '../data/GWAS_data_Age_' + dim1 + '_' + dim2 + '.tab'
                if os.path.exists(path):
                    ss = len(pd.read_csv(path, sep='\t').index)
                    Correlations_sample_sizes.loc[dim1, dim2] = ss
                    Correlations_sample_sizes.loc[dim2, dim1] = ss
        Correlations_sample_sizes.to_csv(self.path_data + 'GWAS_correlations_sample_sizes_' + self.target + '.csv')
        
        # Print correlations between main dimensions
        main_dims = ['Abdomen', 'Musculoskeletal', 'Lungs', 'Eyes', 'Heart', 'Arterial', 'Brain', 'Biochemistry',
                     'Hearing', 'ImmuneSystem', 'PhysicalActivity']
        Corrs_main = self._melt_correlation_matrix(Correlations, main_dims)
        Corrs_main_sd = self._melt_correlation_matrix(Correlations_sd, main_dims)
        Corrs_main['Correlation_sd'] = Corrs_main_sd['Correlation']
        Corrs_main['Correlation_str'] = Corrs_main['Correlation'] + '+-' + Corrs_main['Correlation_sd']
        # Fill the table with sample sizes
        sample_sizes = []
        to_remove_ss = []
        for i, row in Corrs_main.iterrows():
            # Fill the sample size
            sample_size = Correlations_sample_sizes.loc[row['Row'], row['Column']]
            if sample_size <= 15000:
                to_remove_ss.append(i)
            sample_sizes.append(sample_size)
        Corrs_main['Sample_size'] = sample_sizes
        self._compute_stats(Corrs_main, 'all pairs')
        self._compute_stats(Corrs_main.drop(index=to_remove_ss), 'after filtering sample sizes <= 15000')
        
        # Print correlations between subdimensions
        pairs_all = \
            [['BrainMRI', 'BrainCognitive'], ['EyesOCT', 'EyesFundus'], ['HeartECG', 'HeartMRI'],
             ['AbdomenLiver', 'AbdomenPancreas'], ['BiochemistryBlood', 'BiochemistryUrine'],
             ['MusculoskeletalScalars', 'MusculoskeletalFullBody'], ['MusculoskeletalScalars', 'MusculoskeletalSpine'],
             ['MusculoskeletalScalars', 'MusculoskeletalHips'], ['MusculoskeletalScalars', 'MusculoskeletalKnees'],
             ['MusculoskeletalFullBody', 'MusculoskeletalSpine'], ['MusculoskeletalFullBody', 'MusculoskeletalHips'],
             ['MusculoskeletalFullBody', 'MusculoskeletalKnees'], ['MusculoskeletalSpine', 'MusculoskeletalHips'],
             ['MusculoskeletalSpine', 'MusculoskeletalKnees'], ['MusculoskeletalHips', 'MusculoskeletalKnees']]
        pairs_musculo = \
            [['MusculoskeletalScalars', 'MusculoskeletalFullBody'], ['MusculoskeletalScalars', 'MusculoskeletalSpine'],
             ['MusculoskeletalScalars', 'MusculoskeletalHips'], ['MusculoskeletalScalars', 'MusculoskeletalKnees']]
        pairs_musculo_images = \
            [['MusculoskeletalFullBody', 'MusculoskeletalSpine'], ['MusculoskeletalFullBody', 'MusculoskeletalHips'],
             ['MusculoskeletalFullBody', 'MusculoskeletalKnees'], ['MusculoskeletalSpine', 'MusculoskeletalHips'],
             ['MusculoskeletalSpine', 'MusculoskeletalKnees'], ['MusculoskeletalHips', 'MusculoskeletalKnees']]
        PAIRS = {'all subdimensions': pairs_all, 'musculo scalars vs others': pairs_musculo,
                 'musculo-no scalars': pairs_musculo_images}
        for _, (key, pairs) in enumerate(PAIRS.items()):
            print(key)
            cors_pairs = []
            for pair in pairs:
                cor = Correlations.loc[pair[0], pair[1]]
                cor_sd = Correlations_sd.loc[pair[0], pair[1]]
                ss = Correlations_sample_sizes.loc[pair[0], pair[1]]
                cors_pairs.append(cor)
                print('Correlation between ' + pair[0] + ' and ' + pair[1] + ' = ' + str(round(cor, 3)) + '+-' +
                      str(round(cor_sd, 3)) + '; sample size = ' + str(ss))
            print('Mean correlation for ' + key + ' = ' + str(round(np.mean(cors_pairs), 3)) + '+-' +
                  str(round(np.std(cors_pairs), 3)) + ', number of pairs = ' + str(len(pairs)))
    
    @staticmethod
    def compare_phenotypic_correlation_with_genetic_correlations():
        Phenotypic_correlations = pd.read_csv('../data/ResidualsCorrelations_bestmodels_eids_Age_test.csv', index_col=0)
        Phenotypic_correlations_sd = pd.read_csv('../data/ResidualsCorrelations_bestmodels_sd_eids_Age_test.csv',
                                                 index_col=0)
        Genetic_correlations = pd.read_csv('../data/GWAS_correlations_Age.csv', index_col=0)
        Genetic_correlations_sd = pd.read_csv('../data/GWAS_correlations_sd_Age.csv', index_col=0)
        Correlations_sample_sizes = pd.read_csv('../data/GWAS_correlations_sample_sizes_Age.csv', index_col=0)
        Genetic_correlations_filtered = Genetic_correlations.where(Correlations_sample_sizes > 15000)
        Phenotypic_correlations_filtered = Phenotypic_correlations.where(~Genetic_correlations_filtered.isna())
        
        dict_dims_layers = {
            'all_dims': Phenotypic_correlations_filtered.columns.values,
            'main_dims': ['Brain', 'Eyes', 'Hearing', 'Lungs', 'Arterial', 'Heart', 'Abdomen', 'Musculoskeletal',
                          'PhysicalActivity', 'Biochemistry', 'ImmuneSystem'],
            'sub_dims': ['BrainCognitive', 'BrainMRI', 'EyesFundus', 'EyesOCT', 'ArterialPulseWaveAnalysis',
                         'ArterialCarotids',
                         'HeartECG', 'HeartMRI', 'AbdomenLiver', 'AbdomenPancreas', 'MusculoskeletalSpine',
                         'MusculoskeletalHips', 'MusculoskeletalKnees', 'MusculoskeletalFullBody',
                         'MusculoskeletalScalars',
                         'BiochemistryUrine', 'BiochemistryBlood']
        }

        def _print_comparisons_between_pheno_and_geno(dimensions):
            Pheno_dims = Phenotypic_correlations_filtered.loc[dimensions, dimensions]
            Geno_dims = Genetic_correlations_filtered.loc[dimensions, dimensions]
            Pheno_dims = Pheno_dims.where(np.triu(np.ones(Pheno_dims.shape), 1).astype(np.bool))
            Pheno_dims = Pheno_dims.stack().reset_index()
            Geno_dims = Geno_dims.where(np.triu(np.ones(Geno_dims.shape), 1).astype(np.bool))
            Geno_dims = Geno_dims.stack().reset_index()
            Pheno_dims.columns = ['Row', 'Column', 'Correlation']
            Geno_dims.columns = ['Row', 'Column', 'Correlation']
            Correlations_dims = Pheno_dims.copy()
            Correlations_dims = Correlations_dims.rename(columns={'Correlation': 'Phenotypic'})
            Correlations_dims['Genetic'] = Geno_dims['Correlation']
            final_cor = str(round(Correlations_dims[['Phenotypic', 'Genetic']].corr().iloc[0, 1], 3))
            # Find min and max difference:
            Correlations_dims['Difference'] = Correlations_dims['Phenotypic'] - Correlations_dims['Genetic']
            # min
            min_diff = Correlations_dims.iloc[Correlations_dims['Difference'].idxmin(), :]
            min_pheno_sd = Phenotypic_correlations_sd.loc[min_diff['Row'], min_diff['Column']]
            min_genetic_sd = Genetic_correlations_sd.loc[min_diff['Row'], min_diff['Column']]
            min_diff_sd = np.sqrt(min_pheno_sd ** 2 + min_genetic_sd ** 2)
            # max
            max_diff = Correlations_dims.iloc[Correlations_dims['Difference'].idxmax(), :]
            max_pheno_sd = Phenotypic_correlations_sd.loc[max_diff['Row'], max_diff['Column']]
            max_genetic_sd = Genetic_correlations_sd.loc[max_diff['Row'], max_diff['Column']]
            max_diff_sd = np.sqrt(max_pheno_sd ** 2 + max_genetic_sd ** 2)
            # print key results
            print('The correlation between phenotypic and genetic correlations is: ' + final_cor)
            print('The min difference between phenotypic and genetic correlations is between ' + min_diff[
                'Row'] + ' and ' +
                  min_diff['Column'] + '. Difference = ' + str(round(min_diff['Difference'], 3)) + '+-' +
                  str(round(min_diff_sd, 3)) + '; Phenotypic correlation = ' + str(
                round(min_diff['Phenotypic'], 3)) + '+-' +
                  str(round(min_pheno_sd, 3)) + '; Genetic correlation = ' + str(round(min_diff['Genetic'], 3)) + '+-' +
                  str(round(min_genetic_sd, 3)))
            print('The max difference between phenotypic and genetic correlations is between ' + max_diff[
                'Row'] + ' and ' +
                  max_diff['Column'] + '. Difference = ' + str(round(max_diff['Difference'], 3)) + '+-' +
                  str(round(max_diff_sd, 3)) + '; Phenotypic correlation = ' + str(
                round(max_diff['Phenotypic'], 3)) + '+-' +
                  str(round(max_pheno_sd, 3)) + '; Genetic correlation = ' + str(round(max_diff['Genetic'], 3)) + '+-' +
                  str(round(max_genetic_sd, 3)))
    
        for _, (dims_name, dims) in enumerate(dict_dims_layers.items()):
            print('Printing the comparison between the phenotypic and genetic correlations at the following level: ' +
                  dims_name)
            _print_comparisons_between_pheno_and_geno(dims)


class GWASAnnotate(Basics):
    
    def __init__(self, target=None):
        Basics.__init__(self)
        self.target = target
        self.All_hits = None
        self.All_hits_missing = None
    
    def download_data(self):
        os.chdir('/Users/Alan/Desktop/Aging/Medical_Images/bash_local/')
        os.system('scp al311@transfer.rc.hms.harvard.edu:/n/groups/patel/Alan/Aging/Medical_Images/data/' +
                  self.path_data + 'GWAS_hits_' + self.target + '_All.csv' + ' ../data/')
        self.All_hits = pd.read_csv(self.path_data + 'GWAS_hits_' + self.target + '_All.csv')
    
    @staticmethod
    def _find_nearest_gene(row, key):
        if row['Overlapped Gene'] != 'None':
            gene = row['Overlapped Gene']
            gene_type = row['Type']
        elif row['Distance to Nearest Downstream Gene'] <= row['Distance to Nearest Upstream Gene']:
            gene = row['Nearest Downstream Gene']
            gene_type = row['Type of Nearest Downstream Gene']
        else:
            gene = row['Nearest Upstream Gene']
            gene_type = row['Type of Nearest Upstream Gene']
        to_return = pd.Series([row[key], gene, gene_type])
        to_return.index = [key, 'Gene', 'Gene_type']
        return to_return
    
    @staticmethod
    def _concatenate_genes(group, key):
        row = group.drop_duplicates(subset=[key])
        unique_genes_rows = group.drop_duplicates(subset=[key, 'Gene'])
        row['Gene'] = ', '.join(list(unique_genes_rows['Gene']))
        row['Gene_type'] = ', '.join(list(unique_genes_rows['Gene_type']))
        return row
    
    def preprocessing_rs(self):
        # Generate the list of SNPs to annotate in two formats to input into https://www.snp-nexus.org/v4/
        # Format 1: based on rs#
        snps_rs = pd.Series(self.All_hits['SNP'].unique())
        snps_rs.index = ['dbsnp'] * len(snps_rs.index)
        snps_rs.to_csv(self.path_data + 'snps_rs.txt', sep='\t', header=False)
    
    def postprocessing_rs(self):
        # Load the output fromsnp-nexus and fill the available rows
        genes_rs = pd.read_csv(self.path_data + 'GWAS_genes_rs.txt', sep='\t')
        # Find the nearest gene
        genes_rs = genes_rs.apply(self._find_nearest_gene, args=(['Variation ID']), axis=1)
        # Concatenate the findinds when several genes matched
        genes_rs = genes_rs.groupby(by='Variation ID').apply(self._concatenate_genes, 'Variation ID')
        # Fill the rows from the main dataframe when possible
        genes_rs.set_index('Variation ID', inplace=True)
        self.All_hits['Gene'] = np.NaN
        self.All_hits['Gene_type'] = np.NaN
        self.All_hits.set_index('SNP', drop=False, inplace=True)
        self.All_hits.loc[genes_rs.index, ['Gene', 'Gene_type']] = genes_rs
    
    def preprocessing_chrbp(self):
        # Format 2: based on CHR and BP
        snps_chrbp = self.All_hits.loc[self.All_hits['Gene'].isna(), ['CHR', 'BP', 'ALLELE0', 'ALLELE1']]
        snps_chrbp['strand'] = 1
        snps_chrbp.index = ['chromosome'] * len(snps_chrbp.index)
        snps_chrbp.drop_duplicates(inplace=True)
        snps_chrbp.to_csv(self.path_data + 'snps_chrbp.txt', sep='\t', header=False)
    
    def postprocessing_chrbp(self):
        # Load the output from snp-nexus and fill the available rows
        genes_chrbp = pd.read_csv(self.path_data + 'GWAS_genes_chrbp.txt', sep='\t')
        genes_chrbp['chrbp'] = genes_chrbp['Chromosome'].astype(str) + ':' + genes_chrbp['Position'].astype(str)
        # Find the nearest gene
        genes_chrbp = genes_chrbp.apply(self._find_nearest_gene, args=(['chrbp']), axis=1)
        # Concatenate the findinds when several genes matched
        genes_chrbp = genes_chrbp.groupby(by='chrbp').apply(self._concatenate_genes, 'chrbp')
        # Fill the rows from the main dataframe when possible
        genes_chrbp.set_index('chrbp', inplace=True)
        self.All_hits['chrbp'] = 'chr' + self.All_hits['CHR'].astype(str) + ':' + self.All_hits['BP'].astype(str)
        self.All_hits.set_index('chrbp', drop=False, inplace=True)
        # Only keep subset of genes that actually are hits (somehow extra SNPs are returned too
        genes_chrbp = genes_chrbp[genes_chrbp.index.isin(self.All_hits.index.values)]
        self.All_hits.loc[genes_chrbp.index, ['Gene', 'Gene_type']] = genes_chrbp
    
    def preprocessing_missing(self):
        # Identify which SNPs were not matched so far, and use zoom locus to fill the gaps
        self.All_hits_missing = self.All_hits[self.All_hits['Gene'].isna()]
        print(str(len(self.All_hits_missing.drop_duplicates(subset=['SNP']).index)) + ' missing SNPs out of ' +
              str(len(self.All_hits.drop_duplicates(subset=['SNP']).index)) + '.')
        self.All_hits_missing.to_csv(self.path_data + 'All_hits_missing.csv', index=False, sep='\t')
    
    def postprocessing_missing(self):
        # The gene_type column was filled using https://www.genecards.org/
        self.All_hits.loc['chr1:3691997', ['Gene', 'Gene_type']] = ['SMIM1', 'protein_coding']
        self.All_hits.loc['chr2:24194313', ['Gene', 'Gene_type']] = ['COL4A4', 'protein_coding']
        self.All_hits.loc['chr2:227896885', ['Gene', 'Gene_type']] = ['UBXN2A', 'protein_coding']
        self.All_hits.loc['chr2:27656822', ['Gene', 'Gene_type']] = ['NRBP1', 'protein_coding']
        self.All_hits.loc['chr2:42396721', ['Gene', 'Gene_type']] = ['AC083949.1, EML4', 'rna_gene, protein_coding']
        self.All_hits.loc['chr2:71661855', ['Gene', 'Gene_type']] = ['ZNF638', 'protein_coding']
        self.All_hits.loc['chr3:141081497', ['Gene', 'Gene_type']] = ['PXYLP1, AC117383.1, ZBTB38',
                                                                        'protein_coding, rna_gene, protein_coding']
        self.All_hits.loc['chr4:106317506', ['Gene', 'Gene_type']] = ['PPA2', 'protein_coding']
        self.All_hits.loc['chr5:156966773', ['Gene', 'Gene_type']] = ['ADAM19', 'protein_coding']
        self.All_hits.loc['chr6:29797695', ['Gene', 'Gene_type']] = ['HLA-G', 'protein_coding']
        self.All_hits.loc['chr6:31106501', ['Gene', 'Gene_type']] = ['PSORS1C1, PSORS1C2',
                                                                     'protein_coding, protein_coding']
        self.All_hits.loc['chr6:31322216', ['Gene', 'Gene_type']] = ['HLA-B', 'protein_coding']
        self.All_hits.loc['chr6:32552146', ['Gene', 'Gene_type']] = ['HLA-DRB1', 'protein_coding']
        self.All_hits.loc['chr6:33377481', ['Gene', 'Gene_type']] = ['KIFC1', 'protein_coding']
        self.All_hits.loc['chr8:9683437', ['Gene', 'Gene_type']] = ['snoU13', 'small_nucleolar_rna_gene']
        self.All_hits.loc['chr8:19822809', ['Gene', 'Gene_type']] = ['LPL', 'protein_coding']
        self.All_hits.loc['chr8:75679126', ['Gene', 'Gene_type']] = ['MIR2052HG', 'rna_gene']
        self.All_hits.loc['chr10:18138488', ['Gene', 'Gene_type']] = ['MRC1', 'protein_coding']
        self.All_hits.loc['chr10:96084372', ['Gene', 'Gene_type']] = ['PLCE1', 'protein_coding']
        self.All_hits.loc['chr11:293001', ['Gene', 'Gene_type']] = ['PGGHG', 'protein_coding']
        self.All_hits.loc['chr15:74282833', ['Gene', 'Gene_type']] = ['STOML1', 'protein_coding']
        self.All_hits.loc['chr15:89859932', ['Gene', 'Gene_type']] = ['FANCI, POLG', 'protein_coding, protein_coding']
        self.All_hits.loc['chr17:44341869', ['Gene', 'Gene_type']] = ['AC005829.1', 'pseudo_gene']
        self.All_hits.loc['chr17:79911164', ['Gene', 'Gene_type']] = ['NOTUM', 'protein_coding']
        self.All_hits.loc['chr20:57829821', ['Gene', 'Gene_type']] = ['ZNF831', 'protein_coding']
        self.All_hits.loc['chr22:29130347', ['Gene', 'Gene_type']] = ['CHEK2', 'protein_coding']
        # The following genes were not found by locuszoom, so I used https://www.rcsb.org/pdb/chromosome.do :
        self.All_hits.loc['chr6:31084935', ['Gene', 'Gene_type']] = ['CDSN', 'protein_coding']
        self.All_hits.loc['chr6:31105857', ['Gene', 'Gene_type']] = ['PSORS1C2', 'protein_coding']
        # The following genes was named "0" in locuszoom, so I used https://www.rcsb.org/pdb/chromosome.do :
        self.All_hits.loc['chr23:13771531', ['Gene', 'Gene_type']] = ['OFD1', 'protein_coding']
        # The following gene did not have a match
        self.All_hits.loc['chr23:56640134', ['Gene', 'Gene_type']] = ['UNKNOWN', 'UNKNOWN']
        
        # Ensuring that all SNPs have been annotated
        print('Ensuring that all SNPs have been annotated:')
        assert self.All_hits['Gene'].isna().sum() == 0
        print('Passed.')
        
        # Counter number of unique genes involved, and generating the list
        unique_genes = list(set((', '.join(self.All_hits['Gene'].unique())).split(', ')))
        print('A total of ' + str(len(unique_genes)) + ' unique genes are associated with accelerated aging.')
        np.save(self.path_data + 'GWAS_unique_genes.npy', np.array(unique_genes))
    
    def postprocessing_hits(self):
        self.All_hits.drop(columns=['chrbp'], inplace=True)
        self.All_hits.to_csv(self.path_data + 'GWAS_hits_' + self.target + '_All_withGenes.csv', index=False)
        for organ in self.organs_XWAS:
            Hits_organ = self.All_hits[self.All_hits['organ'] == organ].drop(columns=['organ'])
            Hits_organ.to_csv(self.path_data + 'GWAS_hits_' + self.target + '_' + organ + '_withGenes.csv', index=False)
    
    def summarize_results(self):
        # Generate empty dataframe
        organs = ['*', '*instances01', '*instances1.5x', '*instances23', 'Abdomen', 'AbdomenLiver', 'AbdomenPancreas',
                  'Arterial', 'ArterialCarotids', 'ArterialPulseWaveAnalysis', 'Biochemistry', 'BiochemistryBlood',
                  'BiochemistryUrine', 'Brain', 'BrainCognitive', 'BrainMRI', 'Eyes', 'EyesFundus', 'EyesOCT',
                  'Hearing', 'Heart', 'HeartECG', 'HeartMRI', 'ImmuneSystem', 'Lungs', 'Musculoskeletal',
                  'MusculoskeletalFullBody', 'MusculoskeletalHips', 'MusculoskeletalKnees', 'MusculoskeletalScalars',
                  'MusculoskeletalSpine', 'PhysicalActivity']
        cols = ['Organ', 'Sample size', 'SNPs', 'Genes', 'Heritability', 'CA_prediction_R2']
        GWAS_summary = np.empty((len(organs), len(cols),))
        GWAS_summary.fill(np.nan)
        GWAS_summary = pd.DataFrame(GWAS_summary)
        GWAS_summary.index = organs
        GWAS_summary.columns = cols
        GWAS_summary['Organ'] = organs
        
        # Fill dataframe
        All_hits = pd.read_csv(self.path_data + 'GWAS_hits_' + self.target + '_All_withGenes.csv')
        Heritabilities = pd.read_csv(self.path_data + 'GWAS_heritabilities_' + self.target + '.csv', index_col=0)
        Performances = pd.read_csv(self.path_data + 'PERFORMANCES_bestmodels_alphabetical_instances_Age_test.csv')
        dict_organ_to_organ_view = {
            'AbdomenLiver': ['Abdomen', 'Liver'],
            'AbdomenPancreas': ['Abdomen', 'Pancreas'],
            'ArterialCarotids': ['Arterial', 'Carotids'],
            'ArterialPulseWaveAnalysis': ['Arterial', 'PulseWaveAnalysis'],
            'BiochemistryBlood': ['Biochemistry', 'Blood'],
            'BiochemistryUrine': ['Biochemistry', 'Urine'],
            'BrainCognitive': ['Brain', 'Cognitive'],
            'BrainMRI': ['Brain', 'MRI'],
            'EyesFundus': ['Eyes', 'Fundus'],
            'EyesOCT': ['Eyes', 'OCT'],
            'HeartECG': ['Heart', 'ECG'],
            'HeartMRI': ['Heart', 'MRI'],
            'MusculoskeletalFullBody': ['Musculoskeletal', 'FullBody'],
            'MusculoskeletalHips': ['Musculoskeletal', 'Hips'],
            'MusculoskeletalKnees': ['Musculoskeletal', 'Knees'],
            'MusculoskeletalScalars': ['Musculoskeletal', 'Scalars'],
            'MusculoskeletalSpine': ['Musculoskeletal', 'Spine']
        }
        
        for organ in organs:
            organ_hits = All_hits[All_hits['organ'] == organ]
            data_organ = pd.read_csv(self.path_data + 'GWAS_data_Age_' + organ + '.tab')
            GWAS_summary.loc[organ, 'Sample size'] = len(data_organ.index)
            GWAS_summary.loc[organ, 'SNPs'] = len(organ_hits['SNP'].unique())
            GWAS_summary.loc[organ, 'Genes'] = len(organ_hits['Gene'].unique())
            if organ in Heritabilities.index:
                GWAS_summary.loc[organ, 'Heritability'] = str(round(Heritabilities.loc[organ, 'h2'] * 100, 1)) + '+-' \
                                                          + str(round(Heritabilities.loc[organ, 'h2_sd'] * 100, 1))
            if organ in Performances['organ'].values:
                GWAS_summary.loc[organ, 'CA_prediction_R2'] = \
                    str(round(Performances[Performances['organ'] == organ]['R-Squared_all'].values[0] * 100,
                              1)) + '+-' + \
                    str(round(Performances[Performances['organ'] == organ]['R-Squared_sd_all'].values[0] * 100, 1))
            else:
                org, view = dict_organ_to_organ_view[organ]
                GWAS_summary.loc[organ, 'CA_prediction_R2'] = \
                    str(round(Performances[(Performances['organ'] == org) &
                                           (Performances['view'] == view)]['R-Squared_all'].values[0] * 100,
                              1)) + '+-' + \
                    str(round(Performances[(Performances['organ'] == org) &
                                           (Performances['view'] == view)]['R-Squared_sd_all'].values[0] * 100, 1))
        
        # Format dataframe
        for col in ['Sample size', 'SNPs', 'Genes']:
            GWAS_summary[col] = GWAS_summary[col].astype(int)
        
        # Save data
        print('GWAS summary:')
        print(GWAS_summary)
        GWAS_summary.to_csv(self.path_data + 'GWAS_summary' + self.target + '.csv')
        
        # Report the correlation between R2s and h2_gs:
        GWAS_summary.dropna(subset=['Heritability', 'CA_prediction_R2'], inplace=True)
        
        # Local helper function
        def extract_h2r2(row):
            h2 = float(row['Heritability'].split('+-')[0])
            r2 = float(row['CA_prediction_R2'].split('+-')[0])
            return pd.Series([h2, r2])
        
        h2r2s = GWAS_summary.apply(extract_h2r2, axis=1)
        Corr_h2s_r2s = h2r2s.corr().loc[0, 1]
        print('Correlation between R2s when predicting chronological age and h2_g when performing the GWAS = ' +
              str(round(Corr_h2s_r2s, 3)))
    
    def upload_data(self):
        files = ['snps_rs.txt', 'GWAS_genes_rs.txt', 'snps_chrbp.txt', 'GWAS_genes_chrbp.txt', 'All_hits_missing.csv',
                 'GWAS_unique_genes.npy']
        for organ in ['All'] + self.organs_XWAS:
            files.append('GWAS_hits_' + self.target + '_' + organ + '_withGenes.csv')
        files = [self.path_data + file for file in files]
        for file in files:
            os.system('scp ' + file +
                      ' al311@transfer.rc.hms.harvard.edu:/n/groups/patel/Alan/Aging/Medical_Images/data/')


class GWASPlots(Basics):
    
    def __init__(self, target=None):
        Basics.__init__(self)
        self.target = target
        self.organs = \
            [file.split('_')[2].replace('.csv', '') for file in glob.glob(self.path_data + 'GWAS_' + target + '_*.csv')]
        self.organs.sort()
        self.FDR_correction = 5e-8
        # 23 colors for plot, to maximize two by two contrast (mostly important for the volcano plot)
        self.dict_chr_to_colors = {'1': '#b9b8b5', '2': '#222222', '3': '#f3c300', '4': '#875692', '5': '#f38400',
                                   '6': '#a1caf1', '7': '#be0032', '8': '#c2b280', '9': '#848482', '10': '#008856',
                                   '11': '#555555', '12': '#0067a5', '13': '#f99379', '14': '#604e97', '15': '#f6a600',
                                   '16': '#b3446c', '17': '#dcd300', '18': '#882d17', '19': '#8db600', '20': '#654522',
                                   '21': '#e25822', '22': '#232f00', '23': '#e68fac'}
        self.dict_colors = {'light_gray': '#b9b8b5', 'black': '#222222', 'vivid_yellow': '#f3c300',
                            'strong_purple': '#875692', 'vivid_orange': '#f38400', 'very_light_blue': '#a1caf1',
                            'vivid_red': '#be0032', 'grayish_yellow': '#c2b280', 'medium_gray': '#848482',
                            'vivid_green': '#008856', 'dark_gray': '#555555', 'strong_blue': '#0067a5',
                            'strong_yellowish pink': '#f99379', 'violet': '#604e97', 'vivid_orange_yellow': '#f6a600',
                            'strong_purplish_red': '#b3446c', 'vivid_greenish_yellow': '#dcd300',
                            'strong_reddish_brown': '#882d17', 'vivid_yellow_green': '#8db600',
                            'vivid_yellowish_brown': '#654522', 'vivid_reddish_orange': '#e25822',
                            'deep_olive_green': '#232f00', 'strong_purplish_pink': '#e68fac'}
    
    def generate_manhattan_and_qq_plots(self):
        GWAS_All = None
        for organ in self.organs + ['All']:
            print('Generating Manhattan plot and QQ plot for ' + organ)
            # Preprocessing
            if organ == 'All':
                GWAS = GWAS_All
            else:
                GWAS = pd.read_csv('../data/GWAS_Age_' + organ + '.csv', usecols=['SNP', 'CHR', 'BP', 'P_BOLT_LMM_INF'])
                GWAS.set_index('SNP', drop=False, inplace=True)
                Genes = pd.read_csv('../data/GWAS_hits_Age_' + organ + '_withGenes.csv', index_col='SNP',
                                    usecols=['SNP', 'Gene'])
                GWAS['Gene'] = Genes['Gene']
                GWAS.loc[GWAS['Gene'].isna(), 'Gene'] = 'not significant'
                # replace 0 with numerical limit for p-values so that log can be safely taken by mhat
                GWAS['P_BOLT_LMM_INF'] = GWAS['P_BOLT_LMM_INF'].replace([0], [10 ** (-323)])
                # Create a column to label SNPs without ambiguity for 'All' organs (can have several hits for same SNP)
                GWAS['SNP_P'] = GWAS['SNP'] + '_' + GWAS['P_BOLT_LMM_INF'].astype(str)
                
            # Generate dictionary of annotations for Genes
            GWAS.sort_values(by='P_BOLT_LMM_INF', inplace=True)
            Genes_annotations = {}
            for chro in GWAS[GWAS['Gene'] != 'not significant']['CHR'].unique():
                df_chr = GWAS[GWAS['CHR'] == chro]
                Genes_annotations.update({df_chr['SNP_P'][0]: df_chr['Gene'][0]})
            GWAS.sort_values(by=['CHR', 'BP'], inplace=True)
            
            # Generate Manhattan plot
            color = [self.dict_chr_to_colors[str(ch)] for ch in GWAS['CHR'].unique()]
            kwargs = {'df': GWAS, 'chr': 'CHR', 'pv': 'P_BOLT_LMM_INF', 'gwas_sign_line': True,
                      'gwasp': self.FDR_correction, 'color': color, 'gfont': 4, 'gstyle': 1, 'r': 600, 'dim': (9, 4),
                      'axlabelfontsize': 11, 'markernames': Genes_annotations, 'markeridcol': 'SNP_P'}
            if GWAS['P_BOLT_LMM_INF'].min() < 10 ** (-23):
                kwargs.update({'ylm': (0, -np.log10(GWAS['P_BOLT_LMM_INF'].min()),
                                       -np.log10(GWAS['P_BOLT_LMM_INF'].min()) / 10)})
            visuz.marker.mhat(**kwargs)
            os.rename('manhatten.png', '../figures/GWAS/GWAS_ManhattanPlot_' + self.target + '_' + organ + '.png')
            
            # Generate QQ plot
            # Prepare data
            observed = GWAS['P_BOLT_LMM_INF'].sort_values()
            lobs = -np.log10(observed)
            expected = np.array(range(len(lobs))) + 1
            lexp = -(np.log10(expected / (len(expected) + 1)))
            # Plot figure
            plt.clf()
            plt.scatter(lexp, lobs, marker='*', color='b', s=3)
            plt.plot(lexp, lexp, color='r')
            plt.xlabel('Expected -log10(p)')
            plt.ylabel('Observed -log10(p)')
            plt.savefig('../figures/GWAS/GWAS_QQPlot_' + self.target + '_' + organ + '.png', dpi=600)
            
            # Append the dataframe to the union of all dataframes for the 'All' plot
            if organ != 'All':
                if GWAS_All is None:
                    GWAS_All = GWAS
                else:
                    GWAS_All = GWAS_All.append(GWAS)
