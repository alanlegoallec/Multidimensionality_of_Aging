"""
Created on Feb 06 2020

@author: Sasha Collin
"""

import numpy as np
import os
import keras
import gc
import pandas as pd
import cv2
import re


class DataFetcher:
    # functions beginning by '_' should not be called outside the class definition
    """
    ID: eid_instance.
    """

    def __init__(self, target, TS_type, debug_mode=False):
        """
        :param target: string, target's type
        :param TS_type: strings (type of the time series)
        :param debug_mode: bool, if True, only a subset of a data is loaded (useful for created model architecture
        for instance)
        """
        # path of the folder containing all the data
        self.data_path = '../series'
        # path to DataFrame where eid, instance, age and sex are stored
        self.path_store = '../../Medical_Images/data'
        # path to DataFrame where Physical Activity tabular data are stored
        self.PA_all_features = '../series/PhysicalActivity/90001/features/PA_all_features.csv'
        self.model2bis_suffix = '_model2bis'

        self.target = target
        self.TS_type = TS_type

        # list because it can be split (e.g. 6 leads ecg => 3 leads for resting and 3 leads for exercising)
        # or also because there can be several time series, e.g. combination of AS and ECG_exercising
        self.X = []
        self.Xtab = None
        self.y = None

        # LOADING data-features_instances.csv
        # dtype = {"eid": np.int64, "instance": np.float16, "Age": np.float32, 'Sex': np.int8},
        self.data_features = pd.read_csv(os.path.join(self.path_store, 'data-features_instances.csv')).rename(
            columns={"Age": "age", "Sex": "sex"},
            errors="raise")

        # DEBUG MODE PARAMETERS
        self.debug_mode = debug_mode
        # number of samples used for debug mode
        self.debug_n_samples = 20

        if self.TS_type == 'ECG_exercising':
            self.data_path = '../series/ECG/exercising/resting_and_after_constant_effort/'
            self.X_subpath = ['ECG_exercising_6leads_X.npy']
            self.Xtab_subpath = 'ECG_exercising_6leads_Xtab.npy'
            self.y_subpath = 'ECG_exercising_6leads_y.npy'
        else:
            self.data_path = os.path.join(self.data_path, self.TS_type.replace('-', '/'))
            self.X_subpath = ['X.npy']
            self.Xtab_subpath = 'Xtab.npy'
            self.y_subpath = 'y.npy'
            self.ID_subpath = 'ID.npy'

    def _load_data(self):
        """
        Download data according to the TS_type
        """

        # LOADING X DATA
        for path in self.X_subpath:
            if self.debug_mode:
                self.X.append(resize_data(np.load(os.path.join(self.data_path, path),
                                                  mmap_mode='r')[:self.debug_n_samples], 3))
            else:
                self.X.append(resize_data(np.load(os.path.join(self.data_path, path), mmap_mode='r'), 3))
        # LOADING ID
        if self.debug_mode:
            ID = np.load(os.path.join(self.data_path, self.ID_subpath))[:self.debug_n_samples]
        else:
            ID = np.load(os.path.join(self.data_path, self.ID_subpath))

        # FETCHING TARGETS AND TAB DATA FROM DATAFRAME
        if 'PhysicalActivity' in self.TS_type:
            # UPDATING INSTANCE IF TS_type is PhysicalActivity
            df_ID = pd.DataFrame({
                'eid': ID[:, 0],
                'instance': ID[:, 1] / 100 + 1.5
            })
            print('Instances have been modified for Physical Activity Data.')
            print('Nb of data: ', len(df_ID))
            print(df_ID.head())
        else:
            df_ID = pd.DataFrame({
                'eid': ID[:, 0],
                'instance': ID[:, 1]
            })

        # add tabular data if model2bis of Physical Activity is used
        if self.TS_type == 'PhysicalActivity-90001-TimeSeries-epochs-5min-model2bis':
            print("Adding Physical Activity tabular data.")
            PA_tab = pd.read_csv(self.PA_all_features)
            PA_tab[['instance']] = PA_tab.id.apply(lambda s: pd.Series({'instance': float(s.split('_')[1])}))
            PA_tab.drop(columns=['id'], inplace=True)
            PA_tab = PA_tab.add_suffix(self.model2bis_suffix)
            PA_tab.rename(columns={'eid' + self.model2bis_suffix: 'eid',
                                   'instance' + self.model2bis_suffix: 'instance'},
                          inplace=True)
            df_ID = df_ID.merge(PA_tab, on=['eid', 'instance'])

        df_merge = df_ID.merge(self.data_features, on=['eid', 'instance'])
        if len(df_merge) != ID.shape[0]:
            raise Exception('Missing Data in DataFrame!')
        self.eid = df_merge['eid'].apply(np.int32).to_numpy()
        self.instance = df_merge['instance'].apply(np.float64).to_numpy()
        self.y = df_merge['age'].apply(np.float32).to_numpy()
        tabular_features = [feature for feature in df_merge.columns \
                            if re.search('(?:Ethnicity|{})'.format(self.model2bis_suffix), feature)]
        tabular_features.append('sex')
        self.Xtab = resize_data(df_merge[tabular_features].apply(np.float32).to_numpy(), 2)

    def _split_data(self):
        """
        Split the data if necessary (depends on TS_type)
        """
        if self.TS_type == 'ECG_exercising' and len(self.X) == 1:
            self.X = [self.X[0][:, :, :3], self.X[0][:, :, 3:]]

    def get_data(self):
        """
        Load and split the data according to TS_type

        Return dictionary with:
        - X: self.X: list of arrays (data)
        - Xtab: self.Xtab: array (tabular data)
        - y: self.y: array (labels)
        """
        # load data only if it's not already done
        if not self.X:
            self._load_data()
        self._split_data()

        return {'X': self.X, 'Xtab': self.Xtab, 'y': self.y, 'eid': self.eid, 'instance': self.instance}


class DataPreprocessor:
    # functions beginning by '_' should not be called outside the class definition
    def __init__(self, model_type, sub_model_type, X, Xtab=None):
        """
        Preprocess data to feed them into the model.
        :param model_type: string, type of the model
        :param X: list of arrays, data
        :param Xtab: array, tabular data
        """
        self.model_type = model_type
        self.sub_model_type = sub_model_type
        self.X = X
        self.Xtab = Xtab

        # leads dictionary
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.names_to_nbs = dict(zip(self.lead_names, np.arange(len(self.lead_names))))

        if self.Xtab is not None:
            # making sure Xtab has the right shape (to be an input for the future models)
            if len(self.Xtab.shape) == 1:
                self.Xtab.resize(self.Xtab.shape + (1,))

        # preprocessing dictionnary
        self.pp_dict = {'SimpleConv2Dv1': self.SimpleConv2Dv1,
                        'MultiInputModel': self.MultiInputModel,
                        'MortalityModel': self.MortalityModel,
                        'DiagnosisModel': self.DiagnosisModel,
                        'PATSConv1DModel': self.PATSConv1DModel,
                        'PATSConvLSTM2D': self.PATSConvLSTM2D}

        # formatted data
        self.X_f = []

    def MultiInputModel(self):
        """
        Formatting of time series: dividing each lead by the max of its absolute value.
        Storing division factors.

        :attribute X_f: list of float arrays (formatted) with shape (n_samples,n_time_steps,n_features): formatted
        time series
        :attribute coefs: list of float arrays with shape (n_samples,n_features): coefs by which each
        lead has been divided
        """

        for k in range(len(self.X)):
            time_series = self.X[k]

            n_samples, n_time_steps, n_features = time_series.shape

            time_series_f = np.zeros(time_series.shape)
            coefs_ = np.zeros((n_samples, n_features))

            for line in range(n_samples):
                coefs_[line, :] = np.amax(np.abs(time_series[line, :, :]), axis=0)
                time_series_f[line, :, :] = time_series[line, :, :] / coefs_[line, :]

            # Expanding dimensions of time series if necessary (depending on the sub_model type)
            if self.sub_model_type == 'MultiInputConv2D' or self.sub_model_type == 'Conv2D':
                time_series_f.resize(time_series_f.shape + (1,))

            self.X_f.append(time_series_f)

            if self.sub_model_type == 'MultiInputConv1D' or self.sub_model_type == 'MultiInputConv2D':
                self.X_f.append(coefs_)

        self.X_f.append(self.Xtab)

        return self.X_f

    def SimpleConv2Dv1(self):
        """
        For SimpleConv2Dv1 model.
        Increase dimension of arrays in X.
        Nothing done to Xtab.
        """
        for i, x in enumerate(self.X):
            # right shape
            if len(x.shape) < 4:
                x.resize(x.shape + (1,))

            # padding
            self.X[i] = np.pad(x, ((0, 0), (60, 60), (0, 0), (0, 0)), 'constant')
        # self.X is just a list of one array
        return self.X

    def MortalityModel(self):
        """
        Format data for MortalityModel.
        """
        # padding
        self.X[0] = np.pad(self.X[0], ((0, 0), (20, 20), (0, 0)), 'constant')

        leads_groups = [['V1', 'II', 'V5'],
                        ['I', 'II', 'III'],
                        ['aVR', 'aVL', 'aVF'],
                        ['V1', 'V2', 'V3'],
                        ['V4', 'V5', 'V6']]

        for group in leads_groups:
            self.X_f.append(self.X[0][:, :, np.array([self.names_to_nbs[lead] for lead in group])])

        self.X_f.append(self.Xtab)
        del self.X

        return self.X_f

    def DiagnosisModel(self):
        """
        Format data for DiagnosisModel.
        WARNING: For this model, input must be: {I, II, III, AVL, AVF, AVR, V1, V2, V3, V4, V5, V6}.
        """

        def resample(ar, new_shape):
            """
            Resample ar according to new_shape. Use bilinear interpolation.
            :param ar: array
            :param new_shape: tuple of ints
            :return: array
            """
            resampled_ar = np.empty(new_shape)
            for k in range(new_shape[0]):
                resampled_ar[k] = cv2.resize(ar[k], (new_shape[2], new_shape[1]))
            return resampled_ar

        normalization_factor = 5 / 100

        # leads order
        leads_order = ['I', 'II', 'III', 'aVL', 'aVF', 'aVR', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

        # swap leads and resample them
        self.X_f.append(resample(self.X[0][:, :, [self.names_to_nbs[lead] for lead in leads_order]],
                                 (self.X[0].shape[0], 4096, 12)) * normalization_factor)
        return self.X_f

    def PATSConv1DModel(self):
        """
        Format data for PATSConv1DModel.
        Nothing to be done.
        """
        self.X_f = self.X
        return self.X_f

    def PATSConvLSTM2D(self):
        """
        For PATSConvLSTM2D model.
        """
        self.X_f = [resize_data(self.X[0], 5)]
        return self.X_f

    def get_data(self):
        """
        Return preprocessed data.
        """
        return self.pp_dict[self.model_type]()


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, model_type, sub_model_type, labeled_data, raw_indexes, batch_size=32, shuffle=True):
        """
        Initialization
        :param model_type: string, type of the model (ex: MultiInputModel)
        :param sub_model_type: string, subtype of the model (ex: Conv1D)
        :param labeled_data: dictionary where data are stored: 'X', 'Xtab', 'y'
            - 'X': list of numpy arrays read in mode mmap_mode = 'r'
            - 'Xtab': numpy array
            - 'y': numpy array (labels)
        :param raw_indexes: list of ints, indexes of data that are considered
        :param batch_size: int
        :param shuffle: bool
        """
        self.model_type = model_type
        self.sub_model_type = sub_model_type
        self.data = labeled_data['X']
        self.labels = labeled_data['y']
        self.data_tab = labeled_data['Xtab']
        self.data_tab_exists = self.data_tab is not None
        self.batch_size = batch_size
        self.raw_indexes = raw_indexes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.raw_indexes) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        raw_indexes_temp = [self.raw_indexes[k] for k in indexes]

        # Generate data
        return self.__data_generation(raw_indexes_temp)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # manually triggering garbage collection
        gc.collect()

        self.indexes = np.arange(len(self.raw_indexes))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, raw_indexes_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)

        if self.data_tab_exists:
            dp = DataPreprocessor(self.model_type, self.sub_model_type, [d[raw_indexes_temp] for d in self.data],
                                  self.data_tab[raw_indexes_temp])
        else:
            dp = DataPreprocessor(self.model_type, self.sub_model_type, [d[raw_indexes_temp] for d in self.data])

        return dp.get_data(), self.labels[raw_indexes_temp]


class DataGeneratorForPredictions(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, model_type, sub_model_type, labeled_data, raw_indexes, batch_size=32):
        """
        Initialization
        :param model_type: string, type of the model (ex: MultiInputModel)
        :param sub_model_type: string, subtype of the model (ex: Conv1D)
        :param labeled_data: dictionary where data are stored: 'X', 'Xtab', 'y'
            - 'X': list of numpy arrays read in mode mmap_mode = 'r'
            - 'Xtab': numpy array
            - 'y': numpy array (labels)
        :param raw_indexes: list of ints, indexes of data that are considered
        :param batch_size: int
        :param shuffle: bool
        """
        self.model_type = model_type
        self.sub_model_type = sub_model_type
        self.data = labeled_data['X']
        self.labels = labeled_data['y']
        self.data_tab = labeled_data['Xtab']
        self.data_tab_exists = self.data_tab is not None
        self.batch_size = batch_size
        self.raw_indexes = raw_indexes
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        nb_batches = len(self.raw_indexes) / self.batch_size
        if int(nb_batches) != nb_batches:
            # last batch will be shorter
            return int(np.floor(nb_batches)) + 1
        else:
            return int(np.floor(nb_batches))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size: min((index + 1) * self.batch_size,
                                                            len(self.indexes))]

        # Find list of IDs
        raw_indexes_temp = [self.raw_indexes[k] for k in indexes]

        # Generate data
        return self.__data_generation(raw_indexes_temp)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        # manually triggering garbage collection
        gc.collect()

        self.indexes = np.arange(len(self.raw_indexes))

    def __data_generation(self, raw_indexes_temp):
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)

        if self.data_tab_exists:
            dp = DataPreprocessor(self.model_type, self.sub_model_type, [d[raw_indexes_temp] for d in self.data],
                                  self.data_tab[raw_indexes_temp])
        else:
            dp = DataPreprocessor(self.model_type, self.sub_model_type, [d[raw_indexes_temp] for d in self.data])

        return dp.get_data(), self.labels[raw_indexes_temp]

# useful functions
def resize_data(data, n_dims):
    """
    Resize data if they don't have enough dimensions
    :param data: array, data to resize potentially
    :param n_dims: int, nb of dimensions data must have
    :return: array, resized data
    """
    while len(data.shape) < n_dims:
        data = data.reshape(data.shape + (1,))

    return data

