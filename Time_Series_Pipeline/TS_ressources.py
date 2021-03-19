"""
Created on Feb 06 2020

@author: Sasha Collin
"""

# setting seeds
seed_value = 0
import os

os.environ['PYTHONHASHSEED'] = str(seed_value)
import random

random.seed(seed_value)
import numpy as np

np.random.seed(seed_value)
import tensorflow as tf

tf.compat.v1.set_random_seed(seed_value)
from keras import backend as K

# keras
from keras.layers import BatchNormalization, Activation, MaxPooling1D, MaxPooling2D, Conv1D, Conv2D, Flatten, Reshape, \
    Dense, Dropout, Input, concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D, ConvLSTM2D
from keras.models import Model, Sequential, load_model
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.optimizers import Adam
from keras import regularizers
from keras import backend as K

# sklearn
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

# GPU
import GPUtil

# miscellaneous
import csv
import sys
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import re

# visualization of models
import keract

from TS_preprocessing import *


#### class definitions ####

class hyperparameters:
    """
    Gather all the hyperparameters of the algorithm
    """

    def __init__(self, parameters=None, version=None):
        """

        :param parameters:  list of parameters
        :param version: string, name of the model in performances csv.
        """
        # print(parameters)
        # DEFAULT HYPER-PARAMETERS
        self.target = 'Age'
        self.TS_type = 'AS'
        self.model_type = 'MultiInputModel'
        self.sub_model_type = 'MultiInputConv1D'
        self.seed = 0
        self.fold = 0
        self.n_layers = 4
        self.n_epochs = 500
        self.batch_size = 1024
        self.dropout = 0.05
        self.kr_pow = '14'
        self.br_pow = '0'
        self.ar_pow = '0'
        self.learning_rate = 1e-3
        self.nodes_factor = 2
        self.overfitting = 1

        if parameters is not None and len(parameters) == 16:
            self.target = parameters[1]
            self.TS_type = parameters[2]
            full_model_type = parameters[3].split('_')
            if len(full_model_type) == 2:
                self.model_type, self.sub_model_type = full_model_type
            else:
                self.model_type, self.sub_model_type = full_model_type[0], ''
            self.seed = int(self.random_generator('seed', parameters[4]))

            # SEEDs RESETTING
            self.seeds_resetting()

            self.fold = int(parameters[5])
            self.n_layers = int(self.random_generator('n_layers', parameters[6]))
            self.n_epochs = int(parameters[7])
            self.batch_size = int(parameters[8])
            self.dropout = int(self.random_generator('dropout', parameters[9])) / 100
            self.kr_pow = self.random_generator('kr_pow', parameters[10])
            self.br_pow = self.random_generator('br_pow', parameters[11])
            self.ar_pow = self.random_generator('ar_pow', parameters[12])
            self.learning_rate = 1 * 10 ** (-int(self.random_generator('learning_rate', parameters[13])))
            self.nodes_factor = max(int(self.random_generator('nodes_factor', parameters[14])), 1)
            self.overfitting = int(self.random_generator('overfitting', parameters[15]))

        elif version is not None:
            parameters = version.split(',')
            for k, p in enumerate(self.__dict__):
                types = [str, str, str, str, int, int, int, int, int, float, str, str, str, float, int, int]
                self.__dict__[p] = types[k](parameters[k])

        else:
            # print('WRONG NUMBER OF HYPER-PARAMETERS, RUNNING WITH DEFAULT ONES')
            raise Exception('WRONG NUMBER OF PARAMETERS')

        # SEEDS RESETTING
        self.seeds_resetting()

        # increase batch size if possible
        self.increase_batch_size()

    def seeds_resetting(self):
        """
        Reset seeds.
        """
        print("Resetting seeds. Chosen seed: {}".format(self.seed))
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.compat.v1.set_random_seed(self.seed)

    @staticmethod
    def random_generator(parameter, value):
        """
        Generate a random value for the given parameter
        :param parameter: string, name of the parameter
        :param value: parameter value got from sys.argv
        """
        if value == '-1':
            if parameter == 'seed':
                return random.randint(0, 1000000)
            elif parameter == 'n_layers':
                return random.randint(2, 8)
            elif parameter == 'dropout':
                return random.randint(0, 50)
            elif parameter == 'kr_pow':
                # '15' => l2(1*10e-5)
                return str(random.randint(1, 9)) + str(random.randint(2, 7))
            elif parameter == 'br_pow':
                # '15' => l2(1*10e-5)
                return str(random.randint(1, 9)) + str(random.randint(2, 7))
            elif parameter == 'ar_pow':
                # '15' => l2(1*10e-5)
                return str(random.randint(1, 9)) + str(random.randint(2, 7))
            elif parameter == 'learning_rate':
                return random.randint(1, 4)
            elif parameter == 'nodes_factor':
                return random.randint(1, 2)
            elif parameter == 'overfitting':
                return random.randint(0, 1)
        return value

    def increase_batch_size(self):
        """
        If GPU's memory is large enough, double batch size.
        """
        if len(GPUtil.getGPUs()) > 0:  # make sure GPUs are available (not true sometimes for debugging)
            if GPUtil.getGPUs()[0].memoryTotal > 20000:
                self.batch_size *= 2
                print('### batch size has been doubled ###')

    def get_version_with_names(self, remove_non_relevant_param=True):
        """
        Return version of the model, without the non relevant parameters (if remove_non_relevant_param is True)
        :param remove_non_relevant_param: True
        :return: string
        """
        # list of non relevant parameters to characterize a model
        non_relevant_param = ['n_epochs', 'batch_size']

        version = ''
        for p in self.__dict__:
            if not remove_non_relevant_param or p not in non_relevant_param:
                version += p + '_' + str(self.__dict__[p]) + '_'
        return version.replace('.', ',')[:-1]

    def get_version(self, remove_non_relevant_param=True, specific_non_relevant_param=None):
        """
        Return version of the model, without the names of the hyperparameters, and without the non relevant parameters
        (if remove_non_relevant_param is True)
        :param remove_non_relevant_param: list of strings, name of the parameters not relevant for the version
        of the model
        :param specific_non_relevant_param: str or list of str (other non relevant parameters to take into account
        (case specific)
        :return: string
        """
        # list of non relevant parameters to characterize a model
        non_relevant_param = ['n_epochs', 'batch_size']

        # UPDATING non_relevant_param
        if specific_non_relevant_param is not None:
            if type(specific_non_relevant_param) == list:
                non_relevant_param += specific_non_relevant_param
            else:
                non_relevant_param += [specific_non_relevant_param]

        version = ''
        for p in self.__dict__:
            if not remove_non_relevant_param or p not in non_relevant_param:
                version += str(self.__dict__[p]) + '_'
        return version.replace('.', ',')[:-1]

    def update_from_model_name(self, model_name):
        """
        Update hyperparameters from model_name.
        Caveats: in model_name, there is not n_epochs and batch_size
        :param model_name:
        """
        parameters = model_name.replace('.h5', '').split('_')
        self.target = parameters[1]
        self.TS_type = parameters[2]
        self.model_type = parameters[3]
        self.sub_model_type = parameters[4]
        self.seed = int(parameters[5])
        self.fold = int(parameters[6])
        self.n_layers = int(parameters[7])
        self.dropout = float(parameters[8].replace(',', '.'))
        self.kr_pow = parameters[9]
        self.br_pow = parameters[10]
        self.ar_pow = parameters[11]
        self.learning_rate = float(parameters[12].replace(',', '.'))
        self.nodes_factor = int(parameters[13])
        self.overfitting = int(parameters[14])


class TS_model_architecture:
    # functions beginning by '_' should not be called outside the class definition
    """
    Create model architecture.
    HOW TO ADD A MODEL?
    - add model's architecture (create a new function in TS_model_architecture)
    - don't forget to give a self.model_name_for_predictions to your model if it's not Conv1D
    - in DataPreprocessor, add a function to do the right preprocessing for this model
    - complete _data_to_X_Xtab()
    - complete dictionaries :
        - self.models in TS_model_architecture
        - self.pp_dict in DataProcessor
    """

    def __init__(self, hp, data):
        """
        ### Parameters ###
        :param hp: hyperparameters object
        :param data: list of arrays (X and Xtab if exists)

        ### data shape depends on the targeted model ###
        SimpleConv2Dv1: data is a list of only one array.

        MultiInputModel: data is a list of normalized data (e.g. leads) and their normalization factors.
        Normalization factors are considered as tabular data. Therefore, for this model, data looks like (X_1,
        Xtab_1, X_2, Xtab_2, ..., X_n, Xtab_n, Xtab_indep). X_i and Xtab_i are associated (Xtab_i is the
        normalization factors array associated to X_i). Xtab_indep is 'independent' of X's (patient's age).

        ### hp's parameters useful for all models ###
        :param model_type: string, type of the model
        :param dropout: float (between 0 and 1)
        :param kr: float, kernel regularizer
        :param br: float, bias regularizer
        :param ar: float, activity regularizer
        :param lr: float, learning rate

        hp's parameters useful only for MultiInputModel:
        :param n_layers: int, number of convolutional layers
        :param nodes_factor: int, number by which default nodes numbers are multiplied
        """

        self.hp = hp
        self.model_type = self.hp.model_type
        self.sub_model_type = self.hp.sub_model_type
        self._data_to_X_Xtab(data)

        # path to pretrained_models
        self.pm_path = '../data/pretrained_models'

        # regularization parameters
        self.dropout = self.hp.dropout
        self.kr = self.get_regularizer(self.hp.kr_pow)
        self.br = self.get_regularizer(self.hp.br_pow)
        self.ar = self.get_regularizer(self.hp.ar_pow)
        self.lr = self.hp.learning_rate

        # default convolution layer parameters
        self.MP_factor = 2  # max_pooling factor
        self.conv_size = 3  # convolution size
        self.padding = 'same'  # padding type

        # names to models
        self.models = {'MultiInputModel': self.MultiInputModel,
                       'SimpleConv2Dv1': self.SimpleConv2Dv1,
                       'MortalityModel': self.MortalityModel,
                       'DiagnosisModel': self.DiagnosisModel,
                       'PATSConv1DModel': self.PATSConv1DModel,
                       'PATSConvLSTM2D': self.PATSConvLSTM2D}
        # created model
        self.model = None
        # model name displayed in predictions title files
        self.model_name_for_predictions = '1DCNN'

    @staticmethod
    def get_regularizer(power, norm=regularizers.l2):
        """
        Return the regularizer associated to power and norm
        For bash simplification, the use of power is the following : the first figure is the significant figure, and the second is the power
        e.g. : power = '15' => lam = 1e-5 ; power = '56' => lam = 5e-6
        :param power: string.
        :param norm: regularizer
        :return: regularizer
        """
        if power != '0':
            lam = int(power[0]) * 10 ** (-int(power[1]))
            return (norm(lam))
        else:
            return None

    def _data_to_X_Xtab(self, data):
        """
        According to the model_type, extract X's and Xtab's shapes from data
        X_shapes and Xtab_shapes are lists of tuples (or None if they don't exist)
        :param data: see __ini__()
        """
        if self.model_type == 'SimpleConv2Dv1':
            self.X_shapes = [data[0].shape[1:]]
            self.Xtab_shapes = None
        elif self.sub_model_type == 'MultiInputConv1D' or self.sub_model_type == 'MultiInputConv2D':
            self.X_shapes = []
            self.Xtab_shapes = []
            for k in range(0, len(data) - 1, 2):
                self.X_shapes.append(data[k].shape[1:])
                self.Xtab_shapes.append(data[k + 1].shape[1:])
            self.Xtab_shapes.append(data[-1].shape[1:])
        elif self.sub_model_type == 'Conv1D' or self.sub_model_type == 'Conv2D':
            self.X_shapes = [data[k].shape[1:] for k in range(len(data) - 1)]
            self.Xtab_shapes = [data[-1].shape[1:]]
        elif self.model_type == 'MortalityModel':
            self.X_shapes = [data[k].shape[1:] for k in range(len(data) - 1)]
            self.Xtab_shapes = [data[-1].shape[1:]]
        elif self.model_type == 'DiagnosisModel':
            # no need to get samples shapes as model is loaded
            pass
        elif self.model_type == 'PATSConv1DModel':
            self.X_shapes = [data[0].shape[1:]]
        elif self.model_type == 'PATSConvLSTM2D':
            self.X_shapes = [data[0].shape[1:]]
        else:
            raise Exception('No corresponding model')

    def SimpleConv2Dv1(self):
        """
        For ECG.
        Model from the article "Age and Sex estimation using artificial intelligence from Standard 12-lead ECGs".
        :return:
        """
        self.model_name_for_predictions = 'Conv2D'

        padding = 'same'
        n_leads = self.X_shapes[0][1]
        model = Sequential()
        conv_size = [7, 5, 5, 5, 5, 3, 3, 3]
        nodes = [16, 16, 32, 32, 64, 64, 64, 64]
        fc_nodes = [128, 64]
        n_spatial_filers = 128
        MP_factor = [2, 4, 2, 4, 2, 2, 2]
        spatial_MP_factor = 2
        model.add(
            Conv2D(nodes[0], (conv_size[0], 1), input_shape=self.X_shapes[0], padding=padding,
                   kernel_regularizer=self.kr,
                   bias_regularizer=self.br, activity_regularizer=self.ar))
        for k in range(8):
            if k > 0:
                model.add(
                    Conv2D(nodes[k], (conv_size[k], 1), padding=padding, kernel_regularizer=self.kr,
                           bias_regularizer=self.br, activity_regularizer=self.ar))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            if k < 7:
                model.add(MaxPooling2D((MP_factor[k], 1)))
        model.add(Conv2D(n_spatial_filers, (1, n_leads), padding='valid', kernel_regularizer=self.kr,
                         bias_regularizer=self.br, activity_regularizer=self.ar))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((spatial_MP_factor, 1)))
        model.add(Flatten())
        for k in range(2):
            model.add(Dense(fc_nodes[k], kernel_regularizer=self.kr,
                            bias_regularizer=self.br, activity_regularizer=self.ar))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(self.dropout))
        model.add(Dense(1, activation='linear'))
        self.model = model

    def DenseLayersToOutput(self, tensor, nodes):
        """
        Build the last dense layers of a model.
        :param tensor: tf.Tensor
        :param nodes: list of DLs nodes
        :return: tf.Tensor, (tensor giving model's output)
        """
        for k, node in enumerate(nodes):
            if k == 0:
                dense = Dense(node, kernel_regularizer=self.kr,
                              bias_regularizer=self.br,
                              activity_regularizer=self.ar)(tensor)
            else:
                dense = Dense(node, kernel_regularizer=self.kr,
                              bias_regularizer=self.br,
                              activity_regularizer=self.ar)(dense)
            dense = BatchNormalization()(dense)
            dense = Activation('relu')(dense)
            dense = Dropout(self.dropout)(dense)
        return dense

    def AddTabData(self, input_nb, tensor):
        """
        Concatenate tabular data to tensor 'nodes' and add dense layers on top of this.

        :param input_nb: index of the signal in self.Xtab_shapes
        :param tensor: tf.Tensor
        :return: input layer for tabular data, output tensor
        """
        # tabular data
        visible2 = Input(shape=self.Xtab_shapes[input_nb])
        # number of nodes for tabular data >= 10% of nodes from time series
        tab_dense = Dense(max(max(10, self.Xtab_shapes[input_nb][0]), K.int_shape(tensor)[-1] // 10))(visible2)

        # merge input models
        merge = concatenate([tensor, tab_dense])

        # computing number of nodes for the next dense layers
        merge_nodes = K.int_shape(tensor)[-1] + K.int_shape(tab_dense)[-1]

        # power of two just below nb_nodes
        p2 = int(np.log(merge_nodes) / np.log(2))
        DL_nodes = [min(2 ** k, 1024) for k in range(p2, p2 - 2, -1)]

        # DENSE LAYERS
        dense = self.DenseLayersToOutput(merge, DL_nodes)

        return visible2, dense

    def MultiInputConv1D(self, input_nb, add_tab_data=True):
        """
        Create the architecture to analyse a single normalized group of time series (as the 12 leads of an ECG) with the tabular data associated
        :param input_nb: index of the signal in the list self.X_shapes and self.Xtab_shapes
        :param add_tab_data: bool, if true, call AddTabData for tab data associated to the time series n°input_nb.
        :return: tensor input1 (time series), tensor input2 (tabular data), output tensor
        """
        ts_length = self.X_shapes[input_nb][0]  # length of ts

        # time series
        visible1 = Input(shape=self.X_shapes[input_nb])

        current_ts_length = ts_length  # current length of time series (change because of MP)
        layer = 0

        while current_ts_length >= max(self.MP_factor, self.conv_size) and layer < self.n_layers:
            if layer == 0:
                cnn = Conv1D(self.nodes[layer], self.conv_size, padding=self.padding, kernel_regularizer=self.kr,
                             bias_regularizer=self.br,
                             activity_regularizer=self.ar)(visible1)
            else:
                cnn = Conv1D(self.nodes[layer], self.conv_size, padding=self.padding, kernel_regularizer=self.kr,
                             bias_regularizer=self.br,
                             activity_regularizer=self.ar)(cnn)
            cnn = BatchNormalization()(cnn)
            cnn = Activation('relu')(cnn)
            cnn = MaxPooling1D(self.MP_factor)(cnn)
            layer += 1
            current_ts_length = K.int_shape(cnn)[1]
        cnn = GlobalMaxPooling1D()(cnn)

        if add_tab_data:
            visible2, dense = self.AddTabData(input_nb, cnn)
            return visible1, visible2, dense
        else:
            return visible1, cnn

    def MultiInputConv2D(self, input_nb, add_tab_data=True):
        """
        Process each time series (ts) of a group independently.
        For instance, each lead of a 12 leads ECG will processed independently.
        Once each ts has been processed, info extracted are gathered and processed together, with the tabular data
        :param add_tab_data: bool, if true, call AddTabData for tab data associated to the time series n°input_nb.
        """
        self.model_name_for_predictions = 'Conv2D'

        ts_length = self.X_shapes[input_nb][0]  # length of ts

        # time series
        visible1 = Input(self.X_shapes[input_nb])

        current_ts_length = ts_length  # current length of time series (change because of MP)
        layer = 0

        while current_ts_length >= max(self.MP_factor, self.conv_size) and layer < self.n_layers:
            if layer == 0:
                cnn = Conv2D(self.nodes[0], (self.conv_size, 1), padding=self.padding, kernel_regularizer=self.kr,
                             bias_regularizer=self.br,
                             activity_regularizer=self.ar)(visible1)
            else:
                cnn = Conv2D(self.nodes[layer], (self.conv_size, 1), padding=self.padding, kernel_regularizer=self.kr,
                             bias_regularizer=self.br,
                             activity_regularizer=self.ar)(cnn)
            cnn = BatchNormalization()(cnn)
            cnn = Activation('relu')(cnn)
            cnn = MaxPooling2D((self.MP_factor, 1))(cnn)
            layer += 1
            current_ts_length = K.int_shape(cnn)[1]
        cnn = Conv2D(self.nodes[layer - 1], (current_ts_length, 1), padding='valid')(cnn)
        cnn = Flatten()(cnn)

        if add_tab_data:
            visible2, dense = self.AddTabData(input_nb, cnn)
            return visible1, visible2, dense
        else:
            return visible1, cnn

    def MultiInputModel(self):
        """
        Create architecture taking in input several time series + tabular data.
        Each time series, normalized, is processed in different CNN, with tabular data corresponding to its normalization.
        All the outputs of the CNNs are then gathered, with the independent tabular data and are processed together.
        """

        # hyperparameters
        self.n_layers = self.hp.n_layers
        self.nodes_factor = self.hp.nodes_factor

        if self.sub_model_type == 'MultiInputConv2D' or self.sub_model_type == 'Conv2D':
            # nodes of the convolutional layers
            self.nodes = self.nodes_factor * np.array([16, 16, 32, 32, 64, 64, 64, 64])[:self.n_layers]
        elif self.sub_model_type == 'MultiInputConv1D' or self.sub_model_type == 'Conv1D':
            # nodes of the convolutional layers
            self.nodes = self.nodes_factor * np.array([min(2 ** k, 1024) for k in range(4, self.n_layers + 4)])
            # self.nodes = [min(n, 1024) for n in self.nodes]
        else:
            raise Exception('No sub_model found')

        # nb of time series
        self.n_ts = len(self.X_shapes)
        # nb of tabular data
        if self.Xtab_shapes is not None:
            self.n_tab = len(self.Xtab_shapes)

        inputs = []

        cnns = []
        for k in range(self.n_ts):
            if self.sub_model_type == 'MultiInputConv2D':
                visible1, visible2, cnn = self.MultiInputConv2D(k)
            elif self.sub_model_type == 'Conv2D':
                visible1, cnn = self.MultiInputConv2D(k, add_tab_data=False)
            elif self.sub_model_type == 'MultiInputConv1D':
                visible1, visible2, cnn = self.MultiInputConv1D(k)
            elif self.sub_model_type == 'Conv1D':
                visible1, cnn = self.MultiInputConv1D(k, add_tab_data=False)
            else:
                raise Exception('No sub_model found')
            inputs.append(visible1)
            if self.sub_model_type == 'MultiInputConv1D' or self.sub_model_type == 'MultiInputConv2D':
                inputs.append(visible2)
            cnns.append(cnn)

        visibles = []
        tab_denses = []
        # independent tabular data
        visible = Input(shape=self.Xtab_shapes[-1])
        visibles.append(visible)
        tab_dense = Dense(max(max(10, self.Xtab_shapes[-1][0]), sum([K.int_shape(c)[-1] for c in cnns]) // 100))(
            visible)
        tab_denses.append(tab_dense)

        inputs = inputs + visibles

        # merging
        merge = concatenate(cnns + tab_denses)

        # computing number of nodes for the next dense layers
        merge_nodes = sum([K.int_shape(c)[-1] for c in cnns]) + sum([K.int_shape(tab)[-1] for tab in tab_denses])

        # power of two just below nb_nodes
        p2 = int(np.log(merge_nodes) / np.log(2))
        DL_nodes = [min(2 ** k, 1024) for k in range(p2, max(p2 - 2, 1), -1)]

        # DENSE LAYERS
        dense = self.DenseLayersToOutput(merge, DL_nodes)
        dense = Dense(1)(dense)

        self.model = Model(inputs=inputs, outputs=dense)

    def CellConv1Dv1(self, X_shape, nodes):
        """
        Convolution 1D cell: series of (Conv1D, Batch Normalization, ReLu).
        Stride for convolution is not null, EXCEPT for the last 2 convolutions.

        :param X_shape: shape of the input
        :param nodes: list of ints (nodes of Conv1D layers)
        """
        visible1 = Input(shape=X_shape)

        for k, node in enumerate(nodes):
            if k == 0:
                cnn = Conv1D(nodes[k], self.conv_size, strides=2, padding='same', kernel_regularizer=self.kr,
                             bias_regularizer=self.br, activity_regularizer=self.ar)(visible1)
            else:
                if k < len(nodes) - 1:
                    cnn = Conv1D(nodes[k], self.conv_size, strides=2, padding='same', kernel_regularizer=self.kr,
                                 bias_regularizer=self.br, activity_regularizer=self.ar)(cnn)
                else:
                    cnn = Conv1D(nodes[k], self.conv_size, strides=1, padding='same', kernel_regularizer=self.kr,
                                 bias_regularizer=self.br, activity_regularizer=self.ar)(cnn)
            cnn = BatchNormalization()(cnn)
            cnn = Activation('relu')(cnn)

        cnn = GlobalAveragePooling1D()(cnn)
        return visible1, cnn

    def CellConv1Dv2(self, X_shape, nodes):
        """
        Convolution 1D cell: series of (Conv1D, Batch Normalization, ReLu, MaxPooling).

        :param X_shape: shape of the input
        :param nodes: list of ints (nodes of Conv1D layers)
        """
        visible1 = Input(shape=X_shape)

        for k, node in enumerate(nodes):
            if k == 0:
                cnn = Conv1D(nodes[k], self.conv_size, padding='same', kernel_regularizer=self.kr,
                             bias_regularizer=self.br, activity_regularizer=self.ar)(visible1)
            else:
                cnn = Conv1D(nodes[k], self.conv_size, padding='same', kernel_regularizer=self.kr,
                             bias_regularizer=self.br, activity_regularizer=self.ar)(cnn)
            cnn = BatchNormalization()(cnn)
            cnn = Activation('relu')(cnn)
            if k < len(nodes) - 1:
                cnn = MaxPooling1D(self.MP_factor)(cnn)

        cnn = GlobalAveragePooling1D()(cnn)
        return visible1, cnn

    def MortalityModel(self):
        """
        For ECG.
        This model architecture is inspired from the one described in the article "Prediction of mortality from
        12-lead electrocardiogram voltage data using a deep neural network "
        """
        # PARAMETERS
        # nodes of Conv1D Layers
        cnns_nodes = [16, 32, 64, 128, 256, 512]
        # nodes of Dense Layer for tabular data
        tab_DL_nodes = 64
        # nodes of Dense Layers after merging cnns and dense layers
        DL_nodes = [256, 128, 64, 32, 8]
        # updating hp
        self.hp.n_layers = len(cnns_nodes)

        inputs = []
        cnns = []  # for time series
        tab_dense = []  # for tabular data

        # CONVOLUTION ON TIME SERIES
        for k, shape in enumerate(self.X_shapes):
            if self.sub_model_type == 'CellConv1Dv1':
                visible, cnn = self.CellConv1Dv1(shape, cnns_nodes)
            else:
                # self.sub_model_type == 'CellConv1Dv2'
                visible, cnn = self.CellConv1Dv2(shape, cnns_nodes)
            inputs.append(visible)
            cnns.append(cnn)

        # TABULAR DATA
        visible = Input(shape=self.Xtab_shapes[0])
        tab_dense = Dense(tab_DL_nodes)(visible)

        inputs.append(visible)

        # MERGING
        merge = concatenate([tab_dense] + cnns)

        # FINAL DENSE LAYERS
        for k, node in enumerate(DL_nodes):
            if k == 0:
                dense = Dense(DL_nodes[k], kernel_regularizer=self.kr,
                              bias_regularizer=self.br,
                              activity_regularizer=self.ar)(merge)
            else:
                dense = Dense(DL_nodes[k], kernel_regularizer=self.kr,
                              bias_regularizer=self.br,
                              activity_regularizer=self.ar)(dense)
            dense = BatchNormalization()(dense)
            dense = Activation('relu')(dense)
            if k < 2:
                dense = Dropout(self.dropout)(dense)

        dense = Dense(1)(dense)

        self.model = Model(inputs=inputs, outputs=dense)

    def DiagnosisModel(self):
        """
        For ECG.
        Model from the article "Automatic diagnosis of the 12-lead ECG using a deep neural network "
        """
        model_path = os.path.join(self.pm_path, 'model_0/model.hdf5')
        model = load_model(model_path, compile=False)
        model.layers.pop()
        output = Dense(1)(model.layers[-1].output)
        # define new model
        self.model = Model(inputs=model.inputs, outputs=output)
        if self.sub_model_type == 'FullRetraining':
            pass
        else:
            for layer in model.layers[:-1]:
                layer.trainable = False

    def PATSConv1DModel(self):
        """
        For Physical Activity 1D.
        Model inspired from the article "Extracting biological age from biomedical data via deep learning: too much
        of a good thing? "
        PATS: Physical Activity Time Series
        Conv1D: 1D convolution on 1D data.
        """
        input_shape = self.X_shapes[0]
        self.model = Sequential()

        # convolutional part
        conv_sizes = [128, 32, 8, 8]
        conv_filters = [64, 32, 32, 32]
        MP_size = [4, 4, 3, 3]
        MP_strides = [2, 2, 2, 2]

        # updating hp
        self.hp.n_layers = len(conv_sizes)

        # fully connected part
        dense_filters = [256, 128]
        # dropout_rates = [0.5, 0.5]

        # model building
        self.model.add(BatchNormalization(input_shape=input_shape))
        for k in range(len(conv_sizes)):
            self.model.add(Conv1D(conv_filters[k], conv_sizes[k], padding='same', kernel_regularizer=self.kr,
                                  bias_regularizer=self.br, activity_regularizer=self.ar))
            self.model.add(Activation('relu'))
            self.model.add(MaxPooling1D(pool_size=MP_size[k], strides=MP_strides[k]))

        self.model.add(Flatten())

        for k in range(len(dense_filters)):
            self.model.add(Dense(dense_filters[k]))
            self.model.add(Activation('relu'))
            self.model.add(Dropout(self.dropout))  # dropout_rates[k]

        self.model.add(Dense(1))

    def PATSConvLSTM2D(self):
        """
        For physical activity. Model inspired from the article "Deep Learning using convolutional LStM estimates
        Biological Age from physical Activity".
        """
        self.model_name_for_predictions = 'LSTM2D'

        # (7, 24, 60, 1)
        input_shape = self.X_shapes[0]
        # dropout = 0.3
        dense_filters = [256, 128]

        model = Sequential()
        model.add(ConvLSTM2D(input_shape=input_shape, filters=128, kernel_size=(3, 3),
                             padding="valid", activation="relu"))
        model.add(Flatten())

        for df in dense_filters:
            model.add(Dense(df))
            model.add(Dropout(self.dropout))
        model.add(Dense(1))

        self.model = model

    def compute_model(self, verbose=True):
        """
        Compute model.
        """
        self.models[self.model_type]()

        if verbose:
            self.model.summary()

    def get_model(self, verbose=True):
        """
        Return the created model
        :param verbose: if True, model's summary is printed
        :return: created model (keras.model)
        """
        return self.model


class TS_model(TS_model_architecture):
    """
    Build model architecture.
    Enable to compile it, fit it to the data, and make predictions.
    """

    def __init__(self, hp, labeled_data, verbose=True, prediction=False, visualization=False):
        """
        :param hp: dictionary, hyperparameters
        :param labeled_data: dictionary where data are stored: 'X' (list of arrays), 'Xtab' (array),
        'y' (array), 'eid' (array), 'instance' (array)
        :param prediction: bool, if True, model created to generate predictions (and not for training).
        :param visualization: bool, if True, best model for each fold is loaded, in order to plot attention maps.
        """
        self.hp = hp
        self.version = self.hp.get_version()
        self.labeled_data = labeled_data
        self.verbose = verbose

        # train/val/test SPLIT PARAMETERS
        self.df1 = pd.DataFrame({
            'eid': self.labeled_data['eid'],
            'instance': self.labeled_data['instance'],
            'real_index': np.arange(len(self.labeled_data['eid']))
        })

        # dataframe linking each eid to a specific fold
        self.eids_to_folds = pd.read_csv('/n/groups/patel/Alan/Aging/Medical_Images/eids_split/All_eids.csv',
                                         dtype={'eid': np.int32, 'fold': np.int32})

        # UPDATING DF1 with folds
        self.df1 = self.df1.merge(self.eids_to_folds, how='left', on=['eid'])

        # number of folds
        self.n_splits = len(self.eids_to_folds.fold.unique())  # 10

        # partition: dictionary, linking labels ('train', 'test', 'val') to corresponding arrays of indices.
        self.partition = self.fetch_partition(self.hp.fold)

        # NORMALIZATION OF TARGETS (if necessary)
        self.targets_mean = None
        self.targets_std = None
        self.original_targets = self.labeled_data['y']
        self.targets_normalization()

        # paths
        self.callbacks_path = '../data/callbacks/'
        self.predictions_path = '../../Medical_Images/data_predictions_timeseries'
        # backup for predictions, especially for those before grouping by id and outer_fold
        self.backup_predictions_path = '../data/predictions'

        # Generate a small amount of preprocessed data to build model
        DG = DataGenerator(hp.model_type, hp.sub_model_type, labeled_data, self.partition['train'], batch_size=5,
                           shuffle=False)
        self.data_subset, self.labels_subset = DG[0]

        if not prediction and not visualization:
            # MODEL BUILDING
            self.build_model()

    def build_model(self):
        # Generation of the model's architecture
        super().__init__(self.hp, self.data_subset)

        # Model computation
        self.compute_model(verbose=self.verbose)

    def fetch_partition(self, fold):
        """
        Return partition of real_index, using folds from All_eids.csv.
        :param fold: int, index of the test set
        :return : dictionary of int arrays
            - 'train': numpy array of indexes for training set,
            - 'val': numpy array of indexes for validation set
            - 'test': numpy array of indexes for testing set
        """
        partition = dict()

        partition['test'] = self.df1[self.df1.fold == fold]['real_index'].to_numpy()
        partition['val'] = self.df1[self.df1.fold == (fold - 1) % self.n_splits]['real_index'].to_numpy()
        partition['train'] = self.df1[(self.df1.fold != fold) &
                                      (self.df1.fold != (fold - 1) % self.n_splits)]['real_index'].to_numpy()
        return partition

    def targets_normalization(self):
        """
        Normalize targets this way: targets <- (targets - mean(targets))/std(targets).
        Only use data in the train dataset to compute mean and std.
        """
        if self.hp.target == 'Age':
            # RETRIEVING ORIGINAL TARGETS
            self.labeled_data['y'] = self.original_targets
            # UPDATING targets_mean AND targets_std WITH ## TRAINING DATA ##
            self.targets_mean = np.mean(self.labeled_data['y'][self.partition['train']])
            self.targets_std = np.std(self.labeled_data['y'][self.partition['train']])
            # NORMALIZATION OF TARGETS
            self.labeled_data['y'] = (self.labeled_data['y'] - self.targets_mean) / self.targets_std

    def targets_rescaling(self, ar):
        """
        Rescale an array, using the following parameters:
        - self.targets_mean: mean of targets used during model training
        - self.targets_std: standard deviation of targets used during model training.
        """
        if self.hp.target == 'Age':
            if self.targets_mean is not None:
                return self.targets_mean + self.targets_std * ar
        return ar

    def compile(self, optimizer=Adam, loss='mean_squared_error', metrics=None):
        """
        Compile model with the given optimizer, loss and metrics.
        """
        if metrics is None:
            metrics = [R_squared, root_mean_squared_error]
        self.model.compile(optimizer=optimizer(self.hp.learning_rate), loss=loss,
                           metrics=metrics)

    def fit(self, verbose=2, callbacks_metric='R_squared'):
        """
        Fit the model to the data.
        :param callbacks_metric: string, name of reference metric for the callbacks
        :param verbose: int, define what is displayed during the training. See keras documentation.
        :return:
        """
        # Checkpoint
        if self.hp.overfitting == 0:
            # no overfitting
            callbacks_list = define_callbacks(self.callbacks_path, self.version, 'val_' + callbacks_metric, 40)
        else:
            # overfitting
            callbacks_list = define_callbacks(self.callbacks_path, self.version, callbacks_metric, 20)

        # Generators
        training_generator = DataGenerator(self.model_type, self.sub_model_type, self.labeled_data,
                                           self.partition['train'], batch_size=self.hp.batch_size, shuffle=True)
        validation_generator = DataGenerator(self.model_type, self.sub_model_type, self.labeled_data,
                                             self.partition['val'], batch_size=self.hp.batch_size, shuffle=True)

        return self.model.fit_generator(generator=training_generator, epochs=self.hp.n_epochs,
                                        validation_data=validation_generator, callbacks=callbacks_list, verbose=verbose,
                                        use_multiprocessing=False, workers=1)

    def get_best_model(self, target, TS_type, fold='\d', model='[a-zA-Z0-9]*', submodel='[a-zA-Z0-9]*'):
        """
        Fetch best model name for the given parameters (and especially the fold).
        """
        logger_pattern = 'logger_{target}_{TS_type}_{model}_{submodel}_\d_{fold}.*\.csv'.format(target=target,
                                                                                                TS_type=TS_type,
                                                                                                model=model,
                                                                                                submodel=submodel,
                                                                                                fold=fold)
        files = [file for file in os.listdir(self.callbacks_path) if re.search(logger_pattern, file)]
        max_R2_val = -np.inf
        best_model = None
        for file in files:
            try:
                df = pd.read_csv(os.path.join(self.callbacks_path, file))
                # print(file, df['epoch'].max(), df['R_squared'].max(), df['val_R_squared'].max())
                if df['val_R_squared'].max() > max_R2_val:
                    best_model = file.replace('logger', 'model-weights').replace('.csv', '.h5')
                    max_R2_val = df['val_R_squared'].max()
            except:
                pass
        return best_model

    def load_weights(self):
        """
        Loading model's weights.
        """
        self.model.load_weights(os.path.join(self.callbacks_path, 'model-weights_' + self.version + '.h5'))

    def predict(self, sample):
        return self.targets_rescaling(self.model.predict(sample))

    def predict_on_dataset(self, dataset_name):
        """
        Compute predictions for a specific dataset ('train', 'val' or 'train')
        :param dataset_name: str, 'val' or 'train'
        :return : DataFrame, with columns [eid, fold, prediction]
        """
        # DATA PROCESSING
        DP = DataPreprocessor(self.hp.model_type, self.hp.sub_model_type,
                              [x[self.partition[dataset_name]] for x in self.labeled_data['X']],
                              self.labeled_data['Xtab'][self.partition[dataset_name]])
        processed_data = DP.get_data()

        # MAKING PREDICTIONS
        predictions = self.targets_rescaling(self.model.predict(processed_data).flatten())

        # GENERATING DATAFRAME
        def eid_instance_to_ID(x):
            """
            Convert iterable (eid, instance) into a string 'eid_instance' (ID)
            :param x: iterable of length 2
            :return: string
            """
            if int(x[1]) == x[1]:
                # x[1] (instance) is an int
                return str(int(x[0])) + '_' + str(int(x[1]))
            else:
                # x[1] (instance) is a float
                return str(int(x[0])) + '_' + str(float(x[1]))

        ids = np.fromiter(map(eid_instance_to_ID, np.stack([self.labeled_data['eid'][self.partition[dataset_name]],
                                                            self.labeled_data['instance'][
                                                                self.partition[dataset_name]]],
                                                           axis=1)), dtype='<U128')

        df = pd.DataFrame({
            'id': ids,
            'outer_fold': (self.hp.fold - 1) % self.n_splits * np.ones(self.partition[dataset_name].size),
            'pred': predictions
        })

        return df

    def predict_on_dataset_with_DataGenerator(self, dataset_name):
        """
        Compute predictions for a specific dataset ('train', 'val' or 'train')
        :param dataset_name: str, 'val' or 'train'
        :return : DataFrame, with columns [eid, fold, prediction]
        """
        # DATA PROCESSING
        prediction_generator = DataGeneratorForPredictions(self.model_type, self.sub_model_type, self.labeled_data,
                                                           self.partition[dataset_name], batch_size=self.hp.batch_size)

        # MAKING PREDICTIONS
        predictions = self.targets_rescaling(self.model.predict_generator(prediction_generator).flatten())

        # GENERATING DATAFRAME
        def eid_instance_to_ID(x):
            """
            Convert iterable (eid, instance) into a string 'eid_instance' (ID)
            :param x: iterable of length 2
            :return: string
            """
            if int(x[1]) == x[1]:
                # x[1] (instance) is an int
                return str(int(x[0])) + '_' + str(int(x[1]))
            else:
                # x[1] (instance) is a float
                return str(int(x[0])) + '_' + str(float(x[1]))

        ids = np.fromiter(map(eid_instance_to_ID, np.stack([self.labeled_data['eid'][self.partition[dataset_name]],
                                                            self.labeled_data['instance'][
                                                                self.partition[dataset_name]]],
                                                           axis=1)), dtype='<U128')

        df = pd.DataFrame({
            'id': ids,
            'outer_fold': (self.hp.fold - 1) % self.n_splits * np.ones(self.partition[dataset_name].size),
            'pred': predictions
        })

        return df

    def outer_CV(self):
        """
        Outer cross-validation: make predictions on the whole dataset.
        Generate predictions for validation datasets and test datasets.
        """
        # dataframes of predictions for train datasets
        train_frames = []
        # dataframes of predictions for validation datasets
        val_frames = []
        # dataframes of predictions for test datasets
        test_frames = []

        # COMPUTING PREDICTIONS
        for k in range(self.n_splits):
            self.hp.fold = k
            best_model = self.get_best_model(self.hp.target, self.hp.TS_type, str(self.hp.fold))
            print(k, best_model)
            self.hp.update_from_model_name(best_model)
            self.version = self.hp.get_version()
            self.partition = self.fetch_partition(self.hp.fold)
            self.targets_normalization()
            self.build_model()
            self.load_weights()
            train_frames.append(self.predict_on_dataset_with_DataGenerator('train'))
            val_frames.append(self.predict_on_dataset_with_DataGenerator('val'))
            test_frames.append(self.predict_on_dataset_with_DataGenerator('test'))

            print("FOLD N°{} DONE".format(k))

        # CONCATENATING RESULTS
        train_df = pd.concat(train_frames)
        val_df = pd.concat(val_frames)
        test_df = pd.concat(test_frames)

        predictions_name = "Predictions_instances_{}_".format(self.hp.target)
        name_suffix = "{}_{}_0_{}_{}_0_0_0".format(self.model_name_for_predictions,
                                                   self.hp.n_layers,
                                                   'Adam',
                                                   self.hp.learning_rate)
        if self.hp.TS_type == 'AS':
            predictions_name += "Arterial_PulseWaveAnalysis_TimeSeries_" + name_suffix
        elif 'ECG' in self.hp.TS_type:
            predictions_name += "Heart_ECG_TimeSeries_" + name_suffix
        elif 'PhysicalActivity' in self.hp.TS_type:
            if '90004' in self.hp.TS_type:
                predictions_name += "PhysicalActivity_FullWeek_Acceleration_" + name_suffix

            elif self.hp.TS_type == 'PhysicalActivity-90001-TimeSeries-epochs-5min':
                predictions_name += "PhysicalActivity_FullWeek_TimeSeriesFeatures_" + name_suffix

            elif self.hp.TS_type == 'PhysicalActivity-90001-TimeSeries-epochs-5min-model2bis':
                predictions_name += "PhysicalActivity_FullWeek_TimeSeriesFeaturesAndScalars_" + name_suffix

            elif 'PhysicalActivity-90001-TimeSeries-acceleration-3D-walking' in self.hp.TS_type:
                predictions_name += "PhysicalActivity_Walking_3D_" + name_suffix
            else:
                raise Exception("No TimeSeries type found.")
        else:
            raise Exception("No TimeSeries type found.")

        # SAVING RAW RESULTS (before any grouping)
        train_df.to_csv(os.path.join(self.backup_predictions_path, predictions_name + '_train.csv'),
                        index=False)
        val_df.to_csv(os.path.join(self.backup_predictions_path, predictions_name + '_val.csv'),
                      index=False)
        test_df.to_csv(os.path.join(self.backup_predictions_path, predictions_name + '_test.csv'),
                       index=False)

        # SAVING RESULTS (after grouping by id and outer_fold)
        train_df.groupby(['id', 'outer_fold']).mean().reset_index().to_csv(
            os.path.join(self.predictions_path, predictions_name + '_train.csv'),
            index=False)
        val_df.groupby(['id', 'outer_fold']).mean().reset_index().to_csv(
            os.path.join(self.predictions_path, predictions_name + '_val.csv'),
            index=False)
        test_df.groupby(['id', 'outer_fold']).mean().reset_index().to_csv(
            os.path.join(self.predictions_path, predictions_name + '_test.csv'),
            index=False)

    def load_model_for_visualization(self):
        """
        Load best model for a given fold.
        """
        print("LOADING MODEL FOR VISUALIZATION")
        best_model = self.get_best_model(self.hp.target, self.hp.TS_type, str(self.hp.fold))
        print(best_model)
        self.hp.update_from_model_name(best_model)
        self.version = self.hp.get_version()
        self.build_model()
        self.load_weights()

    def get_preprocessed_sample(self, dataset_name, index=0, select_random=False):
        """
        Return a preprocessed input sample from selected dataset, and the corresponding target.
        :param dataset_name: string ('train', 'test' or 'val')
        :param index: int, index in the dataset. By default, the first element is taken.
        :param select_random: bool, if True, selected index is random.
        """
        index = min(index, len(self.partition[dataset_name]) - 1)
        if select_random:
            index = np.random.randint(0, len(self.partition[dataset_name]))
        real_index = self.partition[dataset_name][index]
        DP = DataPreprocessor(self.hp.model_type, self.hp.sub_model_type,
                              [np.expand_dims(x[real_index], axis=0) for x in self.labeled_data['X']],
                              np.expand_dims(self.labeled_data['Xtab'][real_index], axis=0))
        return DP.get_data(), self.labeled_data['y'][real_index]

    def get_selected_preprocessed_sample(self, _id):
        """
        Return a preprocessed input sample for a given id.
        :param _id: string, 'eid_instance'
        """
        eid, instance = _id.split('_')
        eid = int(eid)
        instance = float(instance)
        real_index = int(self.df1[(self.df1.eid == eid) & (self.df1.instance == instance)].iloc[0].real_index)
        DP = DataPreprocessor(self.hp.model_type, self.hp.sub_model_type,
                              [np.expand_dims(x[real_index], axis=0) for x in self.labeled_data['X']],
                              np.expand_dims(self.labeled_data['Xtab'][real_index], axis=0))
        return DP.get_data(), self.labeled_data['y'][real_index]

    def get_selected_raw_sample(self, _id):
        """
        Return a raw input sample for a given id.
        :param _id: string, 'eid_instance'
        """
        eid, instance = _id.split('_')
        eid = int(eid)
        instance = float(instance)
        real_index = int(self.df1[(self.df1.eid == eid) & (self.df1.instance == instance)].iloc[0].real_index)
        return self.labeled_data['X'][0][real_index].squeeze()


class KeractBasedVisualization:
    """
    Generate saliency maps using keract library.
    """
    def __init__(self, model, x, y):
        """
        :param model: compiled keras.models.Model
        :param x: Numpy array to feed the model as input. In the case of multi-inputs, x should be of type List.
        :param y: Labels (numpy array). Keras convention.
        """
        self.model = model
        self.x = x
        if len(y.shape) == 0:
            self.y = np.expand_dims(y, axis=0)
        else:
            self.y = y
        self.cmap = 'bwr'
        # computed arrays using keract
        self.keract_arrays = None
        # full layer names we want to visualize
        self.layers = None

    def activations(self, layer_name='conv1d_1'):
        """
        Compute activations for chosen layer(s).
        If layer_name is input, all the inputs layer are selected.
        REMARK:
        - activations for input layers are the inputs themselves (useless).
        - if the time dimension of the layer output is not the same as the input, there will be an error during
        the plot
        :param layer_name: string
        """
        # computing activations
        self.keract_arrays = keract.get_activations(self.model, self.x)
        # GETTING FULL NAMES OF LAYERS
        if layer_name == 'input':
            self.layers = [key for key in list(self.keract_arrays.keys()) if re.search(layer_name, key)]
        else:
            self.layers = [layer_name]

    def gradients_of_activations(self, layer_name='input'):
        """
        Compute gradients of activations for chosen layer(s).
        If layer_name is input, all the inputs layer are selected.
        :param layer_name: string
        """
        # computing gradients of activations
        self.keract_arrays = keract.get_gradients_of_activations(self.model, self.x, self.y)

        # GETTING FULL NAMES OF LAYERS
        # GETTING FULL NAMES OF LAYERS
        if layer_name == 'input':
            self.layers = [key for key in list(self.keract_arrays.keys()) if re.search(layer_name, key)]
        else:
            self.layers = [layer_name]

        if self.layers == []:
            self.layers = [list(self.keract_arrays)[0]]

    def plot_gradients(self):
        """
        Plot computed gradients of activations on top of the inputs (for visualization purpose).
        """
        for k, name in enumerate(self.layers):
            if len(self.x[k].shape) == 3:
                # shape is like (1, n_time_steps, n_features)
                # several time series (e.g. several leads)
                # number of time_series
                sh = self.x[k].squeeze().shape
                if len(sh) == 1:
                    n_ts = 1
                else:
                    n_ts = self.x[k].squeeze().shape[1]
            else:
                n_ts = 1
            for j in range(n_ts):
                if n_ts == 1:
                    x_ = self.x[k].flatten()
                    keract_array = self.keract_arrays[name].flatten()
                else:
                    x_ = self.x[k].squeeze()[:, j]
                    keract_array = self.keract_arrays[name].squeeze()[:, j]

                if x_.size > 50:
                    # time series
                    # we don't plot tabular data
                    plt.figure(figsize=(30, 10))
                    plt.title('')
                    plt.scatter(np.arange(x_.size), x_, c=keract_array, cmap=self.cmap)
                    plt.colorbar()
        plt.show()

    def get_gradients(self):
        """
        Return inputs data with computed gradients of activations (for visualization purpose).
        """
        output_arrays = []
        for k, name in enumerate(self.layers):
            if len(self.x[k].shape) == 3:
                # shape is like (1, n_time_steps, n_features)
                # several time series (e.g. several leads)
                # number of time_series
                sh = self.x[k].squeeze().shape
                if len(sh) == 1:
                    n_ts = 1
                else:
                    n_ts = self.x[k].squeeze().shape[1]
            else:
                n_ts = 1
            for j in range(n_ts):
                if n_ts == 1:
                    x_ = self.x[k].flatten()
                    keract_array = self.keract_arrays[name].flatten()
                else:
                    x_ = self.x[k].squeeze()[:, j]
                    keract_array = self.keract_arrays[name].squeeze()[:, j]

                if x_.size > 50:
                    # time series
                    # we don't plot tabular data
                    output_arrays.append(np.vstack((x_, keract_array)))
        return output_arrays


def fetch_model(version, verbose=False):
    """
    Load data, build model and load pre-trained weights, using the information stored in version.
    :param version: string
    :param verbose: bool (when building model)
    :return: TS_Model
    """
    # FETCHING HYPER PARAMETERS
    hp = hyperparameters(version=version)
    version = hp.get_version()
    print(version)

    # LOADING DATA
    print('LOADING DATA')
    DF = DataFetcher(hp.target, hp.TS_type)
    labeled_data = DF.get_data()  # dictionary

    # BUILDING MODEL
    print('BUILDING MODEL')
    model = TS_model(hp, labeled_data, verbose, visualization=True)

    # LOADING WEIGHTS
    print('LOADING WEIGHTS')
    model.load_model_for_visualization()

    return model


def R_squared(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def define_callbacks(path_store, version, metric, ES_patience=20):
    csv_logger = CSVLogger(path_store + 'logger_' + version + '.csv', separator=',')
    model_checkpoint = ModelCheckpoint(path_store + 'model-weights_' + version + '.h5', monitor=metric, mode='max',
                                       verbose=1, save_best_only=True, save_weights_only=False)
    reduce_lr_on_plateau = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1, mode='min', min_delta=0,
                                             cooldown=0, min_lr=0)
    if 'R_squared' in metric:
        early_stopping = EarlyStopping(monitor=metric, min_delta=0, patience=ES_patience, verbose=0, mode='max')
    else:
        early_stopping = EarlyStopping(monitor=metric, min_delta=0, patience=ES_patience, verbose=0, mode='auto')
    return [csv_logger, model_checkpoint, reduce_lr_on_plateau, early_stopping]


def convert_time(seconds):
    """
    Convert seconds to hours, minutes and seconds
    :param seconds: float
    :return: string
    """

    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)
