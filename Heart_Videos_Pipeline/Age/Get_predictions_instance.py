import os
import pickle

import re
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import multiprocessing as mp
from pathlib import Path

import sys
sys.path.append('/n/groups/patel/JbProst/Heart/')
from DataFunctions import *
from EvaluatingFunctions import *
from PlottingFunctions import *
from EvaluatingModels import R2Callback, resume_training, save_print_lr
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tqdm import tqdm

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

cpu_count = len(os.sched_getaffinity(0))
config = tf.ConfigProto(intra_op_parallelism_threads=cpu_count,
                        inter_op_parallelism_threads=cpu_count, 
                        allow_soft_placement=True, 
                        device_count = {'CPU': cpu_count})
config.gpu_options.allow_growth = True
sess= tf.Session(config = config)
print('{} CPUs'.format(cpu_count))
print('tensorflow version : ', tf.__version__)
print('Build with Cuda : ', tf.test.is_built_with_cuda())
print('Gpu available : ', tf.test.is_gpu_available())
print('Available ressources : ', tf.config.experimental.list_physical_devices())
from keras.layers import LSTM, Flatten, Dense, Input, Reshape, BatchNormalization, InputLayer, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv3D, MaxPooling3D
from keras.models import Sequential, load_model
from keras import regularizers
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import multi_gpu_model
from keras import initializers
K.set_session(session= sess)
K.tensorflow_backend._get_current_tf_device()
K.tensorflow_backend._get_available_gpus()


def test_predictions(test_generator, directory_, label, epoch):
    cpu_count = len(os.sched_getaffinity(0))

    if epoch:
        weights_file = directory_ + 'epoch_{:04.0f}.h5'.format(int(epoch))
    else:
        weights_files = [directory_ + w for w in os.listdir(directory_) if w.endswith('.h5')]
        weights_file = max(weights_files, key=os.path.getctime)
        
    print('Loading ', weights_file)
    model = Sequential()
    model= load_model(weights_file, {'RMSE' : RMSE, 'R2_': R2_})
    
    pred_te= model.predict_generator(test_generator, verbose =1,
                                     use_multiprocessing =True, 
                                     workers= cpu_count)
    
    del model
    return pred_te

def get_predictions(epoch, directory, kfold, fold, instance):
    channel = [c for c in directory.split('/') if c.endswith('ch')][0]
    #get positive labels    
    input_shape = (25, 150, 150, 1)
    data_directory = '/n/scratch3/users/j/jp379/shape'+str(input_shape[0]) +\
    'x'+str(input_shape[1])+'x'+str(input_shape[1])+'-' + channel +'_{}/'.format(instance)

    csv_dir = '/n/groups/patel/JbProst/Heart/Data/Folds/data-features_Heart_20208_{}_Age_{}_{}.csv'.format(instance,
                                                                                                                   fold,
                                                                                                                  kfold)
    partition, labels = get_data_fold(kfold, csv_dir, target = 'Age_raw')
    outputs = []

    params_test = {'data_directory' : data_directory,
            'ids' : partition[fold],
            'labels' : labels[fold],
            'dim' : input_shape,
            'balanced' : False,
            'batch_size': 1}
    test_generator = TestGenerator(**params_test)
    
    predictions = pd.DataFrame(labels[fold])
    predictions['Pred'] = test_predictions(test_generator, directory, fold, epoch)
    predictions.to_csv(directory +'/'+fold +'predictions_{}.csv'.format(instance))
    
    
def main():
    print(sys.argv)
    channel = sys.argv[1]
    kfold =sys.argv[2]
    part = sys.argv[3]
    instance = sys.argv[4]

    dir_ = '/n/groups/patel/JbProst/Heart/Scripts/CrossVal/Age/{}ch/Fold{}/'.format(channel, kfold)
    df_scores, _  = retrieve_score(dir_)
    dir_ = '/n/groups/patel/JbProst/Heart/Scripts/CrossVal/Age/{}ch/Fold{}/'.format(channel, kfold)
    weights_files = [int((dir_ + w).split('_')[-1].split('.')[0]) for w in os.listdir(dir_) if w.endswith('.h5')]
    epoch = int(df_scores['R2_val'].loc[weights_files].argmax())
    
    if os.path.exists(dir_ +part +'predictions_{}.csv'.format(instance)):
        print(dir_ +'/'+part +'predictions_{}.csv'.format(instance))
    else: 
        print('Get predictions')
        get_predictions(epoch, dir_, kfold, part, instance)

if __name__ == '__main__':
    main()
    sess.close()
    sys.exit(1)