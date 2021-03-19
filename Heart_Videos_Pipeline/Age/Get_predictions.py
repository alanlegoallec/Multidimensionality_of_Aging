import os
import pickle
import multiprocessing
import GPUtil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
from pathlib import Path

import sys
sys.path.append('/n/groups/patel/JbProst/Heart/')
from DataFunctions import *
from EvaluatingFunctions import *
from PlottingFunctions import *

from EvaluatingModels import F1Callback, resume_training, save_print_lr

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tqdm import tqdm_notebook as tqdm

import warnings
warnings.filterwarnings('ignore')

cpu_count = len(os.sched_getaffinity(0))
config = tf.ConfigProto(intra_op_parallelism_threads=cpu_count,
                        inter_op_parallelism_threads=cpu_count, 
                        allow_soft_placement=True, 
                        device_count = {'CPU': cpu_count})
config.gpu_options.allow_growth = True
#config.graph_options.optimizer_options.global_jit_level = jit_level
sess= tf.Session(config = config)

from keras.layers import LSTM, Flatten, Dense, Input, Reshape, BatchNormalization, InputLayer, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv3D, MaxPooling3D
from keras.models import Sequential, load_model
from keras import regularizers
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras import backend as K
from keras.utils import multi_gpu_model
from keras import initializers
K.set_session(session= sess)
K.tensorflow_backend._get_available_gpus()

from sklearn.metrics import r2_score


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

def get_predictions(epoch, directory, kfold, fold, csv_dir = '/n/groups/patel/JbProst/Heart/Data/FoldsAugmented/data-features_Heart_20208_Augmented_Age_{}_{}.csv' ):
    channel = [c for c in directory.split('/') if c.endswith('ch')][0]
    #get positive labels    
    input_shape = (25, 150, 150, 1)
    data_directory ='/n/scratch3/users/j/jp379/shape'+str(input_shape[0]) +'x'+str(input_shape[1])+'x'+str(input_shape[1])+'-' + channel +'_Augmented/'
    
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
    return predictions 
    
        

def main():
             
    channel = int(sys.argv[1])
    kfold = int(sys.argv[2])
    part = sys.argv[3]
    print(sys.argv[4])
    epoch = int(sys.argv[4])
    
    dir_ = '/n/groups/patel/JbProst/Heart/Scripts/CrossVal/Age/{}ch/'.format(channel)
    df_scores, _  = retrieve_score(dir_)
    
    dir_ = '/n/groups/patel/JbProst/Heart/Scripts/CrossVal/Age/{}ch/Fold{}/'.format(channel, kfold)
    if not epoch:
        weights_files = [int((dir_ + w).split('_')[-1].split('.')[0])-1 for w in os.listdir(dir_) if w.endswith('.h5')]
        epoch = int(df_scores['R2_val','Fold{}'.format(kfold)].loc[weights_files].argmax() +1)
        
    preds = get_predictions(epoch, dir_, kfold, part)
    if int(sys.argv[4]) :
        preds.to_csv(dir_ +'/{}predictions_augmented_{}.csv'.format(part,epoch))
    else:
        preds.to_csv(dir_ +'/{}predictions_augmented.csv'.format(part))

if __name__ == "__main__":
    main()
    sess.close()
    print('Closing')
    sys.exit(1)