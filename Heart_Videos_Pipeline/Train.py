import os
import sys

sys.path.append('../Helpers/')
from DataFunctions import DataGenerator, get_data_fold
from EvaluatingFunctions import R2_, RMSE
from EvaluatingModels import R2Callback, resume_training, save_print_lr

import warnings
warnings.filterwarnings('ignore')

# Tensorflow/Keras set up
import tensorflow as tf
cpu_count = len(os.sched_getaffinity(0))
config = tf.ConfigProto(intra_op_parallelism_threads=cpu_count,
                        inter_op_parallelism_threads=cpu_count, 
                        allow_soft_placement=True, 
                        device_count = {'CPU': cpu_count})
config.gpu_options.allow_growth = True
sess= tf.Session(config = config)

from keras.models import Sequential, load_model
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K
K.set_session(session= sess)
K.tensorflow_backend._get_available_gpus()


# ------------------------------------------------------------------------------------------------------------------------

def main():
    
    k_fold = int(sys.argv[1])
    channel = f'{int(sys.argv[2])}ch'
             
    csv_dir = '/n/groups/patel/JbProst/Heart/Data/FoldsAugmented/data-features_Heart_20208_Augmented_Age_{}_{}.csv'
    partition, labels = get_data_fold(k_fold, csv_dir, target = 'Age_raw')

    input_shape = (25, 150, 150, 1)
    directory = '/n/scratch3/users/j/jp379/shape'+str(input_shape[0]) +'x'+str(input_shape[1])+'x'+str(input_shape[1])+'-' + channel +'_Augmented/'
    
    batch_size = 8
    
    prepro = {'normalize' : False, 'flip' : False, 'rotate' : 3, 'shift': False}
    params_train = {'data_directory' : directory,
                    'ids' : partition['train'],
                    'labels' : labels['train'],
                    'balanced' : None,
                    'max_samples': None,
                    'dim' : input_shape,
                    'pre_processing_dict' : prepro,
                    'batch_size':  batch_size,
                    'shuffle' : True}
    train_generator = DataGenerator(**params_train)

    prepro = {'normalize' : False, 'flip' : False, 'rotate' : False, 'shift': False}
    params_val = {'data_directory' : directory,
                    'ids' : partition['val'],
                    'labels' : labels['val'],
                    'balanced' : False,
                    'max_samples' : None,
                    'dim' : input_shape,
                    'pre_processing_dict' : prepro,
                    'batch_size': batch_size,
                    'shuffle': False}

    val_generator = DataGenerator(**params_val)

    # ------------------------------------------------------------------------------
    #Model
    drp = 0.2
    lr = 0.8e-4
    reg = 0
    nb_nodes = 1024
    
    model_Conv3D = Sequential()

    model_Conv3D.add(Conv3D(16, (3,3,3), strides=(1, 1, 1), activation='selu',
                            input_shape = input_shape, padding ='same', kernel_regularizer= regularizers.l2(reg)))
    model_Conv3D.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))

    model_Conv3D.add(Conv3D(64, (3,3,3), activation='selu', padding= 'same', kernel_regularizer= regularizers.l2(reg)))
    model_Conv3D.add(MaxPooling3D(pool_size=(1, 5, 5), strides=(1, 2, 2)))

    model_Conv3D.add(Conv3D(int(nb_nodes/2), (5,5,5), activation='selu', kernel_regularizer= regularizers.l2(reg)))
    model_Conv3D.add(MaxPooling3D(pool_size=(5, 5, 5), strides=(1, 2, 2)))

    model_Conv3D.add(Conv3D(nb_nodes, (5,7,7), activation='selu', kernel_regularizer= regularizers.l2(reg)))
    model_Conv3D.add(MaxPooling3D(pool_size=(1, 5, 5), strides=(1, 2, 2)))

    model_Conv3D.add(Flatten())
    model_Conv3D.add(BatchNormalization())
    model_Conv3D.add(Dense(1024, kernel_regularizer= regularizers.l2(reg), activation = 'selu'))

    model_Conv3D.add(Dropout(drp))
    model_Conv3D.add(Dense(1, activation='linear'))
    # ------------------------------------------------------------------------------
    model_Conv3D, lr = resume_training(model_Conv3D, lr, best=False)
    adam = Adam(lr = lr, clipnorm = 1.)
    
    model_Conv3D.compile(optimizer =adam, loss=RMSE,
                         metrics=[R2_])
    model_Conv3D.summary()
    # ------------------------------------------------------------------------------

    lr_reducing= ReduceLROnPlateau(monitor= 'loss', factor=0.8, patience=2, mode='min', verbose=1)
    R2_cb = R2Callback(val_generator, patience=100, restore_best_weights=False, restore=True,
                      input_dir = '/n/groups/patel/JbProst/Heart/Scripts/CrossVal/Age/{}/Fold{}/'.format(channel, k_fold),
                      output_dir = '/n/groups/patel/JbProst/Heart/Scripts/CrossVal/Age/{}/Fold{}/'.format(channel, k_fold))
    
    display_lr = LearningRateScheduler(save_print_lr, verbose=1)

    model_Conv3D.fit_generator(generator=train_generator,  verbose=2,
                               use_multiprocessing=True, workers = cpu_count, 
                               epochs = 11,
                               callbacks = [lr_reducing, R2_cb, display_lr])

if __name__ == "__main__":
    main()
    sess.close()
    sys.exit(1)