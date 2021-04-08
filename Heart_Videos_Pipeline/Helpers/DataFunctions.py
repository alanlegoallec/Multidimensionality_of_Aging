import os
import pickle
import numpy as np
import pandas as pd
import warnings
import h5py

#import tensorflow as tf
import keras
from scipy.ndimage import shift, rotate
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def get_data_fold(k_fold, csv_dir, target):
    """
    Get the data previously split in k-folds for cross validation in .csv files 
    containing the patient ids ('eid'), their age and their sex.
    Each fold is decomposed into 3 set : one for each training, validation and testing.
    
    Inputs:
        - k_fold [int] :  the k-th fold to consider
        - csv_dir [str] : directory of the csv file. Must be of form: "..._{}_{}.csv" so that
         the correponding train/test/val and k-fold can be extracted.
         
    Outputs:
        - partition [dict] :  dict for train/test/val containing a list of 
        int patient ids 
        - labels [dict] : dict for train/test/val containing a pd.Series with patient 
        (int) ids as index and the corresponding value of interest (Age or Sex)
    """
    
    folds = ['train', 'test', 'val']
    partition = dict()
    labels = dict()

    for f in folds:
        data_path = csv_dir.format(f,k_fold)
        data = pd.read_csv(data_path, index_col='eid')
        if isinstance(data.index[0], str):
            data.index = data.index.map(lambda x: int(x[:-4]))
        else: 
            pass
        partition[f] = data.index.to_list()
        labels[f] = data[target]
        
    return partition, labels
    
### Data Generator Class    
    
class DataGenerator(keras.utils.Sequence):
    """
    Inherits form the Sequence class (https://keras.io/utils/)
    Custom data generator that avoid loading the entire dataset into the memory. 
    Manages all the data from retrieving to providing to the model.
    Applies transformation at the population scale (uniformization, class porportion),
    and at the sample scale (rotation, shifting, normalization) without any data 
    duplication.
    Process batches of data in parallele insInstead, yield batch of data to feed the model.
    
    The instance is call by an method of a Keras instance Sequential
    (https://keras.io/models/sequential/) 
    or Model (https://keras.io/models/model/).
    The metods are :  fit, evaluate, predict, fit/predict/evaluate _generator.
    """                                 
    
    def __init__(self, data_directory, ids, labels, dim,
                 pre_processing_dict= {'normalize' : False, 'rotate' : False, 'shift': False},
                 balanced =False,
                 max_samples= None , 
                 testing= False, 
                 batch_size=32,
                 shuffle=True, 
                 uniform = False,
                 autoencoder =False):
        """
        INPUTS:
            data_directory = directory of files location [str]
            ids = list of samples [list]
            labels = labels associated with samples [pd.Series]
            dim = dimension of a samples [tuple]
            pre_processing_dict =  transformation to be applied [dic]
            balanced = porportion of positive class [boolean or 0<float=<0.5 ],
            max_samples = max number of samples for shorter epochs [int]
            testing = activate/deactivate testing mode [boolean]
            batch_size = size of a batch [int]
            shuffle = shuffle data at the beginning of epoch [boolean]
            uniform = make the distribution of labels uniform [boolean]
        OUTPUTS:
            The __getitem__ method is called with the batch index as argument
        """
        
        self.data_directory = data_directory #where to access the samples
        # Get the ids that are in teh data base and in the partitioning
        ids_in_database = [int(file.split('.')[0]) for file in os.listdir(data_directory)]
        self.list_IDs = list(set(ids) & set(ids_in_database)) #list of samples
        self.labels = labels #corresponding labels
        self.batch_size = batch_size #size of the batch
        self.autoencoder= autoencoder
        
        if len(np.unique(labels))>3: # if regression task
            balanced = False #ensures that the data will not be balanced

        if max_samples: #determines a mx nb of samples
            self.max_samples = max_samples
        else: 
            self.max_samples = len(self.list_IDs)
        
        #varying porportion of class proportions
        if balanced:
            self.balanced = True 
            self._balance_data_(balanced) # transform proportion
            self.__getitem__ = self.__getitem__balanced_            
        else:
            self.balanced = False
            self._reduce_dataset_() #reduce tto max_sample size
            self.__getitem__ = self.__getitem__unbalanced_
            
        if uniform: #makes the dsitribution of labels uniform
            self.list_IDs = self._uniform_()
        
        # DImension of the data : changes the way files are managed
        self.dim = dim # dimension of a sample [n_sequence, height, width, chamber view]
        if self.dim[-1] < 3: #if the sample has only a single channel
            self.multi_channel = False
            self.__data_generation = self.__data_generation_single
        else :
            self.multi_channel = True
            self.__data_generation = self.__data_generation_multi
            
        self.shuffle = shuffle #if data is shuffle at every epoch 
        
        # Preprocessing
        self.pre_processing = pre_processing_dict
        self.normalize = False
        self.shift = False
        self.rotate = False
        self._which_preprocessing_(uniform)
        
        if testing: #testing mode
            self._testing_()
            self.__getitem__ = self.__getitem__unbalanced_
        
        # Allocate to memory once for all
        self.npy = '.npy'
        
        # Display summary of what manipulation have been done
        self._print_info_(testing, uniform)
        
        #initialize the learning process
        self.on_epoch_end()
        
    def _print_info_(self ,testing, uniform):
        """
        Prints the info/set-up of the Generator when the instance is created
        """
        dir_= self.data_directory.split('/')[-2]
        #Number of samples
        if len(np.unique(self.labels))>3: #regression
            info = '{} regression samples from {}.'.format(len(self.labels), dir_)
            if uniform:
                info = info + '({} samples)  Uniform distribution'.format(len(self.list_IDs))
                
        else: #binary classification
            if self.balanced :
                info = '{} positive samples and {} negative ones from {}'.format(len(self.list_IDs_pos),
                                                                                len(self.list_IDs_neg),
                                                                                dir_)
            else:
                info = '{} positive samples and {} negative ones from {}'.format(np.sum(self.labels),
                                                                                len(self.labels)-np.sum(self.labels),
                                                                                dir_)
        
        print(info)
        if testing:
            print('Testing Mode')
        
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        if self.balanced:
            return int(np.floor(len(self.list_IDs_pos) / self.batch_size))            
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))
            
    def get_labels(self):
        """
        Get teh labels
        """
        return self.labels
    
    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.balanced:
            self.neg_indexes = np.arange(len(self.list_IDs_neg))
            self.pos_indexes = np.arange(len(self.list_IDs_pos))
            if self.shuffle == True:
                np.random.shuffle(self.pos_indexes)
                np.random.shuffle(self.neg_indexes)
                
        else:
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)
            
    def get_class_weights(self):
        """
        Returns the proportion of classes
        """
        if self.balanced:
            
            return {1: len(self.neg_indexes)/len(self.pos_indexes),
                    0: len(self.pos_indexes)/len(self.neg_indexes)}
        else:
            class_w = class_weight.compute_class_weight('balanced', 
                                                 np.unique(self.labels),
                                                 self.labels)
            return {0: class_w[0], 1: class_w[1]}
    
    def _reduce_dataset_(self):
        """
        Shrinks the dataset to max_samples size.
        """        
        self.list_IDs = self.list_IDs[:self.max_samples]        
        self.labels = self.labels.iloc[:self.max_samples]
    
    def _uniform_(self):
        """
        Makes the distribution unifrom of labels value. 
        Used for regression only.
        """
        #count the max occurence of a label
        counts =  self.labels.value_counts()
        max_counts = counts.max()
        df_uniform = pd.Series()
        
        #for each count associated with each label
        for value, count in zip(counts.index, counts):
            sub_arr = self.labels[self.labels==value]
            ratio_sub = max_counts / len(sub_arr)

            # more than Twice, add the entire sub-set
            if ratio_sub > 2:
                for n in range(int(ratio_sub)-1):
                    sub_arr = sub_arr.append(self.labels[self.labels==value])

            # rest of the division, add until reaching the max_count
            diff_sub = max_counts - len(sub_arr)
            if diff_sub != 0:
                sub_arr = sub_arr.append(self.labels[self.labels==value].iloc[:diff_sub])
            df_uniform = pd.concat([df_uniform, sub_arr])
        
        return list(df_uniform.index)

    def _balance_data_(self, balanced):
        """
        Duplicates positive IDs such that positive samples will be augmented
        """

        self.list_IDs_neg = list(self.labels[self.labels==0].index)[:int((1-balanced)*self.max_samples)]
        
        self.list_IDs_pos = self.labels[self.labels==1].index
        
        if len(self.list_IDs_pos) < int(balanced * self.max_samples):
            #tiles the positive index to reach the same number of positive as the negative
            tile_nb = int((balanced*self.max_samples)/len(self.list_IDs_pos))
            
            if tile_nb <1:
                pass
            else :
                self.list_IDs_pos = list(np.tile(self.list_IDs_pos, tile_nb))
            # tile_nb can only be int, ad the remaining fraction to reach the desired length
            self.list_IDs_pos = self.list_IDs_pos + self.list_IDs_pos[:int((balanced*self.max_samples) - len(self.list_IDs_pos))]
        else:
            self.list_IDs_pos = self.list_IDs_pos[:int(balanced * self.max_samples)]
            
    def _testing_(self):
        """
        For testing mode, ensure that the following parameters are correctly set
        """
        self.suffle = False
        self.batch_size = 1
        self.unifrom = False
        self.normalize = False
        self.shift = False
        self.rotate = False
            
     ### --- Pre processing functions --- ###
    
    def _which_preprocessing_(self, uniform):
        """
        Which preprocessing should be done
        """
        if self.pre_processing['normalize']:
            self.normalize = True
            
        if self.pre_processing['shift']:
            self.shift = self.pre_processing['shift'] + 1
        
        if self.pre_processing['rotate']:
            self.rotate = self.pre_processing['rotate'] + 1
            
        if (self.balanced or uniform) and (not self.rotate and not self.shift):
            message = 'WARNING : the data distribution has been modified without sample modficiation, could cause overfitting.' 
            warnings.warn(message)
    
    def _normalize_(self, array):
        """
        Normalize each frame
        """
        return array / 255. # array.max(axis = 3).max(axis= (1,2))
                
    def _shift_(self, array):
        """
        Shift range
        """
        shift_pix = np.random.randint(0,self.shift)
        if self.multi_channel:
            shifted = shift(array, [0, 0, shift_pix])
        else:
            shifted = shift(array, [0, 0, shift_pix, 0])
        return shifted 
    
    def _rotate_(self, array):
        """ 
        Applies a random rotation of the image in the
        [ -'rotate_range' ; 'rotate_range' ] range to a stack
        """
        angle = np.random.randint(0, self.rotate)
        sign = np.random.randint(0,2)
        return rotate(array, -sign *angle, (1,2), reshape=False)
        
    def _preprocess_sample_(self, array):
        """
        Applies a random pre processing to a single sample
        """
        if self.normalize :
            array = self._normalize_(array)
        if self.shift:
            array = self._shift_(array)
        if self.rotate:
            array = self._rotate_(array) 
        return array
                
    ### --- Data reaching --- ###
    
    def __data_generation_single(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """ 
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            try:
                x = np.load(self.data_directory + str(ID) + self.npy)#[:,:,:,0]
                try:
                    #pre-processing    
                    X[i,] = self._preprocess_sample_(x)[..., np.newaxis]
                    # Store class
                    y[i] = self.labels[ID]
                except (ValueError,KeyError):
                    print('Shape/key-error {} sample'.format(ID))
                    
            except FileNotFoundError:
                print(ID, ' not found')
            
            

        return X, y 
    
    def __data_generation_multi(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # X : [n_samples, time, widht, height, n_channels]
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            try:
                
                try:
                    x = self._preprocess_sample_(np.load(self.data_directory + str(ID) + self.npy, allow_pickle=False))
                except ValueError:
                    x = self._preprocess_sample_(np.load(self.data_directory + str(ID) + self.npy, allow_pickle=True))
                X[i,] = x

                # Store class
                y[i] = self.labels[ID]
                
            except (ValueError, OSError):
                print(str(ID))
                pass

        return X, y 

    def __getitem__unbalanced_(self, index):
        """
        Generate a batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y    
                                 
    def __getitem__balanced_(self, index):
        """
        Generate a batch of balanced data
        """
        # Generate indexes of the batch
        pos_idx = self.pos_indexes[int(index*self.batch_size/2):int((index+1)*self.batch_size/2)]
        neg_idx = self.neg_indexes[int(index*self.batch_size/2):int((index+1)*self.batch_size/2)]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs_pos[k] for k in pos_idx]
        list_IDs_temp =  list_IDs_temp + [self.list_IDs_neg[k] for k in neg_idx]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y    
    
    def __getitem__(self, index):
        if self.balanced:
            return self.__getitem__balanced_(index)
        else:
            return self.__getitem__unbalanced_(index)
        
        if self.autoencoder:
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
            list_IDs_temp = [self.list_IDs[k] for k in indexes]
            X = np.empty((self.batch_size, *self.dim))

            # Generate data
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                x = np.load(self.data_directory + str(ID) + self.npy)#[:,:,:,0]
                #pre-porcessing    
                X[i,] = self._preprocess_sample_(x)[..., np.newaxis]

            return X, X 
                
class TestGenerator(keras.utils.Sequence):

    def __init__(self, data_directory, ids, labels, dim, balanced = False, batch_size=6,
                 n_classes=2):
        'Initialization'
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = ids
        
        if balanced:
            self._balanced_()
        
        self.dim = dim
        self.n_classes = n_classes
        self.npy = '.npy'
        self.on_epoch_end()
        
        if (len(np.unique(labels))>3) or (labels.min() >1): #regression
            print('{} regression samples'.format(len(self.labels)))
        else:
            print('{} positive samples and {} negative ones'.format(np.sum(self.labels), len(self.labels)-np.sum(self.labels)))


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Find list of IDs
        list_IDs_temp = self.list_IDs[index*self.batch_size:(index+1)*self.batch_size]
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            try:
                #try:
                    #pre-processing    
                X[i,] = np.load(self.data_directory + str(ID) + self.npy)[..., np.newaxis]
                # Store class
                y[i] = self.labels[ID]
                #except (ValueError,KeyError):
                    #print('Shape/key-error {} sample'.format(ID))
                    
            except FileNotFoundError:
                print(ID, ' not found')
            

        return X, y
    
    def _balanced_(self):
        """ """
        list_IDs_pos = list(self.labels[self.labels==1].index)
        list_IDs_neg = list(self.labels[self.labels==0].index[:len(list_IDs_pos)])
        
        self.list_IDs = list_IDs_pos+list_IDs_neg
        self.labels = self.labels[self.list_IDs]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))


    def get_labels(self):
        """
        Returns the labels of the dataset
        """
        return self.labels
    
    def get_class1_proportion(self):
        """
        Returns the proportion of Class 1 samples
        """
        
        return np.sum(self.labels)/len(self.labels)
    
class ArrayGenerator(keras.utils.Sequence):

    def __init__(self, x_set, y_set, risk_factors=None, batch_size=16, balanced = False):
        self.batch_size = batch_size
        
        if not isinstance(x_set,str ):
            self.x = x_set
            self.load_all =True
        else :
            self.x = x_set # is a path to a hdf5 file
            self.load_all =False
        
        # tabular data
        if (risk_factors is not None):
            y_set = y_set.to_frame()
            y_set['id_nb'] = range(len(y_set))
            self.tabular = y_set
            del y_set
            
            for risk_factor in risk_factors:
                self.tabular = pd.concat((self.tabular, risk_factor), axis=1)
                self.tabular = self.tabular.dropna(how='any')
            self.tabular.rename(columns={self.tabular.columns[0]: 'labels'}, inplace=True)   
            
            if self.load_all:
                self.x = x_set[self.tabular.id_nb.astype(int).to_list(), ]
            else:
                self.x_idx = self.tabular.id_nb.astype(int).to_list()
                
            self.tabular.drop(columns='id_nb', inplace=True)
            self.labels =self.tabular.labels

        else :
            self.labels = y_set
            self.__getitem__ = self.__getitem__image

    def __len__(self):
        return int(np.ceil(len(self.labels) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        
        if self.load_all :
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        else:
            with h5py.File(self.x, 'r') as f:
                batch_x = f['X_train'][idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.tabular.labels.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_tab = self.tabular.iloc[idx * self.batch_size:(idx + 1) * self.batch_size,1: ]
        
        if len(batch_x)!=len(batch_y):
            print(idx)
        else:
            return [np.array(batch_x), np.array(batch_tab)], np.array(batch_y)
    def __getitem__image(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        if len(batch_x)!=len(batch_y):
            print(idx)
        else:
            return  np.array(batch_x), np.array(batch_y)
    
    def get_class_weights(self):
        """
        Returns the proportion of classes
        """
        class_w = class_weight.compute_class_weight('balanced', 
                                             np.unique(self.labels),
                                             self.labels)
        return {0: class_w[0], 1: class_w[1]} 
    
class DataGenerator_Autoencoder(keras.utils.Sequence):
                                    
    def __init__(self, data_directory, ids, dim,
                 pre_processing_dict= {'normalize' : False, 'rotate' : False, 'shift': False},
                 testing= False,
                 single_image = False,
                 batch_size=32,
                 shuffle=True):
        
        self.data_directory = data_directory #where to access the samples
        self.list_IDs = ids #list of samples
        self.batch_size = batch_size #size of the batch
        self.dim = dim
        
        if dim[0] == 1 : 
            self.multi_channel = False
            self.__data_generation = self.__data_generation_single
        else :
            self.multi_channel = True
            self.__data_generation = self.__data_generation_multi
            
        self.shuffle = shuffle #if data is shuffle at every epoch 
        
        # Preprocessing
        self.pre_processing = pre_processing_dict
        self.normalize = False
        self.shift = False
        self.rotate = False
        self._which_preprocessing_()
        
        if testing: #testing mode
            self._testing_()
        
        # Allocate to memory once for all
        self.npy = '.npy'
        
        # Display summary of what manipulation have been done
        self._print_info_(testing)
        
        #initialize the learning process
        self.on_epoch_end()
        
    def _print_info_(self ,testing):
        """
        Prints the info/set-up of the Generator when the instance is created
        """
        dir_= self.data_directory.split('/')[-2]
        #Number of samples
        info = '{} samples from {}'.format(len(self.list_IDs), dir_)    
        print(info)
        if testing:
            print('Testing Mode')
        
    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))
            
    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _testing_(self):
        """
        For testing mode, ensure that the following parameters are correctly set
        """
        self.suffle = False
        self.batch_size = 1
        self.normalize = False
        self.shift = False
        self.rotate = False
            
     ### --- Pre processing functions --- ###
    def _which_preprocessing_(self):
        """
        Which preprocessing should be done
        """
        if self.pre_processing['normalize']:
            self.normalize = True
        
        if self.pre_processing['flip']:
            self.flip = True
        
        if self.pre_processing['shift']:
            self.shift = self.pre_processing['shift'] + 1
        
        if self.pre_processing['rotate']:
            self.rotate = self.pre_processing['rotate'] + 1
    
    def _normalize_(self, array):
        """
        Normalize each frame
        """
        return array / 255. # array.max(axis = 3).max(axis= (1,2))
                
    def _shift_(self, array):
        """
        Shift range
        """
        shift_pix = np.random.randint(0,self.shift)
        if self.multi_channel:
            shifted = shift(array, [0, 0, shift_pix, 0])
        else:
            shifted = shift(array, [0, 0, shift_pix, 0])
        return shifted 
    
    def _rotate_(self, array):
        """ 
        Applies a random rotation of the image in the
        [ -'rotate_range' ; 'rotate_range' ] range to a stack
        """
        angle = np.random.randint(0, self.rotate)
        sign = np.random.randint(0,2)
        return rotate(array, -sign *angle, (1,2), reshape=False)

    def _preprocess_sample_(self, array):
        """
        Applies a random pre processing to a single sample
        """
        if self.normalize :
            array = self._normalize_(array)
        if self.shift:
            array = self._shift_(array)
        if self.rotate:
            array = self._rotate_(array) 
        return array
                
    ### --- Data reaching --- ###
    
    def __data_generation_single(self, list_IDs_temp):
        """
        Generates data containing batch_size samples
        """ 
        # Initialization
        X = np.empty((self.batch_size, *self.dim[1:]))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x = np.load(self.data_directory + str(ID) + self.npy)[0,]
            #pre-porcessing    
            X[i,] = x[..., np.newaxis]#self._preprocess_sample_(x)[..., np.newaxis]
            # Store class

        return X, X 
    
    def __data_generation_multi(self, list_IDs_temp):
        """Generates data containing batch_size samples"""
        # X : [n_samples, time, widht, height, n_channels]
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            x = np.load(self.data_directory + str(ID) + self.npy)
            #pre-porcessing    
            X[i,] = x[..., np.newaxis]#self._preprocess_sample_(x)[..., np.newaxis]
            # Store class

        return X, X 

    def __getitem__(self, index):
        """
        Generate a batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y    
