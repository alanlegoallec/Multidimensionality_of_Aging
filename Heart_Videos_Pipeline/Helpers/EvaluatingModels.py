import os
import re
import keras
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from keras.callbacks import Callback
from sklearn.metrics import r2_score, mean_squared_error, f1_score, log_loss, average_precision_score

from DataFunctions import DataGenerator, TestGenerator, ArrayGenerator
from EvaluatingFunctions import R2_, RMSE

### ----------- Loading weights and resume training functions ------------ ###    

def atoi(text):
    """
    Converts str digits into int
    """
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    Alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    
    """
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def save_print_lr(index, lr):
    """
    Print the learning rate and save its value for later use
    """
    print('Learning Rate : ', lr)
    np.save('lr.npy',lr)
    return lr

def resume_training(model, lr, best = False, dir_ = None):
    """
    Check is the score.css file is present in the directory. If it is,
    load the last/best model'weigths.
    """
    #if not a certain directory is given
    if not dir_:
        dir_ = os.getcwd()+'/'
        
    if 'score.csv' in os.listdir(dir_):
        print('Resume Training')
        weights_files = [w for w in os.listdir(dir_) if w.endswith('.h5')]
        weights_files.sort(key=natural_keys)
        
        if best:
            df_score = pd.read_csv('score.csv', index_col ='Unnamed: 0')
            argmax_score = df_score['R2_val'].idxmax()
            #weights files:
            print('Loading ', dir_ + weights_files[argmax_score-1])
            model.load_weights(dir_ + weights_files[argmax_score-1])

        else: #load last weights 
            #print('Loading ', dir_ + weights_files[-1])
            weights_files = [dir_ + w for w in os.listdir(dir_) if w.endswith('.h5')]
            w = max(weights_files, key=os.path.getctime)
            print('Loading ',w)
            model.load_weights(w) #dir_ + weights_files[-1]

        if 'lr.npy' in os.listdir(dir_):
            lr_read = np.load(dir_ + 'lr.npy')

            if lr > lr_read:
                lr = lr_read
            else:
                pass

        return model, lr
    else:
        print('No .h5 file found - First training.')
        return model, lr

### ----------- Custom CallBacks ------------ ###    
    
class CustomCallback(Callback):
    """
    This class enabales to gathe rall callback into a single one.
    It has several roles :
    1) Evaluate the validation set trhought trainign at the end of each epoch.
       This is done by the Keras Sequential.fit/fit_generator function but the
       score is not computed over all the valdiation set but rather as the 
       average score over all the batches. For R2 metric, this could cause some issues.
    2) Save models through training (model check point)
    3) Keep track of the performance and the loss
    4) Stop training when learning has stopped (early stopping)
    4) Terminate training if nan occurs 
    
    This class should be implemented completely by which metric and loss.
    
    it inherits form the Callback class from Keras (https://keras.io/callbacks/)
    """
    def __init__(self, validation_data, patience, restore_best_weights, baseline=False, 
                 output_dir = False, input_dir = False, restore= True, save_best = False, 
                 save = True):        
        
        if isinstance(validation_data, DataGenerator) or  isinstance(validation_data, keras.utils.Sequence) or isinstance(validation_data, ArrayGenerator) :
            self.val_data = validation_data
            self.val_labels =  validation_data.labels
            self.val_data.batch_size = 1 #make sure the batch_size is 1
            self.get_predictions = self.get_predictions_generator
        elif isinstance(validation_data, tuple):
            self.val_data = validation_data[0]
            self.val_labels = validation_data[1]
            self.get_predictions = self.get_predictions_array
        else:
            raise TypeError('Validation Data is not in the correct format. DataGenerator or tuple (X,y)')
            
        self.monitor_op = np.greater #operation to compare the evolution of the validation score
        self.verbose = 1
        self.baseline = baseline # baseline score for training resuming
        self.patience = patience # how many epoch before earlystopping
        self.min_delta = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.save_best = save_best
        self.best_val_score = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.save = save

        if output_dir:
            #self.output_dir = output_dir 
            self.output_dir = os.getcwd() + '/'
        else:
            self.output_dir = os.getcwd() + '/'
            
        if input_dir:
            self.input_dir = input_dir
        else:
            self.input_dir = './'
        
        print(self.input_dir)
        
        if ('score.csv' in os.listdir(self.input_dir)) and restore :
            print('Load existing Scores and Losses.')
            self.train_loss = list(np.load(self.input_dir+'Tr_loss.npy'))
            try : 
                self.df_score = pd.read_csv(self.input_dir+'score.csv')
                self.idx_epoch = self.df_score.shape[0]
            except EmptyDataError :
                print('Empty CSV')
                self.idx_epoch = 0
                self.df_score = pd.DataFrame(index =range(self.idx_epoch))
        else:
            print('No csv found')
            self.df_score = pd.DataFrame()
            self.train_loss = []
            self.idx_epoch= 1

    def on_train_begin(self, logs=None):
        """
        Set the parameters for a new training
        """
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0

        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def terminateOnNan(self, batch, logs):
        """
        Stops the learning process if a Nan is produced
        """
        logs = logs or {}
        loss = logs.get('loss')

        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print('Batch %d: Invalid loss, terminating training' % (batch))
                self.model.stop_training = True        

    def earlyStopping(self, epoch, logs, score):
        """
        Stops the training process if the validation score does not improve
        """
        
        if self.monitor_op(score - self.min_delta, self.best):
            self.best = score
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()

        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('Validation R2 not imporving, stop training')
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of '
                              'the best epoch')
                    self.model.set_weights(self.best_weights)

    def on_batch_end(self, batch, logs=None):
        
        self.train_loss.append(logs.get('loss'))
        self.terminateOnNan(batch, logs)
    
    def get_predictions_generator(self):
        """
        Computes the predictions of the validation DataGenerator
        """
        return self.model.predict_generator(self.val_data)
    
    def get_predictions_array(self):
        """
        Computes the predictions of the validation data array
        """
        return self.model.predict(self.val_data)

    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError("Subclasses should implement this!")

    def save_model(self):
        raise NotImplementedError("Subclasses should implement this!")
        
class R2Callback(CustomCallback):
    """
    Uses the R-squarred metric as monitoring, and RMSE as loss function
    """
    def __init__(self, validation_data, patience, restore_best_weights, baseline=None, 
                 output_dir = './', input_dir = False, restore= True, save_best=False, 
                 save = True):
        super().__init__(validation_data,  patience, 
                         restore_best_weights, baseline, 
                         output_dir, input_dir, restore, save_best, save)
        print("R square callback")
        
    def on_epoch_end(self, epoch, logs=None):
        """
        
        """
        try:
            preds = self.get_predictions()
        except ValueError:
            preds = np.zeros(len(self.val_labels))
        
        self.val_R2 = r2_score(self.val_labels, preds)
        RMSE_tmp = np.sqrt(mean_squared_error(self.val_labels, preds))

        print('R2 val : {} , loss val : {}'.format(np.round(self.val_R2,3), np.round(RMSE_tmp,3)))
        self.df_score.loc[self.idx_epoch, 'R2_val']= self.val_R2
        self.df_score.loc[self.idx_epoch, 'RMSE_val']= RMSE_tmp
        self.df_score.loc[self.idx_epoch, 'RMSE_tr']= logs.get('loss')
        self.df_score.loc[self.idx_epoch, 'R2_tr']= logs.get('R2_')

        # Savings 
        self.df_score.to_csv(self.output_dir + './score.csv')        
        if self.save : 
            np.save(self.output_dir + './Tr_loss.npy', np.array(self.train_loss))
            self.save_model()

        self.earlyStopping(epoch, logs, self.val_R2)
        self.idx_epoch +=1

    def save_model(self):
        """Saves the model at each epoch"""
        if (self.save_best and (self.best_val_score < self.val_R2)):
            self.best_val_score = self.val_R2
            
            try:
                str_ = self.output_dir + 'epoch_{:04.0f}.h5'.format(self.idx_epoch)
                self.model.save(str_)
                print(str_, 'Ouput_dir')
            except:
                print('error self.output')
                str_ = './epoch_{:04.0f}.h5'.format(self.idx_epoch)
                self.model.save(str_)
                print(str_)
            
            if self.save_best:
                print('Best Model epoch_{:04.03f}.h5 saved - {}'.format(self.idx_epoch, np.round(self.best_val_score,3)))
        
        elif not self.save_best:
            try:
                str_ = self.output_dir + 'epoch_{:04.0f}.h5'.format(self.idx_epoch)
                self.model.save(str_)
                print(str_, 'Ouput_dir')
            except:
                print('error self.output')
                str_ = './epoch_{:04.0f}.h5'.format(self.idx_epoch)
                self.model.save(str_)
                print(str_)

class F1Callback(CustomCallback):
    """
    Uses the F-1 metric as monitoring, and binary cross entropy as loss function.
    """
    def __init__(self, validation_data, patience, restore_best_weights, baseline=None, 
                 output_dir = None, input_dir =None, restore= True, save_best=False, 
                 save = True):
        
        super().__init__(validation_data,  patience, 
                         restore_best_weights, baseline, 
                         output_dir, input_dir, restore, save_best, save)
        
        print("F1 callback")
        self.null_f1 = 0
    
    def terminateNullF1(self, F1):
        """ Ends training if F1 is zero"""
        
        if F1 < 1e-5 :
            self.null_f1+=1
            
            if self.null_f1 == 10:
                print('Null F1. Stop Training')
                self.model.stop_training = True  

    def on_epoch_end(self, epoch, logs=None):
        """
        
        """
        preds = self.get_predictions()        
        discr = preds.copy()
        discr[discr > 0.5] = 1
        discr[discr <= 0.5] = 0
        f1_tmp = f1_score(self.val_labels, discr)
        loss_tmp = log_loss(self.val_labels, preds)
        avg_prec = average_precision_score(self.val_labels, preds, pos_label =1)
        self.val_F1 = f1_tmp

        print('f1 val : {} , loss val : {}, Avg. Prec. : {}'.format(np.round(f1_tmp,2), np.round(loss_tmp,2), np.round(avg_prec,2)))
        self.df_score.loc[self.idx_epoch, 'AvgPrec_val']= avg_prec
        self.df_score.loc[self.idx_epoch, 'f1_val']= f1_tmp
        self.df_score.loc[self.idx_epoch, 'loss_val']= loss_tmp
        self.df_score.loc[self.idx_epoch, 'loss_tr']= logs.get('loss')
        self.df_score.loc[self.idx_epoch, 'f1_tr']= logs.get('f1_')
        self.df_score.to_csv(self.output_dir + './score.csv')

        self.terminateNullF1(f1_tmp)
        
        # Savings 
        if self.save:
            np.save(self.output_dir + './Tr_loss.npy', np.array(self.train_loss))
            self.df_score.to_csv(self.output_dir + './score.csv')        
            self.save_model()

        self.earlyStopping(epoch, logs, f1_tmp)
        self.idx_epoch +=1


    def save_model(self):
        """Saves the model at each epoch"""
         
        if (self.save_best and (self.best_val_score < self.val_F1)) or (not self.save_best):
            self.best_val_score = self.val_F1
            str_ = self.output_dir + 'epoch_{:04.0f}.h5'.format(self.idx_epoch)
            
            if self.save_best:
                print('Model epoch_{:04.03f}.h5 saved - {}'.format(self.idx_epoch, np.round(self.best_val_score,3)))

            self.model.save(str_)

### ----------- Retrieve scores -------------- ###

def retrieve_score(directory_ = None):
    """
    Retrieves the results of the grid search sotred in an score.csv file in each folder.
    A single dataframe with multi columns index is returned.
    """
    sub_directories=False
    directories = [x for x in os.listdir(directory_) if x!='Figures']
    for fname in directories:
        if os.path.isdir(os.path.join(directory_,fname)) and fname != '.ipynb_checkpoints':
            sub_directories=True
            break
            
    if not sub_directories :
        df_scores =pd.read_csv(directory_+'score.csv')#, index_col ='Unnamed: 0')
        cols= [c for c in df_scores.columns if c.startswith('Unnamed')]
        df_scores.drop(columns=cols, inplace= True)
        loss = np.load(directory_+'Tr_loss.npy')
        df_losses =pd.DataFrame(data=loss, columns=['loss_tr'])
    
    else:
        folders_name = [o for o in os.listdir(directory_) if ((os.path.isdir(os.path.join(directory_,o))) and (not o.startswith('.')) and (not o=='Figures'))]
        folders_dir = [directory_+ o+'/' for o in folders_name]
        df_scores = pd.DataFrame()
        df_losses = pd.DataFrame()

        for i,fol in enumerate(folders_dir):
            df_tmp =pd.read_csv(fol+'score.csv', index_col ='Unnamed: 0')
            
            unnamed_cols = [col for col in df_tmp.columns if col.startswith('Unnamed')]
            if unnamed_cols:
                df_tmp.drop(columns=unnamed_cols, inplace =True)
            
            df_tmp.columns = pd.MultiIndex.from_product([[folders_name[i]], df_tmp.columns ])
            df_scores = pd.concat((df_scores,df_tmp), axis=1)

            loss = np.load(fol+'Tr_loss.npy')
            df_tmp_l =pd.DataFrame(data=loss, columns=[folders_name[i]])
            df_losses = pd.concat((df_losses,df_tmp_l), axis=1)

        df_scores = df_scores.swaplevel(axis=1).sort_index(level=0,axis=1)
    
    return df_scores, df_losses    