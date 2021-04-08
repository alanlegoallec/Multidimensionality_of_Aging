from keras import backend as K
"""
Functions used to monitor training evolution and model's performance. Use operation from keras/tensorflow to be applied directly on tensorflow.Tensor
"""


def recall_(y_true, y_pred):
    """Compute recall"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_(y_true, y_pred):
    """Compute precision"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_(y_true, y_pred):
    """Compute F1 score"""
    precision = precision_(y_true, y_pred)
    recall = recall_(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def R2_(y_true, y_pred):
    """Compute R squarrd"""
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def RMSE(y_true, y_pred):
    """Root mean square error"""
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 