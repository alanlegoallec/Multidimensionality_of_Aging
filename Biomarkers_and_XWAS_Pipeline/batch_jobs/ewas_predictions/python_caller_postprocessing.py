import numpy as np
import sys
import os
import glob
#from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, StratifiedKFold, RandomizedSearchCV
import pandas as pd

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.environment_processing.base_processing import path_predictions, path_final_preds


model = sys.argv[1]
target_dataset = sys.argv[2]
input_dataset = sys.argv[3]
outer_splits = int(sys.argv[4])


hyperparameters = dict()
hyperparameters['model'] = model
hyperparameters['target_dataset'] = target_dataset
hyperparameters['input_dataset'] = input_dataset
hyperparameters['outer_splits'] = outer_splits
print(hyperparameters)



#def dataset_map_fold(input_dataset, target_dataset, outer_splits):

#    df = load_data(input_dataset, target_dataset)
#    X = df.drop(columns = ['residual', 'Age']).values
#    y = df['residual'].values
#
#    outer_cv = KFold(n_splits = outer_splits, shuffle = False, random_state = 0)
#    list_folds = [elem[1] for elem in outer_cv.split(X, y)]
#    index = df.index
#
#    index_splits = [index[list_folds[elem]].values for elem in range(outer_splits)]
#    index_split_matching = [np.array( [fold]*len(index_splits[fold])) for fold in range(outer_splits) ]
#
#    map_eid_to_fold = dict(zip(np.concatenate(index_splits), np.concatenate(index_split_matching)))
#    return map_eid_to_fold


dataset = '_' + input_dataset + '_' + target_dataset + '_'
list_files = glob.glob(path_predictions + '*%s*%s*.csv' % (dataset, model))

list_train = [elem for elem in list_files if 'train' in elem]
list_test = [elem for elem in list_files if 'test' in elem]
list_val = [elem for elem in list_files if 'val' in elem]

if len(list_train) == outer_splits and len(list_test) == outer_splits and len(list_val) == outer_splits :

    df_train = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_train])
    df_test = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_test])
    df_val = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_val])

    # Avg df_val
    #df_val = df_val.groupby('id').agg({'pred' : 'mean'})
    #df_train = df_train.groupby('eid').agg({'predictions' : 'mean'})
    #map_eid_to_fold = dataset_map_fold(input_dataset, target_dataset, outer_splits)
    #df_val['fold'] = df_val.index.map(map_eid_to_fold)
    #df_train['fold'] = df_train.index.map(map_eid_to_fold)

    ## Save datasets :
    #Predictions_Sex_UrineBiochemestry_100083_main_raw_GradientBoosting_0_0_0_0_test.csv
    dataset = dataset.replace('_', '')

    df_train[['pred', 'outer_fold']].to_csv(path_final_preds + 'Predictions_%s_%s_%s_train.csv' % ( input_dataset, target_dataset,  model))
    df_test[['pred', 'outer_fold']].to_csv(path_final_preds + 'Predictions_%s_%s_%s_test.csv' % ( input_dataset, target_dataset,  model))
    df_val[['pred', 'outer_fold']].to_csv(path_final_preds + 'Predictions_%s_%s_%s_val.csv' % ( input_dataset, target_dataset,  model))

else :
    raise ValueError("ONE OF THE OUTER JOB HAS FAILED ! ")
