import numpy as np
import sys
import os
import glob
import pandas as pd

if sys.platform == 'linux':
	sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
	sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.load_and_save_data import load_data, map_dataset_to_field_and_dataloader, dict_dataset_to_organ_and_view
from aging.processing.base_processing import path_predictions
model = sys.argv[1]
target = sys.argv[2]
dataset = sys.argv[3]
outer_splits = int(sys.argv[4])


hyperparameters = dict()
hyperparameters['model'] = model
hyperparameters['target'] = target
hyperparameters['dataset'] = dataset
hyperparameters['outer_splits'] = outer_splits
print(hyperparameters)


# def dataset_map_fold(dataset, target, outer_splits):
#     dataset = dataset.replace('_', '')
#     df = load_data(dataset)
#     if target == 'Sex':
#         X = df.drop(columns = ['Sex', 'Age when attended assessment centre']).values
#         y = df['Sex'].values
#     elif target == 'Age':
#         X = df.drop(columns = ['Age when attended assessment centre']).values
#         y = df['Age when attended assessment centre'].values
#
#     outer_cv = KFold(n_splits = outer_splits, shuffle = False, random_state = 0)
#     list_folds = [elem[1] for elem in outer_cv.split(X, y)]
#     index = df.index
#
#     index_splits = [index[list_folds[elem]].values for elem in range(outer_splits)]
#     index_split_matching = [np.array( [fold]*len(index_splits[fold])) for fold in range(outer_splits) ]
#
#     map_eid_to_fold = dict(zip(np.concatenate(index_splits), np.concatenate(index_split_matching)))
#     return map_eid_to_fold

if 'Cluster' in dataset:
	dataset_proper = dataset.split('/')[-1].replace('.csv', '').replace('_', '.')
#	field = 'Cluster'
	organ = 'Cluster'
	view = 'main'
	transformation = 'Raw'
else :
	dataset_proper = dataset
	organ, view, transformation =  dict_dataset_to_organ_and_view[dataset_proper]
#	field, _ = map_dataset_to_field_and_dataloader[dataset_proper]

list_files = glob.glob( path_predictions + '*%s_%s*%s*.csv' % (target, dataset_proper, model))

list_train = [elem for elem in list_files if 'train' in elem]
list_test = [elem for elem in list_files if 'test' in elem]
list_val = [elem for elem in list_files if 'val' in elem]

if len(list_train) == outer_splits and len(list_test) == outer_splits and len(list_val) == outer_splits :

	df_train = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_train])
	df_test = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_test])
	df_val = pd.concat([pd.read_csv(elem).set_index('id') for elem in list_val])

	if 'chemestry' in organ:
		organ = organ.replace('chemestry', 'chemistry')


	#print('/n/groups/patel/samuel/preds_alan/Predictions_instances_%s_%s_%s_raw_%s_0_0_0_0_0_0_0_train.csv' % ( target, organ, view, model))
	df_train[['pred', 'outer_fold']].to_csv('/n/groups/patel/samuel/preds_alan2/Predictions_instances_%s_%s_%s_%s_%s_0_0_0_0_0_0_0_train.csv' % ( target, organ, view, transformation, model))
	df_test[['pred', 'outer_fold']].to_csv('/n/groups/patel/samuel/preds_alan2/Predictions_instances_%s_%s_%s_%s_%s_0_0_0_0_0_0_0_test.csv' % ( target, organ, view, transformation, model))
	df_val[['pred', 'outer_fold']].to_csv('/n/groups/patel/samuel/preds_alan2/Predictions_instances_%s_%s_%s_%s_%s_0_0_0_0_0_0_0_val.csv' % ( target, organ, view, transformation, model))


else :
	raise ValueError("ONE OF THE OUTER JOB HAS FAILED ! ")
