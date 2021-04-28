
import sys
import os

if sys.platform == 'linux':
	sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
	sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.specific_predictor import GeneralPredictor


name = sys.argv[1]
n_iter = int(sys.argv[2])
target = sys.argv[3]
dataset = sys.argv[4]
n_splits = int(sys.argv[5])


hyperparameters = dict()
hyperparameters['name'] = name
hyperparameters['n_splits'] = n_splits
hyperparameters['n_iter'] = n_iter
hyperparameters['target'] = target
hyperparameters['dataset'] = dataset


print(hyperparameters)
gp = GeneralPredictor(name, -1, n_splits, n_iter, target, dataset, -1, model_validate = 'HyperOpt')
print("Loading Dataset")
df, organ, view, transformation = gp.load_dataset()
gp.set_organ_view(organ, view, transformation)
print("Dataset Loaded, optimizing hyper")
# df_scaled = gp.normalise_dataset(df)
feature_importance_cols = gp.feature_importance(df)
print("Feature importance over, saving file")
gp.save_features(feature_importance_cols)
print("task complete")
