
import sys
import os

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.environment_predictor import EnvironmentPredictor



name = sys.argv[1]
n_iter = int(sys.argv[2])
target_dataset = sys.argv[3]
input_dataset = sys.argv[4]
n_splits = int(sys.argv[5])


hyperparameters = dict()
hyperparameters['name'] = name
hyperparameters['n_splits'] = n_splits
hyperparameters['n_iter'] = n_iter
hyperparameters['target_dataset'] = target_dataset
hyperparameters['input_dataset'] = input_dataset


print(hyperparameters)
gp = EnvironmentPredictor(name, -1, n_splits, n_iter, target_dataset, input_dataset, -1)
print("Loading Dataset")
df = gp.load_dataset().dropna()
print("Dataset Loaded, optimizing hyper")
#df_scaled = gp.normalise_dataset(df)
feature_importance_cols = gp.feature_importance(df)
print("Feature importance over, saving file")
gp.save_features(feature_importance_cols)
print("task complete")
