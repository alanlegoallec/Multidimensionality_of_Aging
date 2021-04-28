
import sys
import os

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.environment_predictor import EnvironmentPredictor


name = sys.argv[1]
outer_splits = int(sys.argv[2])
inner_splits = int(sys.argv[3])
n_iter = int(sys.argv[4])
target_dataset = sys.argv[5]
input_dataset = sys.argv[6]
fold = int(sys.argv[7])



hyperparameters = dict()
hyperparameters['name'] = name
hyperparameters['outer_splits'] = outer_splits
hyperparameters['inner_splits'] = inner_splits
hyperparameters['n_iter'] = n_iter
hyperparameters['target_dataset'] = target_dataset
hyperparameters['input_dataset'] = input_dataset
hyperparameters['fold'] = fold
print(hyperparameters)

if fold not in list(range(outer_splits)):
    raise ValueError('fold must be < outer_splits, here fold = %s and outer_splits = %s' % (fold, outer_splits) )
else :
    gp = EnvironmentPredictor(name, outer_splits, inner_splits, n_iter, target_dataset, input_dataset, fold)
    print("Loading Dataset")
    df = gp.load_dataset().dropna()
    print("Dataset Loaded, optimizing hyper")
    df_predicts_no_scaled_test, df_predicts_no_scaled_val, df_predicts_no_scaled_train = gp.optimize_hyperparameters_fold(df)
    #print("Hyper Opt over, saving file")
    gp.save_predictions(df_predicts_no_scaled_val, 'val')
    gp.save_predictions(df_predicts_no_scaled_test, 'test')
    gp.save_predictions(df_predicts_no_scaled_train, 'train')
    #print("task complete")
