from .general_predictor import *
from .load_and_save_environment_data import load_data as load_data, save_features_to_csv, save_predictions_to_csv



class EnvironmentPredictor(BaseModel):
    def __init__(self, model, outer_splits, inner_splits, n_iter, target_dataset, env_dataset, fold):
        BaseModel.__init__(self, model, outer_splits, inner_splits, n_iter)
        self.fold = fold
        self.env_dataset = env_dataset
        self.target_dataset = target_dataset
        if model == 'ElasticNet':
            self.model = ElasticNet(max_iter = 2000)
        elif model == 'RandomForest':
            self.model = RandomForestRegressor()
        elif model == 'GradientBoosting':
            self.model = GradientBoostingRegressor()
        elif model == 'Xgboost':
            self.model = XGBRegressor()
        elif model == 'LightGbm':
            self.model = LGBMRegressor()
        elif model == 'NeuralNetwork':
            self.model = MLPRegressor(solver = 'adam', activation = 'relu', hidden_layer_sizes = (128, 64, 32), batch_size = 1000, early_stopping = True)


    def load_dataset(self, **kwargs):
        return load_data(self.env_dataset, self.target_dataset, **kwargs)


    def optimize_hyperparameters_fold(self, df):
        X = df.drop(columns = ['residuals'])
        y = df[['residuals', 'eid']]
        print(X.index, X.columns)
        return self.optimize_hyperparameters_fold_(X, y, 'r2', self.fold, organ = None, view = None, transformation = None)


    def feature_importance(self, df):
        X = df.drop(columns = ['residuals'])
        y = df[['residuals', 'eid']]
        self.features_importance_(X, y, 'r2', organ = None, view = None, transformation = None)
        return df.drop(columns = ['eid', 'residuals']).columns



    def save_features(self, cols):
        if not hasattr(self, 'features_imp'):
            raise ValueError('Features importance not trained')
        save_features_to_csv(cols, self.features_imp, self.target_dataset, self.env_dataset, self.model_name)

    def save_predictions(self, predicts_df, step):
        if not hasattr(self, 'best_params'):
            raise ValueError('Predictions not trained')
        save_predictions_to_csv(predicts_df, step, self.target_dataset, self.env_dataset, self.model_name, self.fold, self.best_params)
