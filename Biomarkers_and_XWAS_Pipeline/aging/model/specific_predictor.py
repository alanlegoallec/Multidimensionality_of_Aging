
from .general_predictor import *
from .load_and_save_data import load_data, save_features_to_csv, save_predictions_to_csv, dict_dataset_to_organ_and_view


class GeneralPredictor(BaseModel):
    def __init__(self, model, outer_splits, inner_splits, n_iter, target, dataset, fold, model_validate = 'HyperOpt'):
        BaseModel.__init__(self, model, outer_splits, inner_splits, n_iter, model_validate)
        self.fold = fold
        self.dataset = dataset
        if target == 'Sex':
            self.scoring = 'f1'
            self.target = 'Sex'
            if model == 'ElasticNet':
                self.model = SGDClassifier(loss = 'log', penalty = 'elasticnet', max_iter = 2000)
            elif model == 'RandomForest':
                self.model = RandomForestClassifier()
            elif model == 'GradientBoosting':
                self.model = GradientBoostingClassifier()
            elif model == 'Xgboost':
                self.model = XGBClassifier()
            elif model == 'LightGbm':
                self.model = LGBMClassifier()
            elif model == 'NeuralNetwork':
                self.model = MLPClassifier(solver = 'adam', activation = 'relu', hidden_layer_sizes = (128, 64, 32), batch_size = 1000, early_stopping = True)
        elif target == 'Age':
            self.scoring = 'r2'
            self.target = 'Age'
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
        else :
            raise ValueError('target : "%s" not valid, please enter "Sex" or "Age"' % target)

    def set_organ_view(self, organ, view, transformation):
        self.organ = organ
        self.view = view
        self.transformation = transformation

    def load_dataset(self, **kwargs):
        return load_data(self.dataset, **kwargs)


    def optimize_hyperparameters_fold(self, df):
        if self.target == 'Sex':
            X = df.drop(columns = ['Sex', 'Age when attended assessment centre'])
            y = df[['Sex', 'eid']]
        elif self.target == 'Age':
            X = df.drop(columns = ['Age when attended assessment centre'])
            y = df[['Age when attended assessment centre', 'eid']]
        else :
            raise ValueError('GeneralPredictor not instancied')
        return self.optimize_hyperparameters_fold_(X, y, self.scoring, self.fold, self.organ, self.view, self.transformation)


    def feature_importance(self, df):
        if self.target == 'Sex':
            X = df.drop(columns = ['Sex', 'Age when attended assessment centre'])
            y = df[['Sex', 'eid']]
            self.features_importance_(X, y, self.scoring, self.organ, self.view, self.transformation)
            return df.drop(columns = ['Sex', 'Age when attended assessment centre', 'eid']).columns
        elif self.target == 'Age':
            X = df.drop(columns = ['Age when attended assessment centre'])
            y = df[['Age when attended assessment centre', 'eid']]
            self.features_importance_(X, y, self.scoring, self.organ, self.view, self.transformation)
            return df.drop(columns = ['Age when attended assessment centre', 'eid']).columns
        else :
            raise ValueError('GeneralPredictor not instancied')


    def save_features(self, cols):
        if 'Cluster' in self.dataset:
            dataset_proper = self.dataset.split('/')[-1].replace('.csv', '').replace('_', '.')
        else :
            dataset_proper = self.dataset
        if not hasattr(self, 'features_imp') and self.model_name != 'Correlation' :
            raise ValueError('Features importance not trained')
        save_features_to_csv(cols, self.features_imp, self.target, self.organ, self.view , self.transformation, self.model_name, method = None)
        save_features_to_csv(cols, self.features_imp_sd, self.target, self.organ, self.view , self.transformation, self.model_name, method = 'sd')
        save_features_to_csv(cols, self.features_imp_mean, self.target, self.organ, self.view, self.transformation, self.model_name, method = 'mean')

    def save_predictions(self, predicts_df, step):
        if 'Cluster' in self.dataset:
            dataset_proper = self.dataset.split('/')[-1].replace('.csv', '').replace('_', '.')
        else :
            dataset_proper = self.dataset
        if not hasattr(self, 'best_params'):
            raise ValueError('Predictions not trained')
        save_predictions_to_csv(predicts_df, step, self.target, dataset_proper, self.model_name, self.fold, self.best_params)
