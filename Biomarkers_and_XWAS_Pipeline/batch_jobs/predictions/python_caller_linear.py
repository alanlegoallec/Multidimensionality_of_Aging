import pandas as pd
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, linregress

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.load_and_save_data import load_data
from aging.model.load_and_save_environment_data import ETHNICITY_COLS
from aging.processing.base_processing import path_output_linear_study

dataset = sys.argv[1]

hyperparameters = dict()
hyperparameters['dataset'] = dataset


df, organ, view = load_data(dataset)
df = df.drop(columns = ['eid'])



## Linear EWAS :
#df_rescaled, scaler_residual = normalise_dataset(df)
columns_sex_ethnicity = ['Sex'] + ETHNICITY_COLS
cols_except_age_sex_residual_ethnicty = df.drop(columns = ['Age when attended assessment centre', 'Sex'] + ETHNICITY_COLS).columns


d = pd.DataFrame(columns = ['feature_name', 'p_val', 'corr_value', 'size_na_dropped'])
for column in cols_except_age_sex_residual_ethnicty:
    df_col = df[[column, 'Age when attended assessment centre'] + columns_sex_ethnicity]
    df_col = df_col.dropna()

    lin_residual = LinearRegression()
    lin_residual.fit(df_col[columns_sex_ethnicity].values, df_col['Age when attended assessment centre'].values)
    res_residual = df_col['Age when attended assessment centre'].values - lin_residual.predict(df_col[columns_sex_ethnicity].values)

    lin_feature = LinearRegression()
    lin_feature.fit(df_col[columns_sex_ethnicity].values, df_col[column].values)
    res_feature = df_col[column].values - lin_feature.predict(df_col[columns_sex_ethnicity].values)

    corr, p_val = pearsonr(res_residual, res_feature)
    d = d.append({'feature_name' : column, 'p_val' : p_val, 'corr_value' : corr, 'size_na_dropped' : df_col.shape[0]}, ignore_index = True)
d.to_csv(path_output_linear_study + 'linear_correlations_%s.csv' % dataset, index=False)
