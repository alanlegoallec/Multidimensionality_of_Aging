import sys
import os
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool


if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

ETHNICITY_COLS = ['Ethnicity.White',
       'Ethnicity.British', 'Ethnicity.Irish', 'Ethnicity.White_Other',
       'Ethnicity.Mixed', 'Ethnicity.White_and_Black_Caribbean',
       'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian',
       'Ethnicity.Mixed_Other', 'Ethnicity.Asian', 'Ethnicity.Indian',
       'Ethnicity.Pakistani', 'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other',
       'Ethnicity.Black', 'Ethnicity.Caribbean', 'Ethnicity.African',
       'Ethnicity.Black_Other', 'Ethnicity.Chinese', 'Ethnicity.Other',
       'Ethnicity.Other_ethnicity', 'Ethnicity.Do_not_know',
       'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA']
from aging.environment_processing.base_processing import path_input_env, path_input_env_inputed
from aging.model.InputtingNans import  load_raw_data, compute_coefs_and_input

cols_age_sex_eid_ethnicity = ['Sex', 'eid', 'Age when attended assessment centre'] + ETHNICITY_COLS


n_cores = int(sys.argv[1])

## Load Full raw data
#to del :
features_cols, final_df = load_raw_data(path_raw = path_input_env, path_output = path_input_env_inputed)
#features_cols, final_df = load_raw_data(path_raw = '/n/groups/patel/samuel/EWAS/test_inputing.csv', path_output = path_input_env_inputed, path_inputs = path_inputs_env)
col_age_id_eid_sex_ethnicty = final_df[cols_age_sex_eid_ethnicity]
## split continuous and categorical


# split_cols = np.array_split(features_cols, n_cores)
# def parallel_group_of_features(final_df, split_col):
#     list_features_split = []
#     for col in split_col:
#         column_modified = compute_coefs_and_input(final_df, col)
#         list_features_split.append(column_modified[col])
#
#     inputed_res = col_age_id_eid_sex_ethnicty
#     for column in list_features_split:
#         inputed_res = inputed_res.join(column)
#     return inputed_res
#
# def parallel_group(split_col):
#     print("Split col : ", split_col)
#     return parallel_group_of_features(final_df, split_col)
#
# pool = Pool(n_cores)
# final_df_inputed_cols = pool.map(parallel_group, split_cols)
# pool.close()
# pool.join()


def parallel_group(col):
    print("Col : ", col)
    column_modified = compute_coefs_and_input(final_df, col)
    return column_modified

pool = Pool(processes=n_cores)
final_df_inputed_cols = pool.map(parallel_group, features_cols)
pool.close()
pool.join()


final_df_inputed = col_age_id_eid_sex_ethnicty
for df in final_df_inputed_cols :
    final_df_inputed = final_df_inputed.join(df, how = 'outer', rsuffix = '_r')
    final_df_inputed = final_df_inputed[final_df_inputed.columns[~final_df_inputed.columns.str.contains('_r')]]


final_df_inputed.to_csv(path_input_env_inputed)
