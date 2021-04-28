import sys
import os
import glob
import pandas as pd
import numpy as np
from multiprocessing import Pool
import time

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



split = int(sys.argv[1])

## Load Full raw data
#to del :
print("Load Data %s " %split)
sys.stdout.flush()
final_df = pd.read_csv('/n/groups/patel/samuel/EWAS/env_data/Dataset_%s.csv' % split).set_index('id')
final_df = final_df[~final_df.index.duplicated(keep='first')]
final_df['eid'] = final_df.index.str.split('_').str[0]
cols_age_sex_eid_ethnicity = ['Sex', 'eid', 'Age when attended assessment centre', 'Ethnicity']
features_cols = [elem for elem in final_df.columns if elem not in cols_age_sex_eid_ethnicity + ETHNICITY_COLS]
col_age_id_eid_sex_ethnicty = final_df[cols_age_sex_eid_ethnicity]



def parallel_group(col):
    print("Col : ", col)
    sys.stdout.flush()
    column_modified = compute_coefs_and_input(final_df, col)
    return column_modified

final_df_inputed = col_age_id_eid_sex_ethnicty
for idx, col in enumerate(features_cols) :
    print("Start inputing %s , %s / %s" % (col, idx, len(features_cols)))
    sys.stdout.flush()
    t1 = time.time()
    inputed_col = parallel_group(col)
    t2 = time.time()
    print("Done inputing %s , Time : %s" % (col, t2 - t1))
    sys.stdout.flush()
    final_df_inputed = final_df_inputed.join(inputed_col, how = 'outer', rsuffix = '_rsuffix')
    final_df_inputed = final_df_inputed[final_df_inputed.columns[~final_df_inputed.columns.str.contains('_rsuffix')]]


final_df_inputed.to_csv('/n/groups/patel/samuel/EWAS/env_data/Dataset_inputed_%s.csv' % split)
