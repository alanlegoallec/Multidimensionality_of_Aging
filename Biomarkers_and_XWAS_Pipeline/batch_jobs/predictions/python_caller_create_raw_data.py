import sys
import os
import glob
import pandas as pd

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.processing.base_processing import path_inputs, path_input



df_sex_age_ethnicity = pd.read_csv('/n/groups/patel/samuel/sex_age_eid_ethnicity.csv').set_index('id')

for idx, elem in enumerate(glob.glob(path_inputs + '*.csv')):
    df_temp = pd.read_csv(elem).set_index('id')
    used_cols = [elem for elem in df_temp.columns if elem not in ['Sex', 'eid', 'Age when attended assessment centre']]
    int_cols = df_temp.select_dtypes(int).columns
    df_temp[int_cols] = df_temp[int_cols].astype('Int64')
    if idx == 0:
        final_df = df_sex_age_ethnicity.join(df_temp[used_cols], how = 'outer')
    else :
        final_df = final_df.join(df_temp[used_cols], how = 'outer')
print("Starting merge")
final_df.to_csv(path_input)
