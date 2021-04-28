import sys
import os
import glob
import pandas as pd

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.environment_processing.base_processing import path_inputs_env, path_input_env
from aging.model.load_and_save_environment_data import map_envdataset_to_dataloader_and_field

ETHNICITY_COLS = ['Ethnicity.White',
       'Ethnicity.British', 'Ethnicity.Irish', 'Ethnicity.White_Other',
       'Ethnicity.Mixed', 'Ethnicity.White_and_Black_Caribbean',
       'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian',
       'Ethnicity.Mixed_Other', 'Ethnicity.Asian', 'Ethnicity.Indian',
       'Ethnicity.Pakistani', 'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other',
       'Ethnicity.Black', 'Ethnicity.Caribbean', 'Ethnicity.African',
       'Ethnicity.Black_Other', 'Ethnicity.Chinese', 'Ethnicity.Other',
       'Ethnicity.Other_ethnicity', 'Ethnicity.Do_not_know',
       'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA', 'Ethnicty.Other']

df_sex_age_ethnicity = pd.read_csv('/n/groups/patel/Alan/Aging/Medical_Images/data/data-features_instances.csv').set_index('id').drop(columns = ['instance', 'Abdominal_images_quality'])
df_sex_age_ethnicity = df_sex_age_ethnicity.rename(columns = {'Age' : 'Age when attended assessment centre'})
for idx, df in enumerate(map_envdataset_to_dataloader_and_field.keys()):
    if 'medical' not in df :
        print(df)
        path = path_inputs_env + df + '.csv'
    #for idx, path in enumerate(glob.glob(path_inputs_env + '*.csv')):
        df_temp = pd.read_csv(path).set_index('id')
        used_cols = [elem for elem in df_temp.columns if elem not in ['Sex', 'eid', 'Age when attended assessment centre'] + ETHNICITY_COLS]
        int_cols = df_temp.select_dtypes(int).columns
        df_temp[int_cols] = df_temp[int_cols].astype('Int64')
        if idx == 0:
            final_df = df_sex_age_ethnicity.join(df_temp[used_cols], how = 'outer', rsuffix = '_rsuffix')
        else :
            final_df = final_df.join(df_temp[used_cols], how = 'outer', rsuffix = '_rsuffix')
        final_df = final_df[final_df.columns[~final_df.columns.str.endswith('_rsuffix')]]
print("Starting merge")
#final_df = final_df.reset_index().merge(df_ethnicity, on = 'eid').set_index('id')
final_df.to_csv(path_input_env)
