import numpy as np
import pandas as pd
from .base_processing import  path_data

"""
4081	Method of measuring blood pressure

4079	Diastolic blood pressure, automated reading
94	Diastolic blood pressure, manual reading

95	Pulse rate (during blood-pressure measurement)
102	Pulse rate, automated reading

4080	Systolic blood pressure, automated reading
93	Systolic blood pressure, manual reading

"""

def read_blood_pressure_data(**kwargs):
    list_df = []

    for instance in range(4):
        age_col = '21003-' + str(instance) + '.0'
        cols_age_eid_sex = ['eid', age_col, '31-0.0']
        cols = ['4081-%s.0' % instance,'4081-%s.1' % instance, '4079-%s.0' % instance, '4079-%s.1' % instance, '94-%s.0' % instance, '94-%s.1' % instance, '95-%s.0' % instance, '95-%s.1' % instance,
                   '102-%s.0' % instance, '102-%s.1' % instance, '4080-%s.0' % instance, '4080-%s.1' % instance, '93-%s.0' % instance, '93-%s.1' % instance]

        d = pd.read_csv(path_data, usecols = cols_age_eid_sex + cols, **kwargs)
        d = d[~d[cols].isna().all(axis = 1)]
        d = d[d['4081-%s.0' % instance].isin([1, 2, 3]) & d['4081-%s.1' % instance].isin([1, 2, 3])]

        def custom_apply(row):
            method_first = row['4081-%s.0' % instance]
            method_second = row['4081-%s.1' % instance]
            if method_first == 1:
                values1 = row['4079-%s.0' % instance], row['102-%s.0' % instance],  row['4080-%s.0' % instance]
            elif method_first in [2, 3]:
                values1 = row['94-%s.0' % instance], row['95-%s.0' % instance], row['93-%s.0' % instance]
            else :
                values1 = np.nan, np.nan, np.nan

            if method_second == 1:
                values2 = row['4079-%s.1' % instance], row['102-%s.1' % instance],  row['4080-%s.1' % instance]

            elif method_second in [2, 3]:
                values2 = row['94-%s.1' % instance], row['95-%s.1' % instance], row['93-%s.1' % instance]
            else :
                values2 = np.nan, np.nan, np.nan

            values1, values2 = list(values1), list(values2)
            if pd.isna(values1 + values2).all():
                values_final = [np.nan, np.nan, np.nan]
            else :
                values_final = list(np.nansum([values1, values2], axis = 0)/2)
            #cols_name = ['eid', 'Age when attended assessment centre', 'Sex', 'Diastolic blood pressure_0', 'Pulse rate_0', 'Systolic blood pressure_0',
            #             'Diastolic blood pressure_1', 'Pulse rate_1', 'Systolic blood pressure_1']

            cols_name = ['eid', 'Age when attended assessment centre', 'Sex', 'Diastolic blood pressure', 'Pulse rate', 'Systolic blood pressure']

            return pd.Series([str(int(row['eid'])), row[age_col], row['31-0.0']] + values_final, index = cols_name)


        df_ = d.apply(custom_apply, axis = 1)
        df_['id'] = df_['eid'] + '_' + str(instance)
        df_['eid'] = df_['eid'].astype(int)
        list_df.append(df_)

    return pd.concat(list_df).set_index('id')
