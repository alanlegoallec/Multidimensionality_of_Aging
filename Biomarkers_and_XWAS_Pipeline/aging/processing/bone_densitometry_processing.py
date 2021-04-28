import numpy as np
import pandas as pd
from .base_processing import path_data



"""
3081	Foot measured for bone density
19	Heel ultrasound method
3146	Speed of sound through heel
3143	Ankle spacing width
3144	Heel Broadband ultrasound attenuation, direct entry
3147	Heel quantitative ultrasound index (QUI), direct entry
3148	Heel bone mineral density (BMD)
78	Heel bone mineral density (BMD) T-score, automated
3086	Speed of sound through heel, manual entry
3085	Heel Broadband ultrasound attenuation (BUA), manual entry
3083	Heel quantitative ultrasound index (QUI), manual entry
3084	Heel bone mineral density (BMD), manual entry
77	Heel bone ultrasound T-score, manual entry


4092	Heel ultrasound method (left)
4095	Heel ultrasound method (right)
4100	Ankle spacing width (left)
4119	Ankle spacing width (right)

4103	Speed of sound through heel (left)
4122	Speed of sound through heel (right)
4142	Speed of sound through heel, manual entry (left)
4147	Speed of sound through heel, manual entry (right)

4141	Heel broadband ultrasound attenuation (BUA), manual entry (left)
4146	Heel broadband ultrasound attenuation (BUA), manual entry (right)
4101	Heel broadband ultrasound attenuation (left)
4120	Heel broadband ultrasound attenuation (right)


4139	Heel quantitative ultrasound index (QUI), manual entry (left)
4144	Heel quantitative ultrasound index (QUI), manual entry (right)
4104	Heel quantitative ultrasound index (QUI), direct entry (left)
4123	Heel quantitative ultrasound index (QUI), direct entry (right)


4140	Heel bone mineral density (BMD), manual entry (left)
4145	Heel bone mineral density (BMD), manual entry (right)
4105	Heel bone mineral density (BMD) (left)
4124	Heel bone mineral density (BMD) (right)


4138	Heel bone mineral density (BMD) T-score, manual entry (left)
4143	Heel bone mineral density (BMD) T-score, manual entry (right)
4106	Heel bone mineral density (BMD) T-score, automated (left)
4125	Heel bone mineral density (BMD) T-score, automated (right)
"""

def read_bone_densitometry_data(**kwargs):
    ## Read first half of the data for instance 0 :
    instance = 0
    age_col = '21003-' + str(instance) + '.0'
    cols_age_eid_sex = ['eid', age_col, '31-0.0']
    d = pd.read_csv(path_data, usecols = cols_age_eid_sex + ['19-0.0', '3146-0.0', '3143-0.0', '3144-0.0', '3147-0.0', '3148-0.0', '78-0.0',
                                     '3086-0.0', '3085-0.0', '3083-0.0', '3084-0.0', '77-0.0'], **kwargs)

    d = d[d['19-0.0'].isin([1, 2])]

    def custom_apply(row):
        method = row['19-0.0']
        cols = ['eid', 'Age when attended assessment centre', 'Sex', 'Ankle spacing width', 'Speed of sound through heel', 'Heel Broadband ultrasound attenuation', 'Heel quantitative ultrasound index (QUI)',
               'Heel bone mineral density (BMD)', 'Heel bone mineral density (BMD) T-score']
        if method == 1:
            values = str(int(row['eid'])), row[age_col], row['31-0.0'], row['3143-0.0'], row['3146-0.0'], row['3144-0.0'], row['3147-0.0'], row['3148-0.0'], row['78-0.0']


        elif method == 2:
            values = str(int(row['eid'])), row[age_col], row['31-0.0'], row['3143-0.0'], row['3086-0.0'], row['3085-0.0'], row['3083-0.0'], row['3084-0.0'], row['77-0.0']

        return pd.Series(values, index = cols)


    d = d.apply(custom_apply, axis = 1)
    d['id'] = d['eid'] + '_' + str(0)
    d['eid'] = d['eid'].astype('int')
    d = d.set_index('id')

    list_df = []
    for instance in range(4):


        age_col = '21003-' + str(instance) + '.0'
        cols_age_eid_sex = ['eid', age_col, '31-0.0']

        cols = [4092, 4095, 4100, 4119, 4103, 4122, 4142, 4147, 4141, 4146, 4101, 4120, 4139, 4144, 4104, 4123, 4140,
               4145, 4105, 4124, 4138, 4143, 4106, 4125]
        cols_instance = [str(elem) + '-%s.0' % instance for elem in cols]
        map_col_to_col_instance = dict(zip(cols, cols_instance))
        raw = pd.read_csv(path_data, usecols = cols_age_eid_sex + cols_instance, **kwargs)
        raw = raw[raw[map_col_to_col_instance[4092]].isin([1, 2]) & raw[map_col_to_col_instance[4095]].isin([1, 2])]

        def custom_apply2(row):
            method_left = row[map_col_to_col_instance[4092]]
            method_right = row[map_col_to_col_instance[4095]]
            if method_left == 1:
                values_left = row[map_col_to_col_instance[4100]], row[map_col_to_col_instance[4103]], row[map_col_to_col_instance[4101]], \
                row[map_col_to_col_instance[4104]], row[map_col_to_col_instance[4105]], row[map_col_to_col_instance[4106]]

            else : #method_left == 2:
                values_left = row[map_col_to_col_instance[4100]], row[map_col_to_col_instance[4142]], row[map_col_to_col_instance[4141]], \
                row[map_col_to_col_instance[4139]], row[map_col_to_col_instance[4140]], row[map_col_to_col_instance[4138]]

            if method_right == 1:
                values_right = row[map_col_to_col_instance[4119]], row[map_col_to_col_instance[4122]], row[map_col_to_col_instance[4120]], \
                row[map_col_to_col_instance[4123]], row[map_col_to_col_instance[4124]], row[map_col_to_col_instance[4125]]

            else:
                values_right = row[map_col_to_col_instance[4119]], row[map_col_to_col_instance[4147]], row[map_col_to_col_instance[4146]], \
                row[map_col_to_col_instance[4144]], row[map_col_to_col_instance[4145]], row[map_col_to_col_instance[4143]]


            values = (np.array(values_left) + np.array(values_right))/2
            cols = ['eid', 'Age when attended assessment centre', 'Sex', 'Ankle spacing width', 'Speed of sound through heel', 'Heel Broadband ultrasound attenuation', 'Heel quantitative ultrasound index (QUI)',
               'Heel bone mineral density (BMD)', 'Heel bone mineral density (BMD) T-score']
            values = [str(int(row['eid'])), row[age_col], row['31-0.0']] + list(values)
            return pd.Series(values, index = cols)

        df_instance = raw.apply(custom_apply2, axis = 1)
        df_instance['id'] = df_instance['eid'] + '_' + str(instance)
        df_instance = df_instance.set_index('id')
        df_instance['eid'] = df_instance['eid'].astype('int')
        list_df.append(df_instance)
    return pd.concat([pd.concat(list_df), d])
