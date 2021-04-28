import numpy as np
import pandas as pd
from .base_processing import path_data


"""
## Multiple features
4237	Time to press 'next' (left)
4232	Triplet correct (left)
4233	Mean signal-to-noise ratio (SNR), (left)
4230	Signal-to-noise-ratio (SNR) of triplet (left)
4247	Time to press 'next' (right)
4243	Triplet correct (right)
4244	Mean signal-to-noise ratio (SNR), (right)
4241	Signal-to-noise-ratio (SNR) of triplet (right)

## Single dimension feature
4270	Volume level set by participant (left)
4277	Volume level set by participant (right)
20019	Speech-reception-threshold (SRT) estimate (left)
20021	Speech-reception-threshold (SRT) estimate (right)
4272	Duration of hearing test (left)
4279	Duration of hearing test (right)

# Single but to remove :
4268	Completion status (left)
4275	Completion status (right)
4276	Number of triplets attempted (right)
4269	Number of triplets attempted (left)
"""

def read_hearing_test_data(**kwargs):
    cols_names_simple = ['Volume level set by participant (left)', 'Volume level set by participant (right)', 'Speech-reception-threshold (SRT) estimate (left)',
                         'Speech-reception-threshold (SRT) estimate (right)', 'Duration of hearing test (left)', 'Duration of hearing test (right)',
                         'Completion status (left)', 'Completion status (right)', 'Number of triplets attempted (right)', 'Number of triplets attempted (left)']

    cols_name_multiple = ["Time to press next (left)", 'Triplet correct (left)', 'Mean signal-to-noise ratio (SNR), (left)',
                          "Signal-to-noise-ratio (SNR) of triplet (left)", "Time to press next (right)", "Triplet correct (right)",
                          'Mean signal-to-noise ratio (SNR), (right)', 'Signal-to-noise-ratio (SNR) of triplet (right)']

    list_df = []
    for instance in range(4):
        age_col = '21003-' + str(instance) + '.0'
        cols_age_eid_sex = ['eid', age_col, '31-0.0']
        cols_single = ['4270', '4277', '20019', '20021', '4272', '4279', '4268', '4275', '4276', '4269']
        cols_single = [elem + '-%s.0' % instance for elem in cols_single]
        map_names_single = dict(zip(cols_single, cols_names_simple))

        cols_multiple = ['4237', '4232', '4233', '4230', '4247', '4243', '4244', '4241']
        cols_names_instance_multiple = [elem + '.%s' % num_triplet for elem in cols_name_multiple for num_triplet in range(1, 16)]
        cols_multiple = [elem + '-%s.%s' % (instance, num_triplet) for elem in cols_multiple for num_triplet in range(1, 16)]
        map_names_multiple = dict(zip(cols_multiple, cols_names_instance_multiple))
        d = pd.read_csv(path_data, usecols = cols_age_eid_sex + cols_single + cols_multiple, **kwargs)
        d = d[(d['4268-%s.0' % instance] == 2) & (d['4275-%s.0' % instance] == 2)]
        d.index = d['eid'].astype(str) + '_%s' % instance
        d.index.names = ['id']

        dict_names = dict(**map_names_single, **map_names_multiple)
        dict_names['31-0.0'] = 'Sex'
        dict_names[age_col] = 'Age when attended assessment centre'
        d = d.rename(dict_names, axis = 1)
        d = d.drop(columns = ['Completion status (left)', 'Completion status (right)', 'Number of triplets attempted (right)', 'Number of triplets attempted (left)'])
        list_df.append(d)
        
    return pd.concat(list_df)
