from ..base_processing import read_complex_data
"""
1110	Length of mobile phone use
1120	Weekly usage of mobile phone in last 3 months
1130	Hands-free device/speakerphone use with mobile phone in last 3 month
1140	Difference in mobile phone use compared to two years previously
1150	Usual side of head for mobile phone use
2237	Plays computer games
"""


def read_electronic_devices_data(instances = [0, 1, 2, 3], **kwargs):

    cont_fill_na = ['1120', '1130', '1140']
    cols_ordinal = ['1110']
    cols_continuous = ['1120', '1130', '1140', '2237']

    dict_onehot = {'1150' : {1 : 'Left', 2 :'Right', 3: 'Equally left and right', -3 : 'Prefer not to answer', -1 : 'Do not know'}}
    cols_numb_onehot = {'1150' : 1}


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           **kwargs)
    return df
