from ..base_processing import read_complex_data

"""
1031	Frequency of friend/family visits
6160	Leisure/social activities
2110	Able to confide

"""

def read_social_support_data(instances = [0, 1, 2, 3], **kwargs):
    dict_onehot = {'6160' : {1 : 'Sports club or gym',  2 : 'Pub or social club', 3 : 'Religious group', 4 : 'Adult education class', 5 : 'Other group activity',
                            -7 : 'None of the above', -3 :'Prefer not to answer'}}

    cols_ordinal = ['2110']
    cols_numb_onehot = {'6160' : 5}
    cols_continuous = ['1031']
    cont_fill_na = []
    cols_half_binary = []


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df
