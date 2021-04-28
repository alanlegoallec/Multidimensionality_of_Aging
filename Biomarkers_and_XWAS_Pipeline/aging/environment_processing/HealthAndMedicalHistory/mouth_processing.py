from ..base_processing import read_complex_data
"""
6149 Mouth/teeth dental problems
"""
def read_mouth_teeth_data(instances = [0, 1, 2, 3], **kwargs):
    dict_onehot = {'6149' : {1 : 'Mouth ulcers', 2 : 'Painful gums', 3 : 'Bleeding gums', 4 : 'Loose teeth', 5 : 'Toothache',6 :'Dentures',-7 : 'None of the above',
                            -3 : 'Prefer not to answer'}}

    cols_numb_onehot = {'6149' : 6}
    cols_ordinal = []
    cols_continuous = []
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
