from ..base_processing import read_complex_data

"""
    6146	Attendance/disability/mobility allowance
    4674	Private healthcare
"""

def read_other_sociodemographics_data(instances = [0, 1, 2, 3], **kwargs):
    dict_onehot = {'6146'  : {1 : 'Attendance allowance', 2 :  'Disability living allowance', 3 : 'Blue badge', -7 :'None of the above',
                              -1 : 'Do not know', -3 : 'Prefer not to answer'}}
    cols_numb_onehot = {'6146' : 3}
    cols_ordinal = ['4674']
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
