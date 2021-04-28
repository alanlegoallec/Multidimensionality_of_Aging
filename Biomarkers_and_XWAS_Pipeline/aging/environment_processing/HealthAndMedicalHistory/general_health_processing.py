from ..base_processing import read_complex_data

""" 2178	Overall health rating
    2188	Long-standing illness, disability or infirmity
    2296	Falls in the last year
    2306	Weight change compared with 1 year ago
"""

def read_general_health_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot = {'2306' : {0 : 'No - weigh about the same', 2 : 'Yes - gained weight', 3 : 'Yes - lost weight',
                             -1 : 'Do not know', -3 : 'Prefer not to answer'}}

    cols_numb_onehot = {'2306' : 1}
    cols_ordinal = ['2188', '2296']
    cols_continuous = ['2178']
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
