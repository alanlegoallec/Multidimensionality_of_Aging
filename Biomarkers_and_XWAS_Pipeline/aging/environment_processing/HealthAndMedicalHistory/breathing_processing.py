from ..base_processing import read_complex_data

"""
2316	Wheeze or whistling in the chest in last year
4717	Shortness of breath walking on level ground

"""
def read_breathing_data(instances = [0, 1, 2, 3], **kwargs):

    dict_onehot = {}

    cols_numb_onehot = {}
    cols_ordinal = ['2316', '4717']
    cols_continuous = []
    cont_fill_na = []
    cols_half_binary = ['2316', '4717']


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df
