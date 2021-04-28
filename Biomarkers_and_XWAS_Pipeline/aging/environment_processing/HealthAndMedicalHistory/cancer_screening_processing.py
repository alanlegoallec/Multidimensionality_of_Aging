from ..base_processing import read_complex_data

"""
2345	Ever had bowel cancer screening
2355	Most recent bowel cancer screening
2365	Ever had prostate specific antigen (PSA) test
3809	Time since last prostate specific antigen (PSA) test
"""

def read_cancer_screening_data(instances = [0, 1, 2, 3], **kwargs):
    dict_onehot = {}
    cols_numb_onehot = {}
    cols_ordinal = ['2345', '2365']
    cols_continuous = ['2355', '3809']
    cont_fill_na = {'2355' : 100,  '3809' : 100}
    cols_half_binary = []


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    df = df.replace(-10, 0)
    return df
