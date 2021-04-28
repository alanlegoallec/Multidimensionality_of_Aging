from ..base_processing import read_complex_data
"""
2139	Age first had sexual intercourse
2149	Lifetime number of sexual partners
2159	Ever had same-sex intercourse
3669	Lifetime number of same-sex sexual partners
"""


def read_sexual_factors_data(instances = [0, 1, 2, 3], **kwargs):

    dict_onehot = {}
    cols_numb_onehot = {}
    cols_ordinal = ['2129', '2139', '2149', '2159', '3669']
    cols_continuous = []
    cont_fill_na = ['3669', '2149', '2159']
    cols_half_binary = []


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    df = df[df['Answered sexual history questions.0'] == 1].drop(columns = ['Answered sexual history questions.0'])
    return df
