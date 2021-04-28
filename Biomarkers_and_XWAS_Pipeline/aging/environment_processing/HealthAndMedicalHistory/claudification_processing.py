from ..base_processing import read_complex_data

"""
4728	Leg pain on walking
5452	Leg pain when standing still or sitting
5463	Leg pain in calf/calves
5474	Leg pain when walking uphill or hurrying
5485	Leg pain when walking normally
5496	Leg pain when walking ever disappears while walking
5507	Leg pain on walking : action taken
5518	Leg pain on walking : effect of standing still
5529	Surgery on leg arteries (other than for varicose veins)
5540	Surgery/amputation of toe or leg

"""

def read_claudication_data(instances = [0, 1, 2, 3], **kwargs):
    dict_onehot = {'5518' : {1 : 'Pain usually continues for more than 10 minutes', 2 : 'Pain usually disappears in less than 10 minutes', -1 : 'Do not know', -3 : 'Prefer not to answer'}}
    cols_numb_onehot = {'5518' : 1}
    cols_ordinal = ['4728', '5452', '5463', '5474', '5485', '5496', '5507', '5529', '5540']
    cols_continuous = []
    cont_fill_na = ['5452', '5463', '5474', '5485', '5496', '5507', '5529', '5540']
    cols_half_binary = ['5452', '5463', '5474', '5485', '5496']


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df
