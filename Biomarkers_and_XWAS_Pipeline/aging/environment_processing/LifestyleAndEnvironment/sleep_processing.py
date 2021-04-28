from ..base_processing import read_complex_data

"""
1160	Sleep duration
1170	Getting up in morning
1180	Morning/evening person (chronotype)
1190	Nap during day
1200	Sleeplessness / insomnia
1210	Snoring
1220	Daytime dozing / sleeping (narcolepsy)
"""

def read_sleep_data(instances = [0, 1, 2, 3], **kwargs):

    dict_onehot = {}
    cols_numb_onehot = {}
    cols_ordinal = ['1170', '1180', '1190', '1200', '1210', '1220']
    cols_continuous = ['1160']
    cont_fill_na = []
    cols_half_binary = {'1180' : 2.5, '1210' : 1.5}


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df
