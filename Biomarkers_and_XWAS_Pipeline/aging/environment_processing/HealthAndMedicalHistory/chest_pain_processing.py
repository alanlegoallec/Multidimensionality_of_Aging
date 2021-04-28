from ..base_processing import read_complex_data

"""
2335	Chest pain or discomfort
3606	Chest pain or discomfort walking normally
3616	Chest pain due to walking ceases when standing still
3751	Chest pain or discomfort when walking uphill or hurrying

"""

def read_chest_pain_data(instances = [0, 1, 2, 3], **kwargs):
    dict_onehot = {}
    cols_numb_onehot = {}
    cols_ordinal = ['2335', '3606', '3616', '3751']
    cols_continuous = []
    cont_fill_na = ['3606', '3616', '3751']
    cols_half_binary = {'2335' : 0.5, '3616' : 0.5, '3606' : 2, '3751' : 2}


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df
