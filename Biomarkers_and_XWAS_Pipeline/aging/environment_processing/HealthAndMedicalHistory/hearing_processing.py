from ..base_processing import read_complex_data

"""
2247	Hearing difficulty/problems
2257	Hearing difficulty/problems with background noise
3393	Hearing aid user
4792	Cochlear implant
4803	Tinnitus
4814	Tinnitus severity/nuisance => Modify encoding
4825	Noisy workplace => modify encoding
4836	Loud music exposure frequency => modify encoding
"""

def read_hearing_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot = {'4803' : {11 : 'Yes, now most or all of the time', 12 : 'Yes, now a lot of the time', 13 : 'Yes, now some of the time',
                             14 : 'Yes, but not now, but have in the past', 0 :'No, never', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   }
    cols_numb_onehot = {'4803' : 1}
    cols_ordinal = ['2247', '2257', '3393', '4792', '4814', '4825', '4836']
    cols_continuous = []
    cont_fill_na = ['4814']
    cols_half_binary = ['2247', '2257']


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)

    ## RE-ENCODE :
    df['Tinnitus severity/nuisance.0'] = df['Tinnitus severity/nuisance.0'].replace(4, 0).replace(13, 1).replace(12, 2).replace(11, 3)
    df['Noisy workplace.0'] = df['Noisy workplace.0'].replace(13, 1).replace(12, 2).replace(11, 3)
    df['Loud music exposure frequency.0'] = df['Loud music exposure frequency.0'].replace(13, 1).replace(12, 2).replace(11, 3)
    return df
