from ..base_processing import read_complex_data

"""
6159	Pain type(s) experienced in last month
3799	Headaches for 3+ months
4067	Facial pains for 3+ months
3404	Neck/shoulder pain for 3+ months
3571	Back pain for 3+ months
3741	Stomach/abdominal pain for 3+ months
3414	Hip pain for 3+ months
3773	Knee pain for 3+ months
2956	General pain for 3+ months
"""

def read_general_pain_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot = {'6159' : {1 :'Headache', 2 : 'Facial pain', 3 : 'Neck or shoulder pain', 4 : 'Back pain',
                             5 : 'Stomach or abdominal pain', 6 : 'Hip pain', 7 :'Knee pain', 8 : 'Pain all over the body',
                             -7 : 'None of the above', -3 : 'Prefer not to answer'},

                  }
    cols_numb_onehot = {'6159' : 7}
    cols_ordinal = ['3799', '4067', '3404', '3571', '3741', '3414', '3773', '2956']
    cols_continuous = []
    cont_fill_na = ['3799', '4067', '3404', '3571', '3741', '3414', '3773', '2956']
    cols_half_binary = ['3799', '4067', '3404', '3571', '3741', '3414', '3773', '2956']


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df
