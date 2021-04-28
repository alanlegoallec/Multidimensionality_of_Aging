from ..base_processing import read_complex_data
"""
1050	Time spend outdoors in summer
1060	Time spent outdoors in winter
1717	Skin colour
1727	Ease of skin tanning
1737	Childhood sunburn occasions
1747	Hair colour (natural, before greying)
1757	Facial ageing
2267	Use of sun/uv protection
2277	Frequency of solarium/sunlamp use
"""

def read_sun_exposure_data(instances = [0, 1, 2], **kwargs):
    dict_onehot = {'1747' : {1 : 'Blonde', 2 : 'Red', 3 : 'Light brown', 4 : 'Dark brown', 5 : 'Black',
                             6 : 'Other', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1757' : {1 : 'Younger than you are', 2 : 'Older than you are', 3 :'About your age', -1 : 'Do not know', -3 : 'Prefer not to answer'}}

    cols_numb_onehot = {'1747' : 1, '1757' : 1}
    cols_ordinal = ['1050', '1060', '1717', '1727', '1737', '2267', '2277']
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
    df = df.replace(-10, 0)
    return df
