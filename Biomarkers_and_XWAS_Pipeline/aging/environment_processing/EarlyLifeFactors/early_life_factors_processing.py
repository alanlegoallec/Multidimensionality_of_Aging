from ..base_processing import read_complex_data

def read_early_life_factors_data(**kwargs):
    instances = [0, 1, 2]
    """
    1647	Country of birth (UK/elsewhere)
    1677	Breastfed as a baby
    1687	Comparative body size at age 10
    1697	Comparative height size at age 10
    1707	Handedness (chirality/laterality)
    1767	Adopted as a child
    1777	Part of a multiple birth
    1787	Maternal smoking around birth
    """
    dict_onehot = {'1647' : {1 : 'England', 2 : 'Wales', 3 : 'Scotland', 4 : 'Northern Ireland', 5 : 'Republic of Ireland', 6 : 'Elsewhere', -1 : 'Do not know',
                             -3 : 'Prefer not to answer' },
                   '1677' : {1 : 'Yes', 0 : 'No', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1687' : {1 : 'Thinner', 2 : 'Plumper', 3 : 'About average', -1 :'Do not know', -3 : 'Prefer not to answer'},
                   '1697' : {1 : 'Shorter', 2 : 'Taller', 3 : 'About average', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1707' : {1 : 'Right-handed', 2 : 'Left-handed', 3 : 'Use both right and left hands equally', -3 : 'Prefer not to answer'},
                   '1787' : {1 : 'Yes', 0 : 'No', -1 : 'Do not know', -3 : 'Prefer not to answer'}}

    cols_numb_onehot = {'1647' : 1,
                        '1677' : 1,
                        '1687' : 1,
                        '1697' : 1,
                        '1707' : 1,
                        '1787' : 1}
    cols_ordinal = ['1767', '1777']
    cols_continuous = []
    cont_fill_na = []


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           **kwargs)

    return df
