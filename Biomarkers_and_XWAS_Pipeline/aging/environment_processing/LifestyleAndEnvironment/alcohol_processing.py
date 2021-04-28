from ..base_processing import read_complex_data

"""
20117	Alcohol drinker status
1558	Alcohol intake frequency.

4407	Average monthly red wine intake
4418	Average monthly champagne plus white wine intake
4429	Average monthly beer plus cider intake
4440	Average monthly spirits intake
4451	Average monthly fortified wine intake
4462	Average monthly intake of other alcoholic drinks
1568	Average weekly red wine intake
1578	Average weekly champagne plus white wine intake
1588	Average weekly beer plus cider intake
1598	Average weekly spirits intake
1608	Average weekly fortified wine intake
5364	Average weekly intake of other alcoholic drinks

1618	Alcohol usually taken with meals
1628	Alcohol intake versus 10 years previously

2664	Reason for reducing amount of alcohol drunk
3859	Reason former drinker stopped drinking alcohol
"""

def read_alcohol_data(instances = [0, 1, 2, 3], **kwargs):

    dict_onehot = {'2664'  : {0 : 'Do not reduce', 1 : 'Illness or ill health', 2: "Doctor's advice", 3 : "Health precaution", 4 : "Financial reasons",
                                                                     5 : "Other reason", -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '3859' : {0 : 'Do not reduce', 1 : 'Illness or ill health', 2: "Doctor's advice", 3 : "Health precaution", 4 : "Financial reasons",
                                                                     5 : "Other reason", -1 : 'Do not know', -3 : 'Prefer not to answer'}}
    cols_numb_onehot = {'2664' : 1, '3859' : 1}
    cols_ordinal = ['20117', '1558'] + ['1618']
    cols_continuous = ['4407', '4418', '4429', '4440', '4451', '4462', '1568', '1578', '1588', '1598', '1608', '5364']
    cont_fill_na = ['1618'] + cols_continuous
    cols_half_binary = []

    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)

    ## RE-ENCODE :
    df['Alcohol usually taken with meals.0'] = df['Alcohol usually taken with meals.0'].replace(-6, 0.5)
    return df
