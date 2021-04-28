from ..base_processing import read_complex_data

"""
20160	Ever smoked : All
20162	Pack years adult smoking as proportion of life span exposed to smoking ? : 160k
20161	Pack years of smoking ? : 160k
20116	Smoking status : All
1239	Current tobacco smoking
1249	Past tobacco smoking
3436	Age started smoking in current smokers
2867	Age started smoking in former smokers
3456	Number of cigarettes currently smoked daily (current cigarette smokers)
3466	Time from waking to first cigarette
3476	Difficulty not smoking for 1 day
1259	Smoking/smokers in household
1269	Exposure to tobacco smoke at home
1279	Exposure to tobacco smoke outside home
"""

def read_smoking_data(instances = [0, 1, 2, 3], **kwargs):

        dict_onehot = {
            '20116' : {-3 : 'Prefer not to answer', 0 : 'Never', 1 : 'Previous', 2 : 'Current'},
            '1239' : {1 : 'Yes, on most or all days', 2 : 'Only occasionally', 0 : 'No', -3 : 'Prefer not to answer'},
            '1249' : {1 : 'Smoked on most or all days', 2 : 'Smoked occasionally', 3 : 'Just tried once or twice',
                      4 : 'I have never smoked', -3 : 'Prefer not to answer'},
            #'2644' : {1 : 'Yes', 0 : 'No', -1 : 'Do not know', -3 : 'Prefer not to answer'},
            #'3446' : {1 : 'Manufactured cigarettes', 2 : 'Hand-rolled cigarettes', 3 : 'Cigars or pipes', -7 : 'None of the above',
            #          -3 : 'Prefer not to answer'},
            #'3486' : {1 : 'Yes, tried but was not able to stop or stopped for less than 6 months', 2 : 'Yes, tried and stopped for at least 6 months',
            #          0 : 'No', -3 : 'Prefer not to answer'},
            #'6157' : {1 : 'Illness or ill health', 2 : "Doctor's advice",3 : 'Health precaution', 4 : 'Financial reasons', -7 : 'None of the above',
            #         -1 : 'Do not know', -3 : 'Prefer not to answer'},
            #'2936' : {1 : 'Yes, definitely', 2 : 'Yes, probably', 3 : 'No, probably not', 4 : 'No, definitely not', -1 : 'Do not know', -3 : 'Prefer not to answer'}
                      }

        cols_numb_onehot = {'20116' : 1, '1239' : 1, '1249' : 1}
        cols_ordinal = ['20160', '3466', '3476', '1259']
        cols_continuous = ['20161', '20162', '3436', '3456', '2867', '1269', '1279']
        cont_fill_na = {'3456' : 0, '2867' : 90, '3466' : 6, '20161' : 0, '20162' : 0, '3436' : 90, '3476' : 0}
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
