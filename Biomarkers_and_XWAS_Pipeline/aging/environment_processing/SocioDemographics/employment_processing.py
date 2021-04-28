from ..base_processing import read_complex_data

"""
796	Distance between home and job workplace
6142	Current employment status
757	Time employed in main current job
767	Length of working week for main job
777	Frequency of travelling from home to job workplace
6143	Transport type for commuting to job workplace
806	Job involves mainly walking or standing
816	Job involves heavy manual or physical work
826	Job involves shift work
3426	Job involves night shift work

NB I removed all retired people => pb ?
"""

def read_employment_data(instances = [0, 1, 2, 3], **kwargs):
    dict_onehot = {'6142' : {1 : 'In paid employment or self-employed', 2 : 'Retired', 3 : 'Looking after home and/or family',4 : 'Unable to work because of sickness or disability',
                             5 : 'Unemployed', 6 : 'Doing unpaid or voluntary work', 7  : 'Full or part-time student', -7 : 'None of the above', -3 : 'Prefer not to Answer', -1 : 'Do not know', -7 : 'None of the above'},
                   '6143' : {1: 'Car/motor vehicle', 2 : 'Walk', 3 : 'Public transport', 4 : 'Cycle',-3 : 'Prefer not to Answer', -1 : 'Do not know', -7 : 'None of the above'}}
    cols_numb_onehot= {'6142': 7, '6143': 4}
    cols_ordinal = ['796', '757', '767', '777', '806', '816', '826', '3426']
    cols_continuous = []

    cont_fill_na = ['3426',  '757', '767', '777','796', '806', '816', '826']




    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           **kwargs)
    df = df.replace(-10, 0)
    return df


# def read_employment_data(instance = 0, **kwargs):
#
#     dict_onehot = {'6142' : {1	: 'In paid employment or self-employed', 2 : 'Retired', 3 : 'Looking after home and/or family',4 : 'Unable to work because of sickness or disability',
#                              5 : 'Unemployed', 6 : 'Doing unpaid or voluntary work', 7  : 'Full or part-time student', -7 : 'None of the above'},
#                    '6143' : {1: 'Car/motor vehicle', 2 : 'Walk', 3 : 'Public transport', 4 : 'Cycle'}}
#
#     cols_onehot = ['6142', '6143']
#     cols_ordinal = ['796', '757', '767', '777', '806', '816', '826', '3426']
#     cols_continous = []
#     """
#         all cols must be strings or int
#         cols_onehot : cols that need one hot encoding
#         cols_ordinal : cols that need to be converted as ints
#         cols_continuous : cols that don't need to be converted as ints
#     """
#
#     ## Format cols :
#     for idx ,elem in enumerate(cols_onehot):
#         if isinstance(elem,(str)):
#             cols_onehot[idx] = elem + '-%s.0' % instance
#         elif isinstance(elem, (int)):
#             cols_onehot[idx] = str(elem) + '-%s.0' % instance
#
#     for idx ,elem in enumerate(cols_ordinal):
#         if isinstance(elem,(str)):
#             cols_ordinal[idx] = elem + '-%s.0' % instance
#         elif isinstance(elem, (int)):
#             cols_ordinal[idx] = str(elem) + '-%s.0' % instance
#
#     for idx ,elem in enumerate(cols_continous):
#         if isinstance(elem,(str)):
#             cols_continous[idx] = elem + '-%s.0' % instance
#         elif isinstance(elem, (int)):
#             cols_continous[idx] = str(elem) + '-%s.0' % instance
#
#     temp = pd.read_csv(path_data, usecols = ['eid'] + cols_onehot + cols_ordinal + cols_continous, **kwargs).set_index('eid')
#     temp = temp.dropna(how = 'all')
#     temp = temp[temp['6142-%s.0' % instance] != 2]
#     temp['3426-%s.0' % instance] = temp['3426-%s.0' % instance].replace(np.nan, 0)
#     temp['757-%s.0' % instance] = temp['757-%s.0' % instance].replace(-10, 0)
#     temp['796-%s.0' % instance] = temp['796-%s.0' % instance].replace(-10, 0)
#     temp['777-%s.0' % instance] = temp['777-%s.0' % instance].replace(-10, 0)
#     temp = temp[~(temp < 0).any(axis = 1)]
#     temp = temp.dropna(how = 'any')
#
#     #757
#     #777
#     #Time employed in main current job.0
#
#     for column in cols_onehot + cols_ordinal:
#         temp[column] = temp[column].astype('Int64')
#
#     for cate in cols_onehot:
#         col_ = temp[cate]
#         d = pd.get_dummies(col_)
#         try :
#             d = d.drop(columns = [-3])
#         except KeyError:
#             d = d
#
#         d.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d.columns if int(elem) >= 0]
#         temp = temp.drop(columns = [cate[:-2] + '.0'])
#         temp = temp.join(d, how = 'inner')
#
#     df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
#     df_features.set_index('FieldID', inplace = True)
#     feature_id_to_name = df_features.to_dict()['Field']
#
#
#     features_index = temp.columns
#     features = []
#     for elem in features_index:
#         split = elem.split('-%s' % instance)
#         features.append(feature_id_to_name[int(split[0])] + split[1])
#     temp.columns = features
#
#
#     return temp
