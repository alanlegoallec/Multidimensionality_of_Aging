from ..base_processing import read_complex_data
"""
670	Type of accommodation lived in
680	Own or rent accommodation lived in
6139	Gas or solid-fuel cooking/heating
6140	Heating type(s) in home
699	Length of time at current address
709	Number in household
6141	How are people in household related to participant
728	Number of vehicles in household
738	Average total household income before tax
"""
def read_household_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot = {'670' : {1 : 'A house or bungalow', 2 : 'A flat, maisonette or apartment', 3 : 'Mobile or temporary structure (i.e. caravan)',  4 : 'Sheltered accommodation',
                            5 : 'Care home', -7 : 'None of the above', -3 : 'Prefer not to answer', -1: 'Do not know'},
                   '680' : {1 : 'Own outright (by you or someone in your household)', 2 : 'Own with a mortgage', 3 : 'Rent - from local authority, local council, housing association', 4 : 'Rent - from private landlord or letting agency',
                            5 : 'Pay part rent and part mortgage (shared ownership)', 6 : 'Live in accommodation rent free', -7 : 'None of the above', -3 : 'Prefer not to answer', -1: 'Do not know'},
                   '6139' : {1 : 'A gas hob or gas cooker', 2 : 'A gas fire that you use regularly in winter time', 3 : 'An open solid fuel fire that you use regularly in winter time', -7 : 'None of the above', -3 : 'Prefer not to answer', -1: 'Do not know'},
                   '6140' : {1 : 'Gas central heating', 2 : 'Electric storage heaters', 3 : 'Oil (kerosene) central heating', 4 : 'Portable gas or paraffin heaters', 5 : 'Solid fuel central heating', 6 : 'Open fire without central heating',
                            -7 : 'None of the above', -3 : 'Prefer not to answer', -1: 'Do not know'},
                   '6141' : {1: 'Husband, wife or partner', 2 : 'Son and/or daughter (include step-children)', 3 : 'Brother and/or sister', 4 : 'Mother and/or father' ,
                             5 : 'Grandparent', 6 : 'Grandchild', 7 : 'Other related', 8 : 'Other unrelated', -3 : 'Prefer not to answer', -1: 'Do not know'}}

    cols_numb_onehot = {'670' : 1, '680': 1, '6139' : 3, '6140': 3, '6141' : 5}
    cols_ordinal = ['699', '709', '728', '738']
    cols_continuous = []
    cont_fill_na = ['709', '728', '738']


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           **kwargs)


    df = df.replace(-10, 0)
    return df


# def read_household_data(instance = 0, **kwargs):
#
#     dict_onehot = {'670' : {1 : 'A house or bungalow', 2 : 'A flat, maisonette or apartment', 3 : 'Mobile or temporary structure (i.e. caravan)',  4 : 'Sheltered accommodation',
#                             5 : 'Care home', -7 : 'None of the above'},
#                    '680' : {1 : 'Own outright (by you or someone in your household)', 2 : 'Own with a mortgage', 3 : 'Rent - from local authority, local council, housing association', 4 : 'Rent - from private landlord or letting agency',
#                             5 : 'Pay part rent and part mortgage (shared ownership)', 6 : 'Live in accommodation rent free', -7 : 'None of the above'},
#                    '6139' : {1 : 'A gas hob or gas cooker', 2 : 'A gas fire that you use regularly in winter time', 3 : 'An open solid fuel fire that you use regularly in winter time', -7 : 'None of the above'},
#                    '6140' : {1 : 'Gas central heating', 2 : 'Electric storage heaters', 3 : 'Oil (kerosene) central heating', 4 : 'Portable gas or paraffin heaters', 5 : 'Solid fuel central heating', 6 : 'Open fire without central heating',
#                             -7 : 'None of the above'},
#                    '6141' : {1: 'Husband, wife or partner', 2 : 'Son and/or daughter (include step-children)', 3 : 'Brother and/or sister', 4 : 'Mother and/or father' ,
#                              5 : 'Grandparent', 6 : 'Grandchild', 7 : 'Other related', 8 : 'Other unrelated'}}
#
#     cols_onehot = ['670', '680', '6139', '6140', '6141']
#     cols_ordinal = ['699', '709', '728', '738']
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
#
#     for column in cols_onehot + cols_ordinal:
#         temp[column] = temp[column].astype('Int64')
#
#
#     for cate in cols_onehot:
#         col_ = temp[cate]
#         d = pd.get_dummies(col_)
#         try :
#             d = d.drop(columns = [-3])
#         except KeyError:
#             d = d
#         try :
#             d = d.drop(columns = [-1])
#         except KeyError:
#             d = d
#         print(d)
#         d.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d.columns]
#         temp = temp.drop(columns = [cate[:-2] + '.0'])
#         temp = temp.join(d, how = 'inner')
#
#     temp['699-%s.0' % instance] = temp['699-%s.0' % instance].replace(-10, 0)
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
#     temp = temp[~(temp < 0).any(axis = 1)]
#
#     return temp
