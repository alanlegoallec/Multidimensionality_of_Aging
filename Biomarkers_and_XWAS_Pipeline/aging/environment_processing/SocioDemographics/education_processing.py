from ..base_processing import read_complex_data

"""
6138 Qualifications
845	Age completed full time education
"""

def read_education_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot  = {'6138' : {1 : 'College or University degree', 2 : 'A levels/AS levels or equivalent', 3 : 'O levels/GCSEs or equivalent',
                              4 : 'CSEs or equivalent', 5 : 'NVQ or HND or HNC or equivalent', 6 : 'Other professional qualifications eg: nursing, teaching',
                             -7 : 'None of the above', -3 : 'Prefer not to answer'}}

    cols_numb_onehot = {'6138' : 6}



    cols_continuous = []
    cols_ordinal = []
    cont_fill_na = []



    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           **kwargs)
    return df


# def read_education_data(instance = 0, **kwargs):
#
#     dict_onehot  = {'6138' : {1 : 'College or University degree', 2 : 'A levels/AS levels or equivalent', 3 : 'O levels/GCSEs or equivalent',
#                               4 : 'CSEs or equivalent', 5 : 'NVQ or HND or HNC or equivalent', 6 : 'Other professional qualifications eg: nursing, teaching',
#                              -7 : 'None of the above'}}
#
#     cols_onehot = ['6138-%s.%s' % (instance, int_)  for int_ in range(6)]
#     cols_ordinal = ['845']
#     cols_continous = []
#     """
#         all cols must be strings or int
#         cols_onehot : cols that need one hot encoding
#         cols_ordinal : cols that need to be converted as ints
#         cols_continuous : cols that don't need to be converted as ints
#     """
#
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
#
#     temp = temp[~(temp < 0).any(axis = 1)]
#     temp = temp.dropna(how = 'any')
#
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
#         d.columns = [cate[:-2] + '.' + dict_onehot[cate[:-4]][int(elem)] for elem in d.columns]
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



# def read_education_data(instances = [0, 1, 2, 3], **kwargs):
#     df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
#     df_features.set_index('FieldID', inplace = True)
#     feature_id_to_name = df_features.to_dict()['Field']
#
#     dict_onehot  = {'6138' : {1 : 'College or University degree', 2 : 'A levels/AS levels or equivalent', 3 : 'O levels/GCSEs or equivalent',
#                               4 : 'CSEs or equivalent', 5 : 'NVQ or HND or HNC or equivalent', 6 : 'Other professional qualifications eg: nursing, teaching',
#                              -7 : 'None of the above', -3 : 'Prefer not to answer'}}
#
#     cols_numb_onehot = {'6138' : 6}
#
#
#     """
#         all cols must be strings or int
#         cols_onehot : cols that need one hot encoding
#         cols_ordinal : cols that need to be converted as ints
#         cols_continuous : cols that don't need to be converted as ints
#     """
#
#
#     for idx_instance, instance in enumerate(instances) :
#
#
#         cols_onehot = [ key + '-%s.%s' % (instance, int_) for key in cols_numb_onehot.keys() for int_ in range(cols_numb_onehot[key])]
#         cols_ordinal = []
#         cols_continuous = []
#
#         for idx ,elem in enumerate(cols_ordinal):
#             if isinstance(elem,(str)):
#                 cols_ordinal[idx] = elem + '-%s.0' % instance
#             elif isinstance(elem, (int)):
#                 cols_ordinal[idx] = str(elem) + '-%s.0' % instance
#
#         for idx ,elem in enumerate(cols_continuous):
#             if isinstance(elem,(str)):
#                 cols_continuous[idx] = elem + '-%s.0' % instance
#             elif isinstance(elem, (int)):
#                 cols_continuous[idx] = str(elem) + '-%s.0' % instance
#
#         temp = pd.read_csv(path_data, usecols = ['eid'] + cols_onehot + cols_ordinal + cols_continuous, **kwargs).set_index('eid')
#         temp = temp.dropna(how = 'all')
#         for column in cols_onehot + cols_ordinal:
#             temp[column] = temp[column].astype('Int32')
#
#
#         for col in cols_numb_onehot.keys():
#             for idx in range(cols_numb_onehot[col]):
#                 cate = col + '-%s.%s' % (instance, idx)
#                 d = pd.get_dummies(temp[cate])
#                 d.columns = [col + '-%s'%instance + '.' + dict_onehot[col][int(elem)] for elem in d.columns ]
#                 temp = temp.drop(columns = [cate])
#
#                 if idx == 0:
#                     d_ = d
#                 else :
#                     common_cols = d.columns.intersection(d_.columns)
#                     remaining_cols = d.columns.difference(common_cols)
#                     if len(common_cols) > 0 :
#                         d_[common_cols] = d_[common_cols].add(d[common_cols])
#                     for col_ in remaining_cols:
#                         d_[col_] = d[col_]
#             temp = temp.join(d_, how = 'inner')
#
#
#         features_index = temp.columns
#         features = []
#         for elem in features_index:
#             split = elem.split('-%s' % instance)
#             features.append(feature_id_to_name[int(split[0])] + split[1])
#         temp.columns = features
#
#         temp['eid'] = temp.index
#         temp.index = (temp.index.astype('str') + '_' + str(instance)).rename('id')
#         if idx_instance == 0 :
#             df = temp
#         else :
#             df = df.append(temp.reindex(df.columns, axis = 1, fill_value=0))
#
#     df = df.replace(-1, np.nan)
#     df = df.replace(-3, np.nan)
#     return df
