from .base_processing import path_data, path_dictionary
import pandas as pd

"""
Features used :
	100020 - FVC, FEV1, PEF
	Errors features : None
	Missing : None
"""

def read_spirometry_data(**kwargs):
	## deal with kwargs
	nrows = None
	if 'nrows' in kwargs.keys():
		nrows = kwargs['nrows']

	## Create feature name dict
	df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
	df_features.set_index('FieldID', inplace = True)
	feature_id_to_name = df_features.to_dict()['Field']

	list_df = []
	for instance in [0, 1, 2, 3]:
		def custom_apply(row, instance):

			flag_0 = row['3061-%s.0' % instance]
			flag_1 = row['3061-%s.1' % instance]
			flag_2 = row['3061-%s.2' % instance]
			if flag_0 == 0 or flag_0 == 32:
				return pd.Series(row[['3064-%s.0' % instance, '3062-%s.0' % instance, '3063-%s.0' % instance] + ['21003-%s.0' % instance, '31-0.0']].values)
			else:
				if flag_1 == 0 or flag_1 == 32:
					return pd.Series(row[['3064-%s.1' % instance, '3062-%s.1' % instance, '3063-%s.1' % instance] + ['21003-%s.0'% instance, '31-0.0']].values)
				else :
					if flag_2 == 0 or flag_2 == 32:
						return pd.Series(row[['3064-%s.2' % instance, '3062-%s.2' % instance, '3063-%s.2'% instance] + ['21003-%s.0'% instance, '31-0.0']].values)
					else:
						return  pd.Series([None, None, None] + ['21003-%s.0'% instance, '31-0.0'])


		cols = ['3064-%s.' % instance, '3062-%s.' % instance, '3063-%s.' % instance, '3061-%s.' % instance]
		temp = pd.read_csv(path_data,  nrows = nrows, usecols =  [elem + str(int_) for elem in cols for int_ in range(3)] +  ['eid', '21003-%s.0'% instance, '31-0.0']).set_index('eid')
		temp.index = temp.index.rename('id')
		temp = temp.apply(lambda row : custom_apply(row, instance = instance), axis = 1)
		df = temp[~temp.isna().any(axis = 1)]
		df.columns = ['3064-%s.0'% instance, '3062-%s.0'% instance, '3063-%s.0'% instance] +  ['21003-%s.0'% instance, '31-0.0']

		features_index = df.columns
		features = []
		for elem in features_index:
			if elem != '21003-%s.0'% instance and elem != '31-0.0':
				features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
			else:
				features.append(feature_id_to_name[int(elem.split('-')[0])])
		df.columns =  features

		df['eid'] = df.index
		df.index = df.index.astype('str') + '_' + str(instance)
		list_df.append(df)

	return pd.concat(list_df)
