import pandas as pd
from .base_processing import path_data, path_dictionary, read_data
from ..environment_processing.base_processing import read_complex_data
"""
Features used :
	100014 - Autorefraction
	Errors features : None
	Missing : None
"""
def read_eye_data(instances = [0, 1], **kwargs):
	a = read_eye_acuity_data(instances, **kwargs)
	b = read_eye_intraocular_pressure_data(**kwargs)
	c = read_eye_autorefraction_data(**kwargs)
	ab = a.join(b, rsuffix = '_del', lsuffix = '', how = 'outer')
	ab = ab[ab.columns[~ab.columns.str.contains('_del')]]
	abc = ab.join(c, rsuffix = '_del', lsuffix = '', how = 'outer')
	abc = abc[abc.columns[~abc.columns.str.contains('_del')]]
	return abc

def read_eye_acuity_data(instances = [0, 1], **kwargs):

    dict_onehot = {'6075' : {0 : 'none', 1 : 'wearing', 2 : 'elsewhere'},
                   '6074' : {0 : 'none', 1 : 'wearing', 2 : 'elsewhere'}}

    cols_numb_onehot = {'6074' : 1, '6075' : 1}
    cols_ordinal = []
    cols_continuous = ['5208', '5201']
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
    return df



def read_eye_intraocular_pressure_data(**kwargs):
	cols_features =  ['5264', '5256', '5265', '5257', '5262', '5254', '5263', '5255']
	cols_filter = []
	instance = [0, 1]
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_eye_autorefraction_data(**kwargs):
	## deal with kwargs


	nrows = None
	if 'nrows' in kwargs.keys():
		nrows = kwargs['nrows']

	## Create feature name dict
	df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
	df_features.set_index('FieldID', inplace = True)
	feature_id_to_name = df_features.to_dict()['Field']

	## create features to extract for each category : left, right and 6mm, 3mm, keratometry
	l6 = sorted(['5113','5118','5134','5102','5097']) # missing 5105
	r6 = sorted(['5114','5117','5106','5133','5101','5098'])
	l3 = sorted(['5108','5159','5115','5116','5160','5107','5132','5100','5099'])
	r3 = sorted(['5111','5156','5112','5119','5163','5104','5135','5103','5096'])
	lg = sorted(['5089','5086','5085'])
	rg = sorted(['5088','5087','5084'])

	## index which corresponds to the best measuring process
	list_df = []
	for instance in [0, 1]:

		index_l3 = '5237-%s.0' % instance
		index_r3 = '5292-%s.0' % instance
		index_l6 = '5306-%s.0' % instance
		index_r6 = '5251-%s.0' % instance
		index_lg = '5276-%s.0' % instance
		index_rg = '5221-%s.0' % instance

		## read all the data

		temp = pd.read_csv(path_data,
		 		usecols = [elem + '-%s.' % instance + str(int_) for elem in r3 + l3 + r6 + l6 for int_ in range(6)]
		                             + [elem + '-%s.' % instance + str(int_) for elem in  lg + rg for int_ in range(10)]
		                             + [index_l3, index_r3, index_l6, index_r6, index_lg, index_rg]
		                             + ['eid', '21003-%s.0' % instance, '31-0.0'],
				nrows = nrows
		                  ).set_index('eid')
		temp.index = temp.index.rename('id')
		temp = temp[~temp[[index_r3, index_l3, index_r6, index_l6, index_lg, index_rg]].isna().any(axis = 1)]

		def apply_custom(row, instance):
			index_l3 = int(row['5237-%s.0' % instance])
			index_r3 = int(row['5292-%s.0' % instance])
			index_l6 = int(row['5306-%s.0' % instance])
			index_r6 = int(row['5251-%s.0' % instance])
			index_lg = int(row['5276-%s.0' % instance])
			index_rg = int(row['5221-%s.0' % instance])

			instance_prefix = '-%s.' % instance
			arr_l3 = [str(elem) + instance_prefix + str(index_l3) for elem in l3]
			arr_r3 = [str(elem) + instance_prefix + str(index_r3) for elem in r3]
			arr_l6 = [str(elem) + instance_prefix + str(index_l6) for elem in l6]
			arr_r6 = [str(elem) + instance_prefix + str(index_r6) for elem in r6]
			arr_lg = [str(elem) + instance_prefix + str(index_lg) for elem in lg]
			arr_rg = [str(elem) + instance_prefix + str(index_rg) for elem in rg]
			return pd.Series(row[sorted(arr_l3 + arr_r3 + arr_l6 + arr_r6 + arr_lg + arr_rg + ['21003-%s.0' % instance, '31-0.0'])].values)

		temp = temp.apply(lambda row : apply_custom(row, instance = instance), axis = 1)
		temp.columns = sorted([elem + '-%s.0'% instance for elem in l3 + r3 + l6 + r6 + lg + rg] + ['21003-%s.0' % instance, '31-0.0'])

		## Remova NAs
		df = temp[~temp.isna().any(axis = 1)]

		## Rename Columns
		features_index = temp.columns
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
