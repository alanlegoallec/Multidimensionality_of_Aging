import pandas as pd
import sys
from copy import deepcopy
import numpy as np
# To edit for dev
if sys.platform == 'linux':
	path_data = "/n/groups/patel/uk_biobank/project_52887_41230/ukb41230.csv"
	path_data2 = "/n/groups/patel/uk_biobank/project_52887_42640/ukb42640.csv"
	path_dictionary = "/n/groups/patel/samuel/HMS-Aging/Data_Dictionary_Showcase.csv"
	path_features = "/n/groups/patel/samuel/EWAS/feature_importances_final/"
	path_predictions = "/n/groups/patel/samuel/EWAS/predictions_final/"
	path_inputs_env = "/n/groups/patel/samuel/EWAS/inputs_final/"
	path_input_env = "/n/groups/patel/samuel/EWAS/Environmental_raw.csv"
	path_input_env_inputed = "/n/groups/patel/samuel/EWAS/Environmental_inputed.csv"
	path_target_residuals = "/n/groups/patel/Alan/Aging/Medical_Images/data/RESIDUALS_bestmodels_instances_Age_test.csv"
	path_output_linear_study = "/n/groups/patel/samuel/EWAS/linear_output_paper/"
	path_final_preds = "/n/groups/patel/samuel/EWAS/preds_final/"
	path_clusters = "/n/groups/patel/samuel/EWAS/AutomaticClusters/"
	path_HC_features="/n/groups/patel/samuel/EWAS/HC_features/"
elif sys.platform == 'darwin':
	path_data = "/Users/samuel/Desktop/ukbhead.csv"
	path_dictionary = "/Users/samuel/Downloads/drop/Data_Dictionary_Showcase.csv"
	path_features = "/Users/samuel/Desktop/EWAS/feature_importances/"
	path_predictions = "/Users/samuel/Desktop/EWAS/predictions/"
	path_inputs_env = "/n/groups/patel/samuel/EWAS/inputs_/"
	path_target_residuals = "/n/groups/patel/samuel/residuals/"


ETHNICITY_COLS = ['Do_not_know', 'Prefer_not_to_answer', 'NA', 'White', 'British',
       'Irish', 'White_Other', 'Mixed', 'White_and_Black_Caribbean',
       'White_and_Black_African', 'White_and_Asian', 'Mixed_Other', 'Asian',
       'Indian', 'Pakistani', 'Bangladeshi', 'Asian_Other', 'Black',
       'Caribbean', 'African', 'Black_Other', 'Chinese', 'Other_ethnicity',
       'Other']

def read_data(cols_categorical, cols_features, instance, **kwargs):
	"""
	Load features.
	cols_categorical : contains the id of the categorical features.
	cols_features : contains the id of the continuous (non categorical) features.
	instance : select the UK-biobank instance.
	"""
    nrows = None
    if 'nrows' in kwargs.keys():
        nrows = kwargs['nrows']


    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']

    #age_col = '21003-' + str(instance) + '.0'
    #print("PATH DATA : ", path_data)
    temp = pd.read_csv(path_data, usecols = ['eid'] + cols_features + cols_categorical, nrows = nrows)
    for col in cols_categorical:
        temp[col] = temp[col].astype('Int64')
    temp.set_index('eid', inplace = True)

    features_index = temp.columns
    features = []
    for elem in features_index:
        features.append(feature_id_to_name[int(elem.split('-')[0])] + elem.split('-')[1][-2:])
    df = temp.dropna(how = 'any')
    df.columns = features
    return df


def read_complex_data(instances, dict_onehot, cols_numb_onehot, cols_ordinal_, cols_continuous_, cont_fill_na_, cols_half_binary_ = [], **kwargs):
    """
        all cols must be strings or int
        cols_onehot : cols that need one hot encoding
        cols_ordinal : cols that need to be converted as ints
        cols_continuous : cols that don't need to be converted as ints
    """

    df_features = pd.read_csv(path_dictionary, usecols = ["FieldID", "Field"])
    df_features.set_index('FieldID', inplace = True)
    feature_id_to_name = df_features.to_dict()['Field']

    for idx_instance, instance in enumerate(instances) :
        int_cols = set()
        cols_continuous = deepcopy(cols_continuous_)
        cols_ordinal = deepcopy(cols_ordinal_)
        cont_fill_na = deepcopy(cont_fill_na_)


        cols_onehot = [ key + '-%s.%s' % (instance, int_) for key in cols_numb_onehot.keys() for int_ in range(cols_numb_onehot[key])]

        for idx ,elem in enumerate(cols_ordinal):
            if isinstance(elem,(str)):
                cols_ordinal[idx] = elem + '-%s.0' % instance
            elif isinstance(elem, (int)):
                cols_ordinal[idx] = str(elem) + '-%s.0' % instance

        for idx ,elem in enumerate(cols_continuous):
            if isinstance(elem,(str)):
                cols_continuous[idx] = elem + '-%s.0' % instance
            elif isinstance(elem, (int)):
                cols_continuous[idx] = str(elem) + '-%s.0' % instance
        try :
            temp = pd.read_csv(path_data, usecols = ['eid'] + cols_onehot + cols_ordinal + cols_continuous, **kwargs).set_index('eid')
        except ValueError :
            temp = pd.read_csv(path_data2, usecols = ['eid'] + cols_onehot + cols_ordinal + cols_continuous, **kwargs).set_index('eid')
        temp = temp.dropna(how = 'all')
        for column in cols_onehot + cols_ordinal:
            temp[column] = temp[column].astype('Int32')

        if isinstance(cont_fill_na, list):
            for col in cont_fill_na:
                temp[col + '-%s.0' % instance] = temp[col + '-%s.0' % instance].replace(np.nan, 0)
        elif isinstance(cont_fill_na, dict):
            for col, value_ in cont_fill_na.items():
                temp[col + '-%s.0' % instance] = temp[col + '-%s.0' % instance].replace(np.nan, value_)

        if isinstance(cols_half_binary_, list):
            for col in cols_half_binary_:
                temp[col + '-%s.0' % instance] = temp[col + '-%s.0' % instance].replace(-1, 0.5)
        elif isinstance(cols_half_binary_, dict):
            for col, value_ in cols_half_binary_.items():
                temp[col + '-%s.0' % instance] = temp[col + '-%s.0' % instance].replace(-1, value_)


        #display(temp.loc[1098710])
        for col in cols_numb_onehot.keys():

            for idx in range(cols_numb_onehot[col]):
                cate = col + '-%s.%s' % (instance, idx)
                d = pd.get_dummies(temp[cate], dummy_na = True)
                d = d[d[np.nan] == 0]
                if not d.empty:
                    d = d[[elem for elem in d.columns if not np.isnan(elem)]]
                    d.columns = [col + '-%s'%instance + '.' + dict_onehot[col][int(elem)] for elem in d.columns ]
                    temp = temp.drop(columns = [cate])

                    if idx == 0:
                        d_ = d
                    else :
                        common_cols = d.columns.intersection(d_.columns)
                        remaining_cols = d.columns.difference(common_cols)
                        if len(common_cols) > 0 :
                            d_[common_cols] = d_[common_cols].add(d[common_cols])
                        for col_ in remaining_cols:
                            d_[col_] = d[col_]
                    int_cols.update(d_.columns)
            temp = temp.join(d_, how = 'outer')

        # temp[list(int_cols)] = temp[list(int_cols)].astype('Int32')
        features_index = temp.columns
        features = []
        for elem in features_index:
            split = elem.split('-%s' % instance)
            features.append(feature_id_to_name[int(split[0])] + split[1])
        temp.columns = features

        temp['eid'] = temp.index
        temp.index = (temp.index.astype('str') + '_' + str(instance)).rename('id')
        if idx_instance == 0 :
            df = temp
        else :
            df = df.append(temp.reindex(df.columns, axis = 1, fill_value=0))


    df = df.replace(-1, np.nan)
    df = df.replace(-3, np.nan)
    return df
