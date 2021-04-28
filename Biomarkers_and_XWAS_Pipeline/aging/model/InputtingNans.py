import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
cols_ethnicity_full = ['Ethnicity.White', 'Ethnicity.British', 'Ethnicity.Irish',
       'Ethnicity.White_Other', 'Ethnicity.Mixed',
       'Ethnicity.White_and_Black_Caribbean',
       'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian',
       'Ethnicity.Mixed_Other', 'Ethnicity.Asian', 'Ethnicity.Indian',
       'Ethnicity.Pakistani', 'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other',
       'Ethnicity.Black', 'Ethnicity.Caribbean', 'Ethnicity.African',
       'Ethnicity.Black_Other', 'Ethnicity.Chinese',
       'Ethnicity.Other_ethnicity', 'Ethnicity.Do_not_know',
       'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA', 'Ethnicity.Other', 'Unnamed: 0']

def load_raw_data(path_raw,
                  path_output,
                  path_ethnicities = '/n/groups/patel/Alan/Aging/Medical_Images/data/data-features_instances.csv'):
    final_df = pd.read_csv(path_raw).set_index('id')
    df_ethnicity = pd.read_csv(path_ethnicities).set_index('eid')
    cols_ethnicity = ['Ethnicity.White', 'Ethnicity.Mixed', 'Ethnicity.Black', 'Ethnicity.Asian', 'Ethnicity.Other', 'Ethnicity.Chinese']
    df_ethnicity = pd.DataFrame(df_ethnicity[cols_ethnicity].idxmax(axis = 1))
    df_ethnicity.columns = ['Ethnicity']
    final_df = final_df.reset_index().merge(df_ethnicity, on ='eid').set_index('id')

    features = [ elem for elem in final_df.columns if elem not in cols_ethnicity_full + ['Sex', 'Age when attended assessment centre', 'Ethnicity', 'eid']]
    return features, final_df


def compute_linear_coefficients_for_each_col(final_df, col):
    age_sex_ethnicity_features = ['Sex', 'Age when attended assessment centre', 'Ethnicity']
    coefs_col = pd.DataFrame(columns= [col, 'Sex', 'Ethnicity'])
    column = final_df[[col, 'eid']  + age_sex_ethnicity_features]
    #distinct_eid_col = column.eid.drop_duplicates().values

    group_eid = column.groupby('eid').count()[col]
    distinct_eid_col = group_eid[group_eid > 1].index
    is_longitudinal = (column.groupby('eid').count()[col] > 1).any()

    if not is_longitudinal:
        return None, column
    else :
        ## Create weights by sex and ethnicty
        for eid in distinct_eid_col:
            points = column[column.eid == eid].dropna()
            num_points = (~points[col].isna()).sum()
            if num_points == 1 or num_points == 0:
                continue
            else :
                if num_points == 2:
                    point1 = points.iloc[0]
                    point2 = points.iloc[1]
                    if np.abs(point2['Age when attended assessment centre'] - point1['Age when attended assessment centre']) < 1e-8 :
                        coef = 0
                    else :
                        coef = (point2[col] - point1[col])/(point2['Age when attended assessment centre'] - point1['Age when attended assessment centre'])

                elif num_points > 2:
                    y = points[col].values.reshape(-1, 1)
                    x = points['Age when attended assessment centre'].values.reshape(-1, 1)
                    lin  = LinearRegression()
                    lin.fit(x, y)
                    coef = lin.coef_[0][0]
                else :
                    raise ValueError('not the right number of points')
                coefs_col = coefs_col.append(pd.Series([float(coef), points['Sex'].mean(), points['Ethnicity'].min()], index=[col, 'Sex', 'Ethnicity'], name= eid))
        coefs_mean = coefs_col.groupby(['Sex', 'Ethnicity']).mean()
        if coefs_mean.shape[0] != 12:
            coefs_mean = coefs_col.groupby(['Sex']).mean()
            if coefs_mean.shape[0] != 2:
                coefs_mean = None
        return coefs_mean, column


def input_variables_in_column(col, column, coefs_mean):
    print("Inputing col : %s"  % col)
    categorical = (column[col].max() == 1) and (column[col].min() == 0)
    all_ethnicity_available = (coefs_mean.shape[0] == 12)
    count = column.groupby('eid').count()
    distinct_eid_col = count.index[(count[col] > 0) & (count['Age when attended assessment centre'] > 1)]
    def recenter_between_0_1(value_):
        if value_ < 0:
            return 0
        elif value_ > 1:
            return 1
        else :
            return value_

    for eid in distinct_eid_col:
        points = column[column.eid == eid]

        ## inputting or not :
        if points[col].isna().any():

            ## count number of availaible points:
            num_avail = points.shape[0]
            num_avail_filled = len(points[col].dropna())
            if num_avail == 1 or num_avail_filled == 0:
                continue
            elif num_avail == 2:
                missing_point = points[points[col].isna()].iloc[0]
                valid_point = points[~points[col].isna()].iloc[0]
                sex = valid_point['Sex']
                age_missing = missing_point['Age when attended assessment centre']
                age_valid = valid_point['Age when attended assessment centre']
                valid_value = valid_point[col]
                if all_ethnicity_available:
                    ethnicity = valid_point['Ethnicity']
                    coef_ = coefs_mean.loc[sex, ethnicity].values
                else :
                    coef_ = coefs_mean.loc[sex].values
                missing_value = valid_value + (age_missing - age_valid) * coef_
                if categorical:
                    missing_value = recenter_between_0_1(missing_value)
                column.loc[missing_point.name, col] = missing_value

            elif num_avail == 3:
                if num_avail_filled == 2:
                    missing_point = points[points[col].isna()].iloc[0]
                    valid_point_1 = points[~points[col].isna()].iloc[0]
                    valid_point_2 = points[~points[col].isna()].iloc[1]
                    sex = missing_point['Sex']
                    if all_ethnicity_available:
                        ethnicity = missing_point['Ethnicity']
                        coef_ = coefs_mean.loc[sex, ethnicity].values
                    else :
                        coef_ = coefs_mean.loc[sex].values
                    age_missing = missing_point['Age when attended assessment centre']
                    age_valid_1 = valid_point_1['Age when attended assessment centre']
                    age_valid_2 = valid_point_2['Age when attended assessment centre']

                    estimated_1 = valid_point_1[col] + (age_missing - age_valid_1) * coef_
                    estimated_2 = valid_point_2[col] + (age_missing - age_valid_2) * coef_

                    dist_1 = abs(age_valid_1 - age_missing)
                    dist_2 = abs(age_valid_2 - age_missing)

                    missing_value = (estimated_1/dist_1 + estimated_2/dist_2) / (1/dist_1 + 1/dist_2)
                    if categorical:
                        missing_value = recenter_between_0_1(missing_value)
                    column.loc[missing_point.name, col] = missing_value
                else : # 2 missing points :
                    missing_point_1 = points[points[col].isna()].iloc[0]
                    missing_point_2 = points[points[col].isna()].iloc[1]
                    valid_point = points[~points[col].isna()].iloc[0]
                    sex = valid_point['Sex']
                    if all_ethnicity_available:
                        ethnicity = valid_point['Ethnicity']
                        coef_ = coefs_mean.loc[sex, ethnicity].values
                    else :
                        coef_ = coefs_mean.loc[sex].values
                    age_missing_1 = missing_point_1['Age when attended assessment centre']
                    age_missing_2 = missing_point_2['Age when attended assessment centre']
                    age_valid = valid_point['Age when attended assessment centre']
                    valid_value = valid_point[col]
                    missing_value_1 = valid_value + (age_missing_1 - age_valid) * coef_
                    missing_value_2 = valid_value + (age_missing_2 - age_valid) * coef_
                    if categorical:
                        missing_value_1 = recenter_between_0_1(missing_value_1)
                        missing_value_2 = recenter_between_0_1(missing_value_2)
                    column.loc[missing_point_1.name, col] = missing_value_1
                    column.loc[missing_point_2.name, col] = missing_value_2
            elif num_avail == 4:
                if num_avail_filled == 1:
                    missing_point_1 = points[points[col].isna()].iloc[0]
                    missing_point_2 = points[points[col].isna()].iloc[1]
                    missing_point_3 = points[points[col].isna()].iloc[2]
                    valid_point = points[~points[col].isna()].iloc[0]
                    sex = valid_point['Sex']
                    if all_ethnicity_available:
                        ethnicity = valid_point['Ethnicity']
                        coef_ = coefs_mean.loc[sex, ethnicity].values
                    else :
                        coef_ = coefs_mean.loc[sex].values
                    age_missing_1 = missing_point_1['Age when attended assessment centre']
                    age_missing_2 = missing_point_2['Age when attended assessment centre']
                    age_missing_3 = missing_point_3['Age when attended assessment centre']
                    age_valid = valid_point['Age when attended assessment centre']
                    valid_value = valid_point[col]

                    missing_value_1 = valid_value + (age_missing_1 - age_valid) * coef_
                    missing_value_2 = valid_value + (age_missing_2 - age_valid) * coef_
                    missing_value_3 = valid_value + (age_missing_3 - age_valid) * coef_
                    if categorical :
                        missing_value_1 = recenter_between_0_1(missing_value_1)
                        missing_value_2 = recenter_between_0_1(missing_value_2)
                        missing_value_3 = recenter_between_0_1(missing_value_3)


                    column.loc[missing_point_1.name, col] = missing_value_1
                    column.loc[missing_point_2.name, col] = missing_value_2
                    column.loc[missing_point_3.name, col] = missing_value_3

                elif num_avail_filled == 2:
                    missing_point_1 = points[points[col].isna()].iloc[0]
                    missing_point_2 = points[points[col].isna()].iloc[1]
                    valid_point_1 = points[~points[col].isna()].iloc[0]
                    valid_point_2 = points[~points[col].isna()].iloc[1]

                    sex = missing_point_1['Sex']
                    if all_ethnicity_available:
                        ethnicity = missing_point_1['Ethnicity']
                        coef_ = coefs_mean.loc[sex, ethnicity].values
                    else :
                        coef_ = coefs_mean.loc[sex].values

                    age_missing_1 = missing_point_1['Age when attended assessment centre']
                    age_missing_2 = missing_point_2['Age when attended assessment centre']

                    age_valid_1 = valid_point_1['Age when attended assessment centre']
                    age_valid_2 = valid_point_2['Age when attended assessment centre']

                    estimated_1_missing_1 = valid_point_1[col] + (age_missing_1 - age_valid_1) * coef_
                    estimated_2_missing_1 = valid_point_2[col] + (age_missing_1 - age_valid_2) * coef_
                    estimated_1_missing_2 = valid_point_1[col] + (age_missing_2 - age_valid_1) * coef_
                    estimated_2_missing_2 = valid_point_2[col] + (age_missing_2 - age_valid_2) * coef_

                    dist_1_missing_1 = abs(age_valid_1 - age_missing_1)
                    dist_2_missing_1 = abs(age_valid_2 - age_missing_1)

                    dist_1_missing_2 = abs(age_valid_1 - age_missing_2)
                    dist_2_missing_2 = abs(age_valid_2 - age_missing_2)

                    missing_value_1 = (estimated_1_missing_1/dist_1_missing_1 + estimated_2_missing_1/dist_2_missing_1) / (1/dist_1_missing_1 + 1/dist_2_missing_1)
                    missing_value_2 = (estimated_1_missing_2/dist_1_missing_2 + estimated_2_missing_2/dist_2_missing_2) / (1/dist_1_missing_2 + 1/dist_2_missing_2)

                    if categorical:
                        missing_value_1 = recenter_between_0_1(missing_value_1)
                        missing_value_2 = recenter_between_0_1(missing_value_2)
                    column.loc[missing_point_1.name, col] = missing_value_1
                    column.loc[missing_point_2.name, col] = missing_value_2


                elif num_avail_filled == 3:
                    missing_point = points[points[col].isna()].iloc[0]
                    valid_point_1 = points[~points[col].isna()].iloc[0]
                    valid_point_2 = points[~points[col].isna()].iloc[1]
                    valid_point_3 = points[~points[col].isna()].iloc[2]

                    sex = missing_point['Sex']
                    if all_ethnicity_available:
                        ethnicity = missing_point['Ethnicity']
                        coef_ = coefs_mean.loc[sex, ethnicity].values
                    else :
                        coef_ = coefs_mean.loc[sex].values
                    age_missing = missing_point['Age when attended assessment centre']

                    age_valid_1 = valid_point_1['Age when attended assessment centre']
                    age_valid_2 = valid_point_2['Age when attended assessment centre']
                    age_valid_3 = valid_point_3['Age when attended assessment centre']

                    estimated_1 = valid_point_1[col] + (age_missing - age_valid_1) * coef_
                    estimated_2 = valid_point_2[col] + (age_missing - age_valid_2) * coef_
                    estimated_3 = valid_point_3[col] + (age_missing - age_valid_3) * coef_

                    dist_1 = abs(age_valid_1 - age_missing)
                    dist_2 = abs(age_valid_2 - age_missing)
                    dist_3 = abs(age_valid_3 - age_missing)
                    missing_value = (estimated_1/dist_1 + estimated_2/dist_2 + estimated_3/dist_3) / (1/dist_1 + 1/dist_2 + 1/dist_3)
                    if categorical:
                        missing_value = recenter_between_0_1(missing_value)
                    column.loc[missing_point.name, col] = missing_value
    print("Done inputing col : %s " % col)
    return column.drop(columns = ['Ethnicity'])

def compute_coefs_and_input(final_df, col):
    #print("Compute mean of coef : %s" % col)
    coefs_mean, column = compute_linear_coefficients_for_each_col(final_df, col)
    if coefs_mean is not None:
        #print("Done Averaging, input missing data in %s" % col )
        column_modified = input_variables_in_column(col, column, coefs_mean)
        #print("Done inputting %s" % col)
        return column_modified
    else :
        #print('Non longitudinal feature or not enought samples in %s' % col)
        return column.drop(columns = ['Ethnicity'])
