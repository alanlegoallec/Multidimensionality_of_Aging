from ..base_processing import read_complex_data
import pandas as pd
import numpy as np
"""
6177	Medication for cholesterol, blood pressure or diabetes
6153	Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones
6154	Medication for pain relief, constipation, heartburn
6155	Vitamin and mineral supplements
6179	Mineral and other dietary supplements
"""

def read_medication_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot = {'6177' : {1 : 'Cholesterol lowering medication', 2 : 'Blood pressure medication', 3 : 'Insulin',
                             -7 : 'None of the above', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '6153' : {1 : 'Cholesterol lowering medication', 2 : 'Blood pressure medication', 3 : 'Insulin',
                             4 : 'Hormone replacement therapy', 5 : 'Oral contraceptive pill or minipill', -7 : 'None of the above',
                             -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '6154' : {1: 'Aspirin', 2 : 'Ibuprofen (e.g. Nurofen)', 3 : 'Paracetamol', 4 : 'Ranitidine (e.g. Zantac)',
                             5 : 'Omeprazole (e.g. Zanprol)', 6 : 'Laxatives (e.g. Dulcolax, Senokot)', -7 : 'None of the above',
                             -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '6155' : {1 : 'Vitamin A', 2 : 'Vitamin B', 3 : 'Vitamin C', 4 : 'Vitamin D', 5 : 'Vitamin E', 6 : 'Folic acid or Folate (Vit B9)',
                             7 : 'Multivitamins +/- minerals', -7 : 'None of the above', -3 : 'Prefer not to answer'},
                   '6179' : {1 : 'Fish oil (including cod liver oil)', 2 : 'Glucosamine', 3 : 'Calcium', 4 :'Zinc', 5 : 'Iron',
                             6 : 'Selenium', -7 : 'None of the above', -3 : 'Prefer not to answer'}
                   }
    cols_numb_onehot = {'6177' : 3, '6153' : 4, '6154' : 6, '6155' : 7, '6179' : 6}
    cols_ordinal = []
    cols_continuous = []
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

    for type_ in ['Cholesterol lowering medication','Blood pressure medication', 'Insulin']:
        male_and_female = df[['Medication for cholesterol, blood pressure or diabetes.' + type_, 'Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones.' + type_]]
        def custom_sum(row, type_):
            male_val = row['Medication for cholesterol, blood pressure or diabetes.' + type_]
            female_val = row['Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones.' + type_]

            if pd.isna(male_val) and pd.isna(female_val):
                return np.nan
            elif ~pd.isna(male_val) and pd.isna(female_val):
                return male_val
            elif pd.isna(male_val) and ~pd.isna(female_val):
                return female_val
            else :
                return (male_val + female_val)%2

        df['Medication for cholesterol, blood pressure or diabetes.' + type_] = male_and_female.apply(lambda row : custom_sum(row, type_ = type_), axis = 1)
        #df['Medication for cholesterol, blood pressure or diabetes.' + type_] = (df['Medication for cholesterol, blood pressure or diabetes.' + type_] + df['Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones.' + type_])%2
        df = df.drop(columns = ['Medication for cholesterol, blood pressure, diabetes, or take exogenous hormones.' + type_])
    return df
