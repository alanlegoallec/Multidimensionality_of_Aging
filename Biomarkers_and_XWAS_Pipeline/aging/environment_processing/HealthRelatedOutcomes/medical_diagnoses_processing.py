
from ..base_processing import path_data, path_inputs_env
import glob
import pandas as pd
import copy
import sys



def read_medical_diagnoses_data(letter, **kwargs):
    """

    Create medical_diagnoses corresponding to the class 'letter', index is id !!

    """
    if not letter.isupper():
        raise ValueError(' %s letter not available chose among [A-Z] !' % letter)
    else :
        filename_all = 'medical_diagnoses.csv'
        list_files_all = glob.glob(path_inputs_env + filename_all)

        ## Check if big file exists or else reconstructs it
        if len(list_files_all) == 1:
            ### Chose columns matching letter
            df_all_cols = pd.read_csv(list_files_all[0], nrows = 1).set_index('eid').columns
            cols_letter = df_all_cols[df_all_cols.str.contains('^%s' % letter)]
            ### Read corresponding columns + eid
            df_letter = pd.read_csv(list_files_all[0], usecols = ['eid'] + list(cols_letter),  **kwargs)
            df_letter = df_letter.set_index('eid')

            print("SIZE RAW LETTER :", sys.getsizeof(df_letter))

        elif len(list_files_all) == 0:
            ### Read all dataframe + saving it
            df_all = read_medical_diagnoses_all_data(**kwargs)
            df_all.to_csv(path_inputs_env + 'medical_diagnoses.csv')

            ### Select columns corresponding to letter
            cols_letter = df_all.columns[df_all.columns.str.contains('^%s' % letter)]
            df_letter = df_all[cols_letter]
            del df_all

        else:
            raise ValueError(' Too many medical_diagnoses files ! ')

        ## Add eid column and create new index : id
        df_letter['eid'] = df_letter.index
        df_letter = df_letter.astype('int8')
        print("SIZE INT8 LETTER :", sys.getsizeof(df_letter))
        list_df = []
        for instance in range(4):
            df_letter_instance = copy.deepcopy(df_letter)
            df_letter_instance.index = (df_letter_instance.index.astype('str') + '_%s' % instance).rename('id')
            list_df.append(df_letter_instance)
        df_letter_final = pd.concat(list_df)
        print("SIZE CONCAT :", sys.getsizeof(df_letter_final))
        df_letter_final.to_csv(path_inputs_env + 'medical_diagnoses_%s.csv' % letter, chunksize = 20000)
        return df_letter_final




def read_medical_diagnoses_all_data(**kwargs):
    """

    read all medical diagnoses
    return df with index eid !

    """
    coding19 = pd.read_csv('/n/groups/patel/samuel/EWAS/coding19.tsv', sep='\t')
    dict_coding_to_meaning = dict(zip(coding19.coding, coding19.meaning))

    temp = pd.read_csv(path_data, usecols = ['eid'] + ['41270-0.%s' % int_ for int_ in range(213)], **kwargs).set_index('eid')
    for idx, col in enumerate(temp.columns):

        d = pd.get_dummies(temp[col])
        d = d.astype('uint8')
        if idx == 0:
            d_ = d
        else :
            common_cols = d.columns.intersection(d_.columns)
            remaining_cols = d.columns.difference(common_cols)
            if len(common_cols) > 0 :
                d_[common_cols] = d_[common_cols].add(d[common_cols])
            for col_ in remaining_cols:
                d_[col_] = d[col_]

    new_cols = [dict_coding_to_meaning[elem] for elem in d_.columns]
    d_.columns = new_cols
    columns_sorted = sorted(d_.columns)
    d_ = d_[columns_sorted]
    return d_
