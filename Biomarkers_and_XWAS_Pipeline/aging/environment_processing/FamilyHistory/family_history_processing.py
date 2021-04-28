from ..base_processing import read_complex_data

"""

20112	Illnesses of adopted father
20113	Illnesses of adopted mother
20114	Illnesses of adopted siblings
20107	Illnesses of father
20110	Illnesses of mother
20111	Illnesses of siblings
#1797	Father still alive
#3912	Adopted father still alive
#2946	Father's age
#1807	Father's age at death
#1835	Mother still alive
#3942	Adopted mother still alive
#1845	Mother's age
#3526	Mother's age at death
1873	Number of full brothers
3972	Number of adopted brothers
1883	Number of full sisters
3982	Number of adopted sisters
5057	Number of older siblings
4501	Non-accidental death in close genetic family


"""

def read_family_history_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot = {'20107' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease',-11: 'Do not know1',-13 : 'Prefer not to answer1', -17 : 'None of the above1',
                              -21 : 'Do not know2', -23 : 'Prefer not to answer2', -27 : 'None of the above2'},
                   '20110' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease',-11: 'Do not know1',-13 : 'Prefer not to answer1', -17 : 'None of the above1',
                              -21 : 'Do not know2', -23 : 'Prefer not to answer2', -27 : 'None of the above2'},
                   '20111' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease',-11: 'Do not know1',-13 : 'Prefer not to answer1', -17 : 'None of the above1',
                              -21 : 'Do not know2', -23 : 'Prefer not to answer2', -27 : 'None of the above2'},
                   '20112' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease',-11: 'Do not know1',-13 : 'Prefer not to answer1', -17 : 'None of the above1',
                              -21 : 'Do not know2', -23 : 'Prefer not to answer2', -27 : 'None of the above2'},
                   '20113' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease',-11: 'Do not know1',-13 : 'Prefer not to answer1', -17 : 'None of the above1',
                              -21 : 'Do not know2', -23 : 'Prefer not to answer2', -27 : 'None of the above2'},
                   '20114' : {13 :'Prostate cancer', 12 :'Severe depression',  11 : "Parkinson's disease", 10 :"Alzheimer's disease/dementia",
                               9 :"Diabetes", 8:'High blood pressure',6:'Chronic bronchitis/emphysema',5:'Breast cancer',4:'Bowel cancer',
                               3 :'Lung cancer',2:'Stroke',1:'Heart disease',-11: 'Do not know1',-13 : 'Prefer not to answer1', -17 : 'None of the above1',
                              -21 : 'Do not know2', -23 : 'Prefer not to answer2', -27 : 'None of the above2'}}

    cols_numb_onehot = {'20107' : 10,
                        '20110' : 11,
                        '20111' : 12,
                        '20114' : 7,
                        '20112' : 7,
                        '20113' : 6}
    cols_ordinal = []
    cols_continuous = []
    cont_fill_na = []


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           **kwargs)

    cols_to_drop  = []
    for substring in ['None of the above', 'Prefer not to answer', 'Do not know']:
        for illness_type in ['Illnesses of father.', 'Illnesses of adopted father.', 'Illnesses of adopted mother.', 'Illnesses of mother.']:
            try :
                df[illness_type + substring] = (df[illness_type + substring + '1'] + df[illness_type + substring + '2']) % 2
                cols_to_drop.append(illness_type + substring + '1')
                cols_to_drop.append(illness_type + substring + '2')
            except KeyError:
                continue

    df = df.drop(columns = cols_to_drop)
    return df
