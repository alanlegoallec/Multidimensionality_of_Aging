from .base_processing import read_data
from ..environment_processing.base_processing import read_complex_data
from ..processing.base_processing import path_data2
import pandas as pd


def read_all_cognitive_data(instances = [2, 3], **kwargs):
    return None

def read_trail_making_data(instances = [2, 3], **kwargs):
    """
    6348	Duration to complete numeric path (trail #1)
    6350	Duration to complete alphanumeric path (trail #2)
    6349	Total errors traversing numeric path (trail #1)
    6351	Total errors traversing alphanumeric path (trail #2)
    """
    df = read_complex_data(instances = [2, 3],
                           dict_onehot = {},
                           cols_numb_onehot = {},
                           cols_ordinal_ = [],
                           cols_continuous_ = [6348, 6349, 6350, 6351],
                           cont_fill_na_ = [],
                           cols_half_binary_ = [],
                           **kwargs)
    return df


def read_reaction_time_data(instances = [0, 1, 2, 3], **kwargs):
    """
    Reaction Time
    Category 100029
    Datafields :
    20023 : Mean time to correctly identify matches
    401	Index for card A in round
    402	Index for card B in round
    404	Duration to first press of snap-button in each round
    """

    list_df = []
    for instance in range(4):
        age_col = '21003-' + str(instance) + '.0'
        cols_age_eid_sex = ['eid', age_col]
        cols = ['401', '402', '404']
        d = pd.read_csv(path_data2, usecols = cols_age_eid_sex + ['20023-%s.0' % instance] +  [elem  + '-%s.%s' % (instance, int_) for int_ in range(12) for elem in cols], **kwargs)
        dict_rename = dict(zip(cols_age_eid_sex + ['20023-%s.0' % instance], ['eid', 'Age when attended assessment centre', 'Mean time to correctly identify matches']))
        d = d.rename(columns = dict_rename)
        for step in range(12):
            d['Duration to first press of snap-button in each round.%s' % step] = d['404-%s.%s' % (instance, step)]
            d['Cards are matching.%s' % step ] = (d['401-%s.%s'% (instance, step)] == d['402-%s.%s'% (instance, step)]).astype(int)
        d['id'] = d['eid'].astype(str) + '_%s' % instance
        d = d.set_index('id')
        d = d[~d['Mean time to correctly identify matches'].isna()]
        d['Mean time to press snap-button'] = d[['Duration to first press of snap-button in each round.%s' % step for step in range(12)]].mean(axis = 1)
        d = d[['eid', 'Age when attended assessment centre', 'Mean time to correctly identify matches', 'Mean time to press snap-button'] +
              ['Cards are matching.%s' % step for step in range(12)]]
        list_df.append(d)
    return pd.concat(list_df)


def read_matrix_pattern_completion_data(**kwargs):
    """
    6373	Number of puzzles correctly solved
    6374	Number of puzzles viewed
    6332	Item selected for each puzzle
    6333	Duration spent answering each puzzle

    ==>
    6373	Number of puzzles correctly solved
    6374	Number of puzzles viewed
    Mean time to solve puzzle

    """
    list_df = []

    for instance in [2, 3]:

        age_col = '21003-' + str(instance) + '.0'
        cols_age_eid_sex = ['eid', age_col]
        other_cols = ['6373-%s.0' % instance, '6374-%s.0' % instance] +  ['6333-%s.%s' % (instance, int_) for int_ in range(15)]
        d = pd.read_csv(path_data2, usecols = cols_age_eid_sex + other_cols, **kwargs)
        d = d[~d[other_cols].isna().all(axis = 1)]
        d['Number of puzzles correctly solved'] = d['6373-%s.0' % instance]
        d['Number of puzzles viewed'] = d['6374-%s.0' % instance]
        d['Mean time to solve puzzles'] = d[['6333-%s.%s' % (instance, int_) for int_ in range(15)]].mean(axis = 1)
        d['id'] = d['eid'].astype(str) + '_%s' % instance
        d = d.set_index('id')
        d = d[['Number of puzzles correctly solved', 'Number of puzzles viewed', 'Mean time to solve puzzles']]
        list_df.append(d)
    return pd.concat(list_df)

def read_pairs_matching_data(**kwargs):
    """
    399	Number of incorrect matches in round
    400	Time to complete round
    396	Number of columns displayed in round
    397	Number of rows displayed in round
    398	Number of correct matches in round
    """
    list_df = []

    for instance in [0, 1, 2, 3]:
        cols_age_eid_sex = ['eid']
        other_cols = ['399-%s.1' % instance, '399-%s.2' % instance] + \
                     ['400-%s.1' % instance, '400-%s.2' % instance] + \
                     ['396-%s.1' % instance, '396-%s.2' % instance] + \
                     ['397-%s.1' % instance, '397-%s.2' % instance] + \
                     ['398-%s.1' % instance, '398-%s.2' % instance]


        d = pd.read_csv(path_data2, usecols = cols_age_eid_sex + other_cols, **kwargs)
        d = d[~d[other_cols].isna().all(axis = 1)]
        list_round = ['First round', 'Second round']
        for idx, elem in enumerate(list_round, 1):
            d['Number of incorrect matches in round.%s' % elem] = d['399-%s.%s' % (instance, idx)]
            d['Time to complete round.%s' % elem] = d['400-%s.%s' % (instance, idx)]
            d['Number of columns displayed in round.%s' % elem] = d['396-%s.%s' % (instance, idx)]
            d['Number of correct matches in round.%s' % elem] = d['398-%s.%s' % (instance, idx)]
            d['Number of rows displayed in round.%s' % elem] = d['397-%s.%s' % (instance, idx)]
        d['id'] = d['eid'].astype(str) + '_%s' % instance
        res = []
        for idx, elem in enumerate(list_round, 1):
            res.append('Number of incorrect matches in round.%s' % elem)
            res.append('Time to complete round.%s' % elem)
            res.append('Number of columns displayed in round.%s' % elem)
            res.append('Number of correct matches in round.%s' % elem)
            res.append('Number of rows displayed in round.%s' % elem)
        d = d.set_index('id')
        d = d[res]
        list_df.append(d)
    return pd.concat(list_df)


def read_tower_rearranging_data(**kwargs):
    """
    Field ID	Description
    21004	Number of puzzles correct
    6383	Number of puzzles attempted
    """
    df = read_complex_data(instances = [2, 3],
                           dict_onehot = {},
                           cols_numb_onehot = {},
                           cols_ordinal_ = [21004, 6383],
                           cols_continuous_ = [],
                           cont_fill_na_ = [],
                           cols_half_binary_ = [],
                           **kwargs)
    return df

def read_symbol_digit_substitution_data(**kwargs):
    """
    Field ID	Description
    23323	Number of symbol digit matches attempted
    23324	Number of symbol digit matches made correctly
    """

    return read_complex_data(instances = [2, 3],
                           dict_onehot = {},
                           cols_numb_onehot = {},
                           cols_ordinal_ = [23323, 23324],
                           cols_continuous_ = [],
                           cont_fill_na_ = [],
                           cols_half_binary_ = [],
                           **kwargs)

def read_paired_associative_learning_data(**kwargs):
    """
    Field ID	Description
    20197	Number of word pairs correctly associated
    6448	Word associated with "huge"
    6459	Word associated with "happy"
    6470	Word associated with "tattered"
    6481	Word associated with "old"
    6492	Word associated with "long"
    6503	Word associated with "red"
    6514	Word associated with "sulking"
    6525	Word associated with "pretty"
    6536	Word associated with "tiny"
    6547	Word associated with "new"
    """

    cols_numb_onehot = {'6448' : 1, '6459' : 1, '6470' : 1,  '6481' : 1, '6492' : 1, '6514' : 1, '6525' : 1, '6536' : 1, '6547' : 1}
    dict_onehot = {'6448' : {1 : 'car', 2 : 'plant', 3 : 'house', 4 : 'trousers', -3 : 'Prefer not to answer'},
                        '6459' : {1 : 'dog', 2 : 'elephant', 3 : 'cat', 4 : 'clown', -3 : 'Prefer not to answer'},
                        '6470' : {1 : 'dress', 2 : 'house', 3 : 'plant', 4 : 'curtains', -3 : 'Prefer not to answer'},
                        '6481' : {1 : 'car', 2 : 'cat', 3 : 'house', 4 : 'elephant', -3 : 'Prefer not to answer'},
                        '6492' : {1 : 'dress', 2 : 'curtains', 3 : 'car', 4 : 'trousers', -3 : 'Prefer not to answer'},
                        '6503' : {1 : 'dish', 2 : 'curtains', 3 : 'plant', 4 : 'car', -3 : 'Prefer not to answer'},
                        '6514' : {1 : 'dog', 2 : 'elephant', 3 : 'clown', 4 : 'cat', -3 : 'Prefer not to answer'},
                        '6525' : {1 : 'cat', 2 : 'dress', 3 : 'dish', 4 : 'plant', -3 : 'Prefer not to answer'},
                        '6536' : {1 : 'dress', 2 : 'house', 3 : 'dog', 4 : 'cat', -3 : 'Prefer not to answer'},
                        '6547' : {1 : 'plant', 2 : 'curtains', 3 : 'house', 4 : 'dress', -3 : 'Prefer not to answer'}
                        }
    cols_ordinal = []
    cols_continuous = ['20197']
    cont_fill_na = []
    cols_half_binary = {}

    df = read_complex_data(instances = [2, 3],
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df


def read_prospective_memory_data(**kwargs):
    """
    20018	Prospective memory result
    4287	Test completion status
    4292	PM: initial answer
    4293	PM: final answer
    4294	Final attempt correct
    4291	Number of attempts
    4288	Time to answer
    """
    cols_numb_onehot = {'4292' : 1, '4293' : 1}
    dict_onehot = {
                        '4292' : {-1 : 'Participant skipped/abandonded', 0 : 'Blue square', 1 : 'Pink star', 2 : 'Grey cross', 3 : 'Orange circle'},
                        '4293' : {-1 : 'Participant skipped/abandonded', 0 : 'Blue square', 1 : 'Pink star', 2 : 'Grey cross', 3 : 'Orange circle'}
                        }

    cols_ordinal = ['20018', '4287', '4294']
    cols_continuous = ['4291', '4288']
    cont_fill_na = []
    cols_half_binary = {}

    df = read_complex_data(instances = [0, 1, 2, 3],
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df

def read_numeric_memory_data(**kwargs):
    """
    Numeric Memory
    Category 100029
    Datafields :
    4282 : Maximum digits remembered correctly
    4285 : Time to complete test

    """
    cols_numb_onehot = {}
    dict_onehot = {}
    cols_ordinal = []
    cols_continuous = ['4282', '4283', '4285']
    cont_fill_na = []
    cols_half_binary = {}

    df = read_complex_data(instances = [0, 2, 3],
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df

def read_fluid_intelligence_data(**kwargs):
    instances = [0, 1, 2, 3]
    """
    20016	Fluid intelligence score
    20128	Number of fluid intelligence questions attempted within time limit
    4924	Attempted fluid intelligence (FI) test.
    4935	FI1 : numeric addition test
    4946	FI2 : identify largest number
    4957	FI3 : word interpolation
    4968	FI4 : positional arithmetic
    4979	FI5 : family relationship calculation
    #4990	FI6 : conditional arithmetic
    #5001	FI7 : synonym
    #5012	FI8 : chained arithmetic
    #5556	FI9 : concept interpolation
    #5699	FI10 : arithmetic sequence recognition
    #5779	FI11 : antonym
    #5790	FI12 : square sequence recognition
    #5866	FI13 : subset inclusion logic
    """
    cols_numb_onehot = {'4935' : 1, '4946' : 1, '4957' : 1,  '4968' : 1, '4979' : 1, '4990' : 1, '5001' : 1, '5012' : 1, '5556' : 1, '5699': 1, '5779' : 1, '5790' : 1, '5866' : 1}
    dict_onehot = {'4935' : {13 : '13', 14  : '14', 15 : '15', 16 : '16', 17 : '17',-1 : 'Do not know', -3 : 'Prefer not to answer'},
                        '4946': {642 : '642', 308 : '308', 987 : '987', 714 : '714', 253 : '253', -1 : 'Do not know',-3 : 'Prefer not to answer'},
                        '4957' : {1 : 'Grow', 2 : 'Develop', 3 :  'Improve', 4 : 'Adult', 5 : 'Old', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                        '4968' : {5 : '5', 6 : '6', 7 : '7', 8 : '8', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                        '4979' : {1 : 'Aunt', 2 : 'Sister', 3 : 'Niece', 4 : 'Cousin', 5 : 'No relation', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                        #'4990' : {68 : '68', 69 : '69', 70 : '70', 71 : '71', 72 : '72', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                        #'5001' : {1 : 'Pause', 2 :'Close', 3 : 'Cease', 4 : 'Break', 5 : 'Rest', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                        #'5012' : {25 : '25', 26 : '26', 27 : '27', 28 : '28', 29 : '29', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                        #'5556' : {1 : 'Long', 2 : 'Deep', 3 : 'Top', 4 : 'Metres', 5 : 'Tall', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                        #'5699' : {96 : '96', 95 : '95', 94 : '94', 93 : '93', 92 : '92', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                        #'5779' : {1 : 'Calm', 2 : 'Anxious', 3 : 'Cool', 4 : 'Worried', 5 : 'Tense', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                        #'5790' : {50 : '50', 49 : '49', 48 : '48', 47 : '47', 46 : '46', 45 : '45', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                        #'5866' : {1 : 'False', 2 : 'True', 3 : 'Neither true nor false', -5 : 'Not sure', -1 : 'Do not know', -3 : 'Prefer not to answer'}
                        }
    cols_ordinal = ['4924']
    cols_continuous = ['20016', '20128']
    cont_fill_na = []
    cols_half_binary = {}


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)
    return df
