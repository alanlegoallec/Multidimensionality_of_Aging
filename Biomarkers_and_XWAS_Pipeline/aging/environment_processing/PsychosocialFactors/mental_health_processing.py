from ..base_processing import read_complex_data



"""
1920	Mood swings
1930	Miserableness
1940	Irritability
1950	Sensitivity / hurt feelings
1960	Fed-up feelings
1970	Nervous feelings
1980	Worrier / anxious feelings
1990	Tense / 'highly strung'
2000	Worry too long after embarrassment
2010	Suffer from 'nerves'
2020	Loneliness, isolation
2030	Guilty feelings
2040	Risk taking
4526	Happiness
4537	Work/job satisfaction +> ? big loss due to retired people
4548	Health satisfaction
4559	Family relationship satisfaction
4570	Friendships satisfaction
4581	Financial situation satisfaction
2050	Frequency of depressed mood in last 2 weeks
2060	Frequency of unenthusiasm / disinterest in last 2 weeks
2070	Frequency of tenseness / restlessness in last 2 weeks
2080	Frequency of tiredness / lethargy in last 2 weeks
2090	Seen doctor (GP) for nerves, anxiety, tension or depression
2100	Seen a psychiatrist for nerves, anxiety, tension or depression
4598	Ever depressed for a whole week
4609	Longest period of depression => Put 0 for nans
4620	Number of depression episodes => put 0 for nans
4631	Ever unenthusiastic/disinterested for a whole week
5375	Longest period of unenthusiasm / disinterest => put 0 for nans
5386	Number of unenthusiastic/disinterested episodes => put 0 for nans
4642	Ever manic/hyper for 2 days
4653	Ever highly irritable/argumentative for 2 days
6156	Manic/hyper symptoms
5663	Length of longest manic/irritable episode => put 0 for nans
5674	Severity of manic/irritable episodes
6145	Illness, injury, bereavement, stress in last 2 years

Missing : '4526', '4548', '4559', '4570', '4581' all instances
"""

def read_mental_health_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot = {'6156' : {11 : 'I was more active than usual', 12 : 'I was more talkative than usual', 13 :'I needed less sleep than usual', 14 : 'I was more creative or had more ideas than usual',
                             15 : 'All of the above', -7 : 'None of the above', 0 : 'No symptoms', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '6145' : {1 : 'Serious illness, injury or assault to yourself', 2 : 'Serious illness, injury or assault of a close relative', 3 : 'Death of a close relative', 4 : 'Death of a spouse or partner',
                             5 : 'Marital separation/divorce', 6 : 'Financial difficulties', -7 : 'None of the above', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '5674' : {11 : 'No problems', 12 : 'Needed treatment or caused problems with work, relationships, finances, the law or other aspects of life', -1 : 'Do not know', -3 : 'Prefer not to answer'}}

    cols_numb_onehot = {'6145' : 6, '6156' : 4, '5674' : 1}
    cols_ordinal = ['1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020', '2030', '2040',
                    #'4526', '4548', '4559', '4570', '4581', '4537'
                    '2050', '2060', '2070', '2080', '2090', '2100',
                    '4598', '4609', '4620', '4631', '5375', '5386', '4642', '4653', '5663']
    cols_continuous = []
    cont_fill_na = ['4609', '4620', '5375', '5386', '5663']
    cols_half_binary = ['1920', '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000', '2010', '2020', '2030', '2040',
                        '2090', '2100', '4598', '4631', '4642', '4653', '5663']


    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           cols_half_binary_ = cols_half_binary,
                           **kwargs)

    return df
