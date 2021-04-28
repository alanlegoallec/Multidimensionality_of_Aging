from ..base_processing import read_complex_data

"""
EYE :

2207	Wears glasses or contact lenses
2217	Age started wearing glasses or contact lenses
6147	Reason for glasses/contact lenses => 6
5843	Which eye(s) affected by myopia (short sight)
5832	Which eye(s) affected by hypermetropia (long sight)
5610	Which eye(s) affected by presbyopia
5855	Which eye(s) affected by astigmatism
6205	Which eye(s) affected by strabismus (squint)
5408	Which eye(s) affected by amblyopia (lazy eye)
5877	Which eye(s) affected by other eye condition
5934	Which eye(s) affected by other serious eye condition
2227	Other eye problems
6148	Eye problems/disorders => 5
5890	Which eye(s) affected by diabetes-related eye disease
6119	Which eye(s) affected by glaucoma
5419	Which eye(s) affected by injury or trauma resulting in loss of vision
5441	Which eye(s) are affected by cataract
5912	Which eye(s) affected by macular degeneration
5901	Age when diabetes-related eye disease diagnosed
4689	Age glaucoma diagnosed
5430	Age when loss of vision due to injury or trauma diagnosed
4700	Age cataract diagnosed
5923	Age macular degeneration diagnosed
5945	Age other serious eye condition diagnosed

"""


def read_eye_history_data(instances = [0, 1, 2, 3], **kwargs):

    dict_onehot = {'6148' : {1: 'Diabetes related eye disease',
                             2: 'Glaucoma',
                             3: 'Injury or trauma resulting in loss of vision',
                             4: 'Cataract',
                             5: 'Macular degeneration',
                             6: 'Other serious eye condition',
                             -1: 'Do not know',
                             -7: 'None of the above',
                             -3: 'Prefer not to answer'},
                   '6147' : {1: 'For short-sightedness, i.e. only or mainly for distance viewing such as driving, cinema etc (called myopia)',
                             2: 'For long-sightedness, i.e. for distance and near, but particularly for near tasks like reading (called hypermetropia)',
                             3: 'For just reading/near work as you are getting older (called presbyopia)',
                             4: 'For astigmatism',
                             5: 'For a squint or turn in an eye since childhood (called strabismus)',
                             6: 'For a lazy eye or an eye with poor vision since childhood (called amblyopia)',
                             7: 'Other eye condition',
                             -1: 'Do not know',
                             -3: 'Prefer not to answer'},
                   '5843' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5832' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5610' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5855' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '6205' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5408' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5877' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5934' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5890' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '6119' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5419' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5441' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},
                   '5912' : {1 :'Right eye',2 :'Left eye',3 :'Both eyes'},}

    cols_numb_onehot = {'6147' : 6,'6148' : 5,
                        '5843' : 1,'5832' : 1,'5610' : 1,'5855' : 1,'6205' : 1,'5408' : 1,'5877' : 1,
                        '5934' : 1,'5890' : 1,'6119' : 1,'5419' : 1,'5441' : 1,'5912' : 1}
    cols_ordinal = ['2207', '2217', '2227']
    cols_continuous = ['5901', '4689', '5430', '4700', '5923', '5945']
    cont_fill_na = ['2217', '5901', '4689', '5430', '4700', '5923', '5945']
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
