from ..base_processing import read_complex_data

"""
    1289	Cooked vegetable intake
    1299	Salad / raw vegetable intake
    1309	Fresh fruit intake
    1319	Dried fruit intake
    1329	Oily fish intake
    1339	Non-oily fish intake
    1349	Processed meat intake
    1359	Poultry intake
    1369	Beef intake
    1379	Lamb/mutton intake
    1389	Pork intake
    1408	Cheese intake
    1438	Bread intake
    1458	Cereal intake
    1478	Salt added to food
    1488	Tea intake
    1498	Coffee intake => MISSING !!
    1518	Hot drink temperature
    1528	Water intake
    1548	Variation in diet


    6144	Never eat eggs, dairy, wheat, sugar
    1418	Milk type used
    1428	Spread type
    2654	Non-butter spread type details
    1448	Bread type
    1468	Cereal type
    1508	Coffee type
    1538	Major dietary changes in the last 5 years

"""

def read_diet_data(instances = [0, 1, 2, 3], **kwargs):


    dict_onehot = {'1418' : {1: 'Full cream', 2 : 'Semi-skimmed', 3 : 'Skimmed', 4 : 'Soya', 5 : 'Other type of milk', 6 : 'Never/rarely have milk', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1428'  : {1 :'Butter/spreadable butter', 3 : 'Other type of spread/margarine', 0 :'Never/rarely use spread', 2 : 'Flora Pro-Active/Benecol', -1 : 'Do not know',-3 : 'Prefer not to answer'},
                   '1448' : {1 : 'White', 2 : 'Brown', 3 : 'Wholemeal or wholegrain', 4 : 'Other type of bread', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1468' : {1 : 'Bran cereal (e.g. All Bran, Branflakes)', 2 : 'Biscuit cereal (e.g. Weetabix)', 3 : 'Oat cereal (e.g. Ready Brek, porridge)', 4 : 'Muesli', 5 : 'Other (e.g. Cornflakes, Frosties)',
                             -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1508' : {1 : 'Decaffeinated coffee (any type)', 2 : 'Instant coffee', 3 : 'Ground coffee (include espresso, filter etc)', 4 : 'Other type of coffee', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '2654' : {4 : 'Soft (tub) margarine',5 : 'Hard (block) margarine',  6 : 'Olive oil based spread (eg: Bertolli)', 7 : 'Polyunsaturated/sunflower oil based spread (eg: Flora)',
                         2 : 'Flora Pro-Active or Benecol', 8 : 'Other low or reduced fat spread', 9 : 'Other type of spread/margarine', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '6144' : {1 : 'Eggs or foods containing eggs', 2 : 'Dairy products', 3 : 'Wheat products', 4 : 'Sugar or foods/drinks containing sugar', 5 : 'I eat all of the above', -3 : 'Prefer not to answer'},
                   '1508' : {1 : 'Decaffeinated coffee (any type)', 2 : 'Instant coffee', 3 : 'Ground coffee (include espresso, filter etc)', 4 : 'Other type of coffee', -1 : 'Do not know', -3 : 'Prefer not to answer'},
                   '1538' : {0 : 'No', 1 : 'Yes, because of illness', 2 : 'Yes, because of other reasons', -3 : 'Prefer not to answer'}
                   }

    cols_numb_onehot = {'1418' : 1,
                        '1428' : 1,
                        '1448' : 1,
                        '1468' : 1,
                        '2654' : 1,
                        '6144' : 4,
                        '1508' : 1,
                        '1538' : 1}

    cols_continuous = ['1289','1299','1309','1319','1329','1339','1349','1359','1369','1379','1389','1408','1438','1458','1478','1488', #'1498',
                           '1518','1528','1548']
    cols_ordinal = []
    cont_fill_na = []



    df = read_complex_data(instances = instances,
                           dict_onehot = dict_onehot,
                           cols_numb_onehot = cols_numb_onehot,
                           cols_ordinal_ = cols_ordinal,
                           cols_continuous_ = cols_continuous,
                           cont_fill_na_ = cont_fill_na,
                           **kwargs)
    df = df.replace(-10, 0)
    return df
