"""
Created on Mai 2020

@author: Sasha Collin
"""
from TS_ressources import *

# FETCHING HYPER PARAMETERS
hp = hyperparameters(sys.argv)
version = hp.get_version(remove_non_relevant_param=False)

print(version)

# LOADING DATA
print('LOADING DATA')
DF = DataFetcher(hp.target, hp.TS_type)
labeled_data = DF.get_data()  # dictionary

# BUILDING MODEL
print('BUILDING MODEL')
model = TS_model(hp, labeled_data, prediction=True)

# OUTER CROSS-VALIDATION
print('OUTER CROSS-VALIDATION')
model.outer_CV()
