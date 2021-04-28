import sys
import os
import glob

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.processing.base_processing import path_input, path_clusters, path_HC_features
from aging.model.HC import *

print("Starting HC")
dict_not_changed_index, Zdf, initial_dict = AglomerativeClusteringFull(path_input, target_dataset = None)

print("Done HC, saving HC")
Zdf.set_index('index_ij').to_csv(path_HC_features + 'HC_biomarkers.csv')
pd.DataFrame(data = {'Features' : list(initial_dict.values())}).to_csv(path_HC_features + 'Features_biomarkers.csv', index = False)

print("Done Saving, starting creating full linkage")
tree, linkage_matrix_raw = Create_full_linkage_matrix(Zdf, initial_dict)

print("Done creating linkage, creating interesting nodes")
list_interesting = GetInterestingNodes(tree, linkage_matrix_raw, printing = False)
mapping_index_to_score, mapping_index_to_features = CreateMappingScoreAndFeatures(linkage_matrix_raw)

print("Done creating interesting nodes, Creating datasets")
CreateClustersFromInterestingNodes(list_interesting, linkage_matrix_raw, path_input, path_clusters)
