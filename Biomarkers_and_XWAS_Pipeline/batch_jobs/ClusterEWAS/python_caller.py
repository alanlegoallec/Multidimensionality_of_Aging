import sys
import os
import glob
import pandas as pd
if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')
sys.setrecursionlimit(10000)
from aging.environment_processing.base_processing import path_input_env_inputed, path_clusters, path_HC_features
from aging.model.HC import AglomerativeClusteringFull, Create_full_linkage_matrix, GetInterestingNodes, CreateMappingScoreAndFeatures, CreateBestClusterFromInterestingNodes

target_dataset = str(sys.argv[1])
env_dataset = str(sys.argv[2])
print("Starting HC")
dict_not_changed_index, Zdf, initial_dict = AglomerativeClusteringFull(path_input_env_inputed, target_dataset = target_dataset, env_dataset = env_dataset)

print("Done HC, saving HC")
Zdf.set_index('index_ij').to_csv(path_HC_features + 'HC_%s.csv' % target_dataset)
pd.DataFrame(data = {'Features' : list(initial_dict.values())}).to_csv(path_HC_features + 'Features_%s.csv' % target_dataset, index = False)

print("Done Saving, starting creating full linkage")
tree, linkage_matrix_raw = Create_full_linkage_matrix(Zdf, initial_dict)

print("Done creating linkage, creating interesting nodes")
list_interesting = GetInterestingNodes(tree, linkage_matrix_raw, printing = False)
print(list_interesting)
mapping_index_to_score, mapping_index_to_features = CreateMappingScoreAndFeatures(linkage_matrix_raw)

print("Done creating interesting nodes, Creating dataset")
print(mapping_index_to_score)
print(mapping_index_to_features)
CreateBestClusterFromInterestingNodes(list_interesting, linkage_matrix_raw, path_input_env_inputed, path_clusters, target = target_dataset, env_df = env_dataset)
