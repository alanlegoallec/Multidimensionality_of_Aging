import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster import hierarchy
import copy
import sys
sys.path.append('/home/sd375')

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster import hierarchy
from .load_and_save_environment_data import load_target_residuals
from ..environment_processing.base_processing import path_inputs_env
from UsefulFonctions import ComputeDistanceMatrix
import os

dict_ensemble_to_datasets = {
                                'ENSEMBLE_HealthAndMedicalHistory' : ['Breathing', 'CancerScreening', 'ChestPain', 'Claudification', 'Eyesight', 'GeneralHealth', 'GeneralPain', 'Hearing', 'Medication', 'Mouth'],
                                'ENSEMBLE_LifestyleAndEnvironment'  : ['Alcohol', 'Diet', 'ElectronicDevices', 'PhysicalActivityQuestionnaire', 'SexualFactors', 'Sleep', 'Smoking', 'SunExposure'],
                                'ENSEMBLE_PsychosocialFactors' : ['MentalHealth', 'SocialSupport'],
                                'ENSEMBLE_SocioDemographics' : ['Education', 'Employment', 'Household', 'OtherSociodemographics']
                            }

cols_ethnicity = ['Ethnicity.White', 'Ethnicity.British', 'Ethnicity.Irish',
       'Ethnicity.White_Other', 'Ethnicity.Mixed',
       'Ethnicity.White_and_Black_Caribbean',
       'Ethnicity.White_and_Black_African', 'Ethnicity.White_and_Asian',
       'Ethnicity.Mixed_Other', 'Ethnicity.Asian', 'Ethnicity.Indian',
       'Ethnicity.Pakistani', 'Ethnicity.Bangladeshi', 'Ethnicity.Asian_Other',
       'Ethnicity.Black', 'Ethnicity.Caribbean', 'Ethnicity.African',
       'Ethnicity.Black_Other', 'Ethnicity.Chinese',
       'Ethnicity.Other_ethnicity', 'Ethnicity.Do_not_know',
       'Ethnicity.Prefer_not_to_answer', 'Ethnicity.NA', 'Ethnicity.Other']

cols_age_sex_eid_ethnicity = ['Sex', 'eid', 'Age when attended assessment centre'] + cols_ethnicity

## Agglomerative Clustering :

# Metrics :
def NegativeIntersection(x, y):
    """
    x, y 1D numpy vectors
    """
    return -x.dot(y) #1 - x.dot(y) / np.sum(x | y)


def CreateDictSizes(path_dataset_full, target_dataset, env_dataset):
    ## Load everything
    dict_name_to_df = {}
    dict_name_to_num_features = {}
    print("Loading Full raw data")
    if env_dataset is not None :
        if 'ENSEMBLE' in env_dataset :
            subdatasets = dict_ensemble_to_datasets[env_dataset]
            usecols = []
            for sub_envdataset in subdatasets :
                usecol = pd.read_csv(path_inputs_env + '%s.csv' % sub_envdataset, nrows = 1).set_index('id').columns
                usecol = [elem for elem in usecol if elem not in ['eid', 'Sex', 'Age when attended assessment centre', 'Unnamed: 0'] + cols_ethnicity]
                usecols += usecol
        else :
            usecols = pd.read_csv(path_inputs_env + '%s.csv' % env_dataset, nrows = 1).set_index('id').columns
            usecols = [elem for elem in usecols if elem not in ['eid', 'Sex', 'Age when attended assessment centre', 'Unnamed: 0'] + cols_ethnicity]
    else :
        usecols = None
    full_df = pd.read_csv(path_dataset_full, usecols = usecols + ['id', 'eid', 'Age when attended assessment centre']).set_index('id')
    if target_dataset is not None :
        target_dataset = target_dataset.replace('\\', '')
        Alan_residuals = load_target_residuals(target_dataset)
        full_df = full_df.join(Alan_residuals)
        full_df = full_df[~full_df['residuals'].isna()]

    print("Starting to convert columns to vectors")
    cols = [elem for elem in full_df.columns if elem not in cols_age_sex_eid_ethnicity + ['residuals']]
    for col in cols :
        if not full_df[col].dropna().empty :
            col_name = col
            dict_name_to_df[col_name] = full_df[[col, 'eid']].dropna()
            dict_name_to_num_features[col_name] = 1
        else :
            continue
    print("End dict series")
    df_age = full_df['Age when attended assessment centre'].index
    return dict_name_to_df, dict_name_to_num_features, df_age


def CreateDataArray(dict_name_to_df, ids):
    n = len(dict_name_to_df)
    dim = len(ids)
    array_fill_0 = np.zeros((n, dim), dtype = 'int')
    map_name_to_idx = dict(zip(dict_name_to_df.keys(), range(len(dict_name_to_df))))
    for name, elem in dict_name_to_df.items():
        idx = map_name_to_idx[name]
        array_fill_0[idx] = ids.isin(elem.index)
    return array_fill_0, map_name_to_idx


def FindArgmin(full_distance_matrix):
    argmin = full_distance_matrix.argmin()
    argmin_i, argmin_j = np.unravel_index(argmin, full_distance_matrix.shape)
    if argmin_i > argmin_j:
        argmin_i, argmin_j = argmin_j, argmin_i
    return argmin_i, argmin_j


def ReplaceIbyIJ(array_fill_0, dataset_ij, argmin_i, argmin_j, ids):
    ## Replace i by ij
    array_fill_0[argmin_i] = 0
    array_fill_0[argmin_i] = ids.isin(dataset_ij.index)
    ## Delete j
    array_fill_0 = np.delete(array_fill_0, argmin_j, axis = 0)
    return array_fill_0


def CreateNewMapping(map_idx_to_name, argmin_i, argmin_j):
    new_mapping = dict()
    for index in range(argmin_i):
        new_mapping[index] = map_idx_to_name[index]
    new_mapping[argmin_i] = map_idx_to_name[argmin_i] + '//' + map_idx_to_name[argmin_j]
    for index in range(argmin_i + 1, argmin_j):
        new_mapping[index] = map_idx_to_name[index]
    for index in range(argmin_j, len(map_idx_to_name) - 1):
        new_mapping[index] = map_idx_to_name[index + 1]

    map_idx_to_name = copy.deepcopy(new_mapping)
    map_name_to_idx = {v : k for k, v in map_idx_to_name.items()}
    return map_idx_to_name, map_name_to_idx


def RecomputeDistanceMatrix(full_distance_matrix, array_fill_0, argmin_i, argmin_j):
    full_distance_matrix = np.delete(np.delete(full_distance_matrix, argmin_j, axis = 0), argmin_j, axis = 1)

    new_point = array_fill_0[argmin_i]
    old_points = array_fill_0
    new_distance_matrix = ComputeDistanceMatrix(new_point[np.newaxis, :], old_points)

    full_distance_matrix[:, argmin_i] = new_distance_matrix[0, :]
    full_distance_matrix[argmin_i, :] = new_distance_matrix[0, :]
    full_distance_matrix[argmin_i, argmin_i] = np.inf

    return full_distance_matrix


def AglomerativeClusteringFull(path_input, target_dataset = None, env_dataset = None):
    ## Load eid and ids, compute max_size and min_size :
    dict_name_to_df, dict_name_to_num_features, ids = CreateDictSizes(path_input, target_dataset, env_dataset)
    ## Create Array with size vectors, and create mapping between idx and dataset names
    array_fill_0, map_name_to_idx = CreateDataArray(dict_name_to_df, ids)
    map_idx_to_name = {v : k for k, v in map_name_to_idx.items()}

    ## Initialise distance matrix
    full_distance_matrix = ComputeDistanceMatrix(array_fill_0, array_fill_0)
    np.fill_diagonal(full_distance_matrix, np.inf)
    print("Done computing full distance matrix ", full_distance_matrix)

    dict_not_changed_index = copy.deepcopy(map_idx_to_name)
    dict_not_changed_index_to_num_features = dict((map_name_to_idx[key], value) for (key, value) in dict_name_to_num_features.items())

    initial_dict = copy.deepcopy(dict_not_changed_index)
    n = array_fill_0.shape[0]
    Zdf = pd.DataFrame(columns = {'index_i', 'index_j', 'index_ij', 'num_features_i', 'num_features_j', 'num_features_ij', 'distance_ij', 'number_ij', 'samplesize_i','samplesize_j', 'samplesize_ij', 'name_i', 'name_j', 'name_ij', 'step'})
    for count in range(n - 1):
        if count % 10 == 0:
            print(count/(n-1))

        ## Find Argmin i and j
        argmin_i, argmin_j = FindArgmin(full_distance_matrix)

        ## Store Names :
        dataset_name_i, dataset_name_j = map_idx_to_name[argmin_i], map_idx_to_name[argmin_j]
        name_ij = dataset_name_i + '//' + dataset_name_j

        ## Store sample sizes
        row_i, row_j = array_fill_0[argmin_i], array_fill_0[argmin_j]
        samplesize_ij = row_i.dot(row_j)
        samplesize_i, samplesize_j = row_i.dot(row_i), row_j.dot(row_j)

        ## Store distance
        distance_ij = full_distance_matrix[argmin_i, argmin_j]

        ## Merge argmin_i and argmin_j
        dataset_i, dataset_j =  dict_name_to_df[dataset_name_i], dict_name_to_df[dataset_name_j]
        dataset_ij = dataset_i.join(dataset_j, how = 'inner', rsuffix = '_r').drop(columns = 'eid_r')
        dict_name_to_df[name_ij] = dataset_ij
        dict_name_to_df.pop(dataset_name_i, None)
        dict_name_to_df.pop(dataset_name_j, None)
        print("Merge %s and %s => %s " % (dataset_name_i, dataset_name_j, name_ij))

        ## Replace i by ij, delete j
        array_fill_0 = ReplaceIbyIJ(array_fill_0, dataset_ij, argmin_i, argmin_j, ids)

        ## New mapping
        map_idx_to_name, map_name_to_idx = CreateNewMapping(map_idx_to_name, argmin_i, argmin_j)

        ## Recompute distances with new cluster
        full_distance_matrix = RecomputeDistanceMatrix(full_distance_matrix, array_fill_0, argmin_i, argmin_j)

        ## Update saving index and creating final Z row:
        dict_not_changed_index[count + n] = name_ij
        i_index_not_changed = [key for key, value in dict_not_changed_index.items() if value == dataset_name_i][0]
        j_index_not_changed = [key for key, value in dict_not_changed_index.items() if value == dataset_name_j][0]
        number_in_ij = len(name_ij.split('//'))

        ## Store number of features
        features_i, features_j = dict_not_changed_index_to_num_features[i_index_not_changed], dict_not_changed_index_to_num_features[j_index_not_changed]
        features_ij = features_i + features_j
        dict_not_changed_index_to_num_features[count + n] = features_ij

        Zdf = Zdf.append({'index_i' : i_index_not_changed, 'index_j' : j_index_not_changed, 'index_ij' : count + n,
                          'num_features_i' : features_i, 'num_features_j' : features_j, 'num_features_ij' : features_ij,
                          'samplesize_ij' : samplesize_ij, 'samplesize_i' : samplesize_i, 'samplesize_j' : samplesize_j,
                          'name_i' : dataset_name_i, 'name_j' : dataset_name_j, 'name_ij' : name_ij,
                          'distance_ij': distance_ij, 'number_ij' : number_in_ij, 'step' : count + n
                         }, ignore_index = True)


    return dict_not_changed_index, Zdf, initial_dict


## Processing HC
def Create_full_linkage_matrix(Zdf, initial_dict):
    Z = Zdf[['index_i', 'index_j', 'distance_ij', 'number_ij']].astype(float).values
    tree = hierarchy.to_tree(Z)
    cols = list(initial_dict.values())
    linkage_matrix_raw = Zdf.set_index('index_ij')
    linkage_matrix_raw['Score_i'] = linkage_matrix_raw['samplesize_i'] * linkage_matrix_raw['num_features_i']
    linkage_matrix_raw['Score_j'] = linkage_matrix_raw['samplesize_j'] * linkage_matrix_raw['num_features_j']
    linkage_matrix_raw['Score_ij'] = linkage_matrix_raw['samplesize_ij'] * linkage_matrix_raw['num_features_ij']
    return tree, linkage_matrix_raw


def CreateMappingScoreAndFeatures(linkage_matrix_raw):
    mapping_index_to_score = {}
    mapping_index_to_features = {}
    first_node = linkage_matrix_raw.index[0]
    for elem in linkage_matrix_raw.index:
        mapping_index_to_score[elem] = linkage_matrix_raw.loc[elem, 'Score_ij']
        mapping_index_to_features[elem] = linkage_matrix_raw.loc[elem, 'num_features_ij']
    for index in range(first_node):
        if index in linkage_matrix_raw['index_i'].values:
            score = linkage_matrix_raw[linkage_matrix_raw.index_i == index]['Score_i'].values[0]
            feature = linkage_matrix_raw[linkage_matrix_raw.index_i == index]['num_features_i'].values[0]
        elif  index in linkage_matrix_raw['index_j'].values:
            score = linkage_matrix_raw[linkage_matrix_raw.index_j == index]['Score_j'].values[0]
            feature = linkage_matrix_raw[linkage_matrix_raw.index_j == index]['num_features_j'].values[0]
        mapping_index_to_score[index] = score
        mapping_index_to_features[index] = feature
    return mapping_index_to_score, mapping_index_to_features

## Find interesting nodes
def GetInterestingNodes(tree_, linkage_matrix_raw, printing = True):
    list_interesting = []
    def recurse(tree):
        score_parent = linkage_matrix_raw.loc[tree.get_id(), 'Score_ij']
        if printing:
            print(" PARENT : ", tree.get_id(), ", Score : ", score_parent)

        ## Compare parent and left child
        if not tree.get_left().is_leaf():
            score_left = linkage_matrix_raw.loc[tree.get_left().get_id(), 'Score_ij']
        else:
            row = linkage_matrix_raw.loc[tree.get_id()]
            if row['index_i'] == tree.get_left().get_id():
                score = row['Score_i']
            else :
                score = row['Score_j']
            score_left = score
        if printing:
            print(" CHILD LEFT : ", tree.get_left().get_id(), ", Score left : ", score_left)

        ## Compare parent and right child
        if  not tree.get_right().is_leaf():
            score_right = linkage_matrix_raw.loc[tree.get_right().get_id(), 'Score_ij']
        else :
            row = linkage_matrix_raw.loc[tree.get_id()]
            if row['index_i'] == tree.get_right().get_id():
                score = row['Score_i']
            else :
                score = row['Score_j']
            score_right = score
        if printing:
            print(" CHILD RIGHT : ", tree.get_right().get_id(), ", Score right : ", score_right)

        ## Append interesting nodes
        if score_right > score_parent:
            list_interesting.append(tree.get_right().get_id())
        if score_left > score_parent:
            list_interesting.append(tree.get_left().get_id())

        ## Explore left and right if possible
        if not tree.get_left().is_leaf():
            recurse(tree.get_left())
        if not tree.get_right().is_leaf():
            recurse(tree.get_right())

        return list_interesting

    list_interesting =  recurse(tree_)
    if list_interesting == [] :
        return [linkage_matrix_raw.index.max()]
    else :
        return list_interesting

def CreateBestClusterFromInterestingNodes(list_interesting, linkage_matrix_raw, path_input, path_clusters, target, env_df):
    print("linkage_matrix_raw", linkage_matrix_raw)
    try :
        df_sorted = linkage_matrix_raw.loc[list_interesting].sort_values('Score_ij', ascending = False)
    except KeyError:
        df_sorted = linkage_matrix_raw.sort_values('Score_ij', ascending = False)
    best_cluster = df_sorted.iloc[0]
    list_features = best_cluster['name_ij'].split('//')
    df_cluster = pd.read_csv(path_input, usecols = ['id'] + list_features ).set_index('id')
    df_sex_age_ethnicity = pd.read_csv('/n/groups/patel/Alan/Aging/Medical_Images/data/data-features_instances.csv').set_index('id').drop(columns = ['Abdominal_images_quality', 'instance', 'outer_fold'])
    df_sex_age_ethnicity = df_sex_age_ethnicity.rename(columns = {'Age' : 'Age when attended assessment centre'})
    df_cluster = df_cluster.join(df_sex_age_ethnicity)
    df_cluster.to_csv(path_clusters + 'Clusters_%s_%s.csv' % (env_df, target))


def CreateClustersFromInterestingNodes(list_interesting, linkage_matrix_raw, path_input, path_clusters, target = None):
    ## EWAS :
    if target is not None:
        os.mkdir(path_clusters + target )
        path_saving = path_clusters + target + '/'
    ## Biomarkers
    else :
        path_saving = path_clusters
    ## Compute interesting Nodes
    for node_id in list_interesting:
        print(node_id)
        features = linkage_matrix_raw.loc[node_id, 'name_ij']
        score = linkage_matrix_raw.loc[node_id, 'Score_ij']
        num_features = linkage_matrix_raw.loc[node_id, 'num_features_ij']
        print(features)
        list_features = features.split('//')
        df_cluster = pd.read_csv(path_input, usecols = ['id'] + list_features ).set_index('id') ## Remember to drop nas
        df_sex_age_ethnicity = pd.read_csv('/n/groups/patel/Alan/Aging/Medical_Images/data/data-features_instances.csv').set_index('id').drop(columns = ['Abdominal_images_quality', 'instance', 'outer_fold'])
        df_sex_age_ethnicity = df_sex_age_ethnicity.rename(columns = {'Age' : 'Age when attended assessment centre'})
        df_cluster = df_cluster.join(df_sex_age_ethnicity)
        df_cluster.to_csv(path_saving + '/Cluster_score_%s_numfeatures_%s.csv' % (score, num_features))
