from .base_processing import read_data

"""
Features used :
	1102 - Subcortical volumes (FIRST) : 25011 -> 25024
	1101 - Regional grey matter volumes (FAST)  : 25782 -> 25920
	135 - dMRI Weighted Means : 25488 -> 25730
	Errors features : None
	Missing : None
"""


def read_grey_matter_volumes_data(**kwargs):
	cols_features = list(range(25782, 25920 + 1))
	cols_filter = [ ]
	instance = [2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_subcortical_volumes_data(**kwargs):
	cols_features = list(range(25011, 25024 + 1))
	cols_filter = [ ]
	instance = [2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_brain_dMRI_weighted_means_data(**kwargs):
	cols_features = list(range(25488, 25730 + 1))
	cols_filter = [ ]
	instance = [2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_brain_data(**kwargs):
	cols_features = list(range(25011, 25024 + 1)) + list(range(25782, 25920 + 1)) + list(range(25488, 25730 + 1))
	cols_filter = [ ]
	instance = [2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)
