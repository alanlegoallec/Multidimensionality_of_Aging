from .base_processing import read_data

"""
Features used :
	125 - Bone composition
	Errors features : None
	Missing : 23207, 23294, 23303, 23211
"""

def read_bone_composition_data(**kwargs):
	missing =  [23207, 23294, 23303, 23211]
	cols_features = list(range(23200, 23243 + 1)) + list(range(23290, 23318 + 1)) + [23320]
	cols_filter = [ ]
	instance = 2
	return read_data([elem for elem in cols_features if elem not in missing], cols_filter, instance, **kwargs)
