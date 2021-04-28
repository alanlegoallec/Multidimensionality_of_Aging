from .base_processing import read_data

"""
Features used :
	124 - Body Composition
	Errors features : None
	Missing : None
"""

def read_body_composition_data(**kwargs):
	cols_features = list(elem in range(23244, 23289 + 1))
	cols_filter = [ ]
	instance = 2
	return read_data(cols_features, cols_filter, instance, **kwargs)
