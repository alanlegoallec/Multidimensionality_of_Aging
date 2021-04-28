from .base_processing import read_data

"""
Features used :
	717 - Biochemestry Markers
	Errors features : None
	Missing : None
"""

## Simple
def read_blood_biomarkers_data(**kwargs):
	cols_features = list(range(30600, 30890 + 1, 10))
	cols_filter = [ ]
	instance = [0, 1]
	cols_features.remove(30820)
	cols_features.remove(30800)
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_blood_count_data(**kwargs):
	cols_features = list(range(30000, 30300 + 1, 10))
	cols_filter = [ ]
	instance = [0, 1, 2]
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_urine_biomarkers_data(**kwargs):
	cols_features = list(range(30500, 30530 + 1, 10))
	cols_filter = [ ]
	instance = [0, 1]
	return read_data(cols_features, cols_filter, instance, **kwargs)

## Aggregate
# Mid
def read_blood_data(**kwargs):
	cols_features = list(range(30000, 30300 + 1, 10)) + list(range(30600, 30890 + 1, 10))
	cols_filter = [ ]
	cols_features.remove(30820)
	cols_features.remove(30800)
	instance = [0, 1]
	return read_data(cols_features, cols_filter, instance, **kwargs)

# Super Aggregate
def read_urine_and_blood_data(**kwargs):
	cols_features = list(range(30000, 30300 + 1, 10)) + list(range(30500, 30530 + 1, 10)) + list(range(30600, 30890 + 1, 10))
	cols_filter = [ ]
	cols_features.remove(30820)
	cols_features.remove(30800)
	instance = [0, 1]
	return read_data(cols_features, cols_filter, instance, **kwargs)
