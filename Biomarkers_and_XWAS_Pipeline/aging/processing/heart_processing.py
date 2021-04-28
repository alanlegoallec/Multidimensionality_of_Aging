from .base_processing import read_data, read_data_and_merge_temporal_features
"""
Features used :
	133 - Left ventricular size and function : 22420, 22421, 22422, 22423, 22424, 22425, 22426, 22427
	128 - Pulse wave analysis : 12673, 12674, 12675, 12676, 12677, 12678, 12679, 12680,
		  12681, 12682, 12683, 12684, 12686, 12687, 12697, 12698, 12699
"""

def read_heart_data(**kwargs):
	a = read_heart_size_data(**kwargs)
	b = read_heart_PWA_data(**kwargs)
	return a.join(b, rsuffix = '_del', lsuffix = '', how = 'inner').drop(columns = ['Age when attended assessment centre_del', 'Sex_del', 'eid_del'])

def read_heart_size_data(**kwargs):
	cols_features = [22420, 22421, 22422, 22423, 22424, 22425, 22426, 22427]
	cols_filter = []
	instance = [2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_heart_PWA_data(**kwargs):
	cols_features = [12673, 12674, 12675, 12676, 12677, 12678, 12679, 12680, 12681, 12682, 12683, 12684, 12685, 12686, 12687]
	timesteps = 4
	instance = [2, 3]
	return read_data_and_merge_temporal_features(cols_features, timesteps, instance, **kwargs)
