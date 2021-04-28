
from .base_processing import read_data

"""
Features used :
	149 - Abdominal Composition : 22403, 22404, 22405, 22406, 22407, 22408, 22409, 22410,  22415, 22416
	Errors features : 22411, 22412, 22413, 22414
	Missing : 22432, 22433, 22434, 22435, 22436
"""

def read_abdominal_data(**kwargs):
	cols_features = ['22403-2.0', '22404-2.0', '22405-2.0', '22406-2.0', '22407-2.0', '22408-2.0',
	'22409-2.0', '22410-2.0', '22415-2.0', '22416-2.0']


	cols_filter = ['22411-2.0', '22412-2.0', '22413-2.0', '22414-2.0']
	instance = 2
	return read_data(cols_features, cols_filter, instance, **kwargs)
