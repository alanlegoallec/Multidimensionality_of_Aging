from .base_processing import read_data

"""
Features used :
	100009 - Impedance measurements
	100010 - BodySize
	Errors features : None
	Missing : 23105
"""

def read_anthropometry_impedance_data(**kwargs):

	"""
	23106	Impedance of whole body
	23110	Impedance of arm (left)
	23109	Impedance of arm (right)
	23108	Impedance of leg (left)
	23107	Impedance of leg (right)
	"""
	cols_features = ['23106', '23107', '23108', '23109',
    '23110' ]
	cols_filter = []
	instance = [0, 1, 2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)


def read_anthropometry_body_size_data(**kwargs):
	"""
	48	Waist circumference
	21002	Weight
	21001	Body mass index (BMI)
	49	Hip circumference
	50	Standing height
	20015	Seatting height
	51	Seated height
	20015	Sitting height
	3077	Seating box height
	"""
	cols_features = ['48', '21002', '21001', '49', '50', '20015', '51', '3077']
	cols_filter = []
	instance = [0, 1, 2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)

# def read_anthropometry_body_size_data(**kwargs):
# 	"""
#
# 	48	Waist circumference
# 	21002	Weight
# 	21001	Body mass index (BMI)
# 	49	Hip circumference
# 	50	Standing height
# 	20015	Seatting height
# 	"""
# 	cols_features = ['48', '21002', '21001', '49', '50', '20015']
# 	cols_filter = []
# 	instance = [0, 1, 2, 3]
# 	return read_data(cols_features, cols_filter, instance, **kwargs)

def read_anthropometry_data(**kwargs):
	cols_features_body = ['48', '21002', '21001', '49', '50', '20015', '51', '3077']
	cols_features_imp = ['23106', '23107', '23108', '23109',
    '23110' ]
	cols_features = cols_features_imp + cols_features_body
	cols_filter = []
	instance = [0, 1, 2, 3]
	return read_data(cols_features, cols_filter, instance, **kwargs)
