from .base_processing import read_data
import pandas as pd

"""
Features used :
	104 - ECG at rest
	Errors features : '12657'
	Missing : None
"""

def read_ecg_at_rest_data(**kwargs):
	"""
	12323	12-lead ECG measuring method
	12657	Suspicious flag for 12-lead ECG
	12653	ECG automated diagnoses
	12654	Number of automated diagnostic comments recorded during 12-lead ECG
	12336	Ventricular rate
	12338	P duration
	22334	PP interval
	22330	PQ interval
	22338	QRS num
	12340	QRS duration
	22331	QT interval
	22332	QTC interval
	22333	RR interval
	22335	P axis
	22336	R axis
	22337	T axis
	12658	Identifier for 12-lead ECG device
	"""

	cols_features =  ['12336', '12338', '12340', '22330', '22331', '22332', '22333', '22334', '22335', '22336', '22337', '22338']
	cols_filter = ['12657']
	instance = [2]
	df_2 = read_data(cols_features, cols_filter, instance, **kwargs)

	cols_features =  ['12336', '12338', '12340']
	cols_filter = []
	instance = [3]
	df_3 = read_data(cols_features, cols_filter, instance, **kwargs)

	return pd.concat([df_2, df_3])
