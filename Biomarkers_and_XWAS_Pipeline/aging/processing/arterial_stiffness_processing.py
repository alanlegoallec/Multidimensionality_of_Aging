from .base_processing import read_data

"""
Features used :
	100007 - Arterial Stiffness
	Errors features : None
	Missing : None
"""
"""
4204	Absence of notch position in the pulse waveform
4199	Position of pulse wave notch
4198	Position of the pulse wave peak
4200	Position of the shoulder on the pulse waveform
4194	Pulse rate
21021	Pulse wave Arterial Stiffness index
4196	Pulse wave peak to peak time
4195	Pulse wave reflection index
#4186	Stiffness method
"""

def read_arterial_stiffness_data(**kwargs):
    cols_features =  ['4194', '4195',  '4196', '4198', '4199',
                        '4200',  '4204', '21021']
    instance = [0, 1, 2, 3]
    cols_filter = []
    return read_data(cols_features, cols_filter, instance, **kwargs)
