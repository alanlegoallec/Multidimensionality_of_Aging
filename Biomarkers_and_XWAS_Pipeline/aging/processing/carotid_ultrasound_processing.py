from .base_processing import read_data


"""
Category 101
Datafields :
22672	Maximum carotid IMT (intima-medial thickness) at 120 degrees
22675	Maximum carotid IMT (intima-medial thickness) at 150 degrees
22678	Maximum carotid IMT (intima-medial thickness) at 210 degrees
22681	Maximum carotid IMT (intima-medial thickness) at 240 degrees
22671	Mean carotid IMT (intima-medial thickness) at 120 degrees
22674	Mean carotid IMT (intima-medial thickness) at 150 degrees
22677	Mean carotid IMT (intima-medial thickness) at 210 degrees
22680	Mean carotid IMT (intima-medial thickness) at 240 degrees
22670	Minimum carotid IMT (intima-medial thickness) at 120 degrees
22673	Minimum carotid IMT (intima-medial thickness) at 150 degrees
22676	Minimum carotid IMT (intima-medial thickness) at 210 degrees
22679	Minimum carotid IMT (intima-medial thickness) at 240 degrees

"""


def read_carotid_ultrasound_data(**kwargs):
    cols_features = list(range(22670, 22681 + 1))
    cols_filter = [ ]
    instance = [2, 3]
    return read_data(cols_features, cols_filter, instance, **kwargs)
