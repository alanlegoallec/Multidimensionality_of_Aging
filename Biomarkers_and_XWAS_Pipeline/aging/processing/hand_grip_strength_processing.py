from .base_processing import read_data

"""
46	Hand grip strength (left)
47	Hand grip strength (right)
"""

def read_hand_grip_strength_data(**kwargs):
    cols_features = [46, 47]
    cols_filter = []
    instance = [0, 1, 2, 3]
    return read_data(cols_features, cols_filter, instance, **kwargs)
