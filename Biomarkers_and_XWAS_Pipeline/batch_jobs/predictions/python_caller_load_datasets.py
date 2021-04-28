import sys
import os

dataset_name = sys.argv[1]

if sys.platform == 'linux':
    sys.path.append('/n/groups/patel/samuel/Aging')
elif sys.platform == 'darwin':
    sys.path.append('/Users/samuel/Desktop/Aging')

from aging.model.load_and_save_data import create_data

create_data(dataset_name)
