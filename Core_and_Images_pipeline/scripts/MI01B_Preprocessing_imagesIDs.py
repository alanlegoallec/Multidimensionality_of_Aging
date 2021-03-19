import sys
from MI_Classes import PreprocessingImagesIDs

# Compute results
Preprocessing_ImagesIDs = PreprocessingImagesIDs()
Preprocessing_ImagesIDs.generate_eids_splits()

# Exit
print('Done.')
sys.exit(0)
