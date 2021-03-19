import sys
from MI_Classes import PreprocessingMain

# Compute results
Preprocessing_Main = PreprocessingMain()
Preprocessing_Main.generate_data()
Preprocessing_Main.save_data()

# Exit
print('Done.')
sys.exit(0)
