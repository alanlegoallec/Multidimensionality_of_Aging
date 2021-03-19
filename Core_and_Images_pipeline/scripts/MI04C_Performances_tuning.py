import sys
from MI_Classes import PerformancesTuning

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('eids')  # pred_type

# Compute results
Performances_Tuning = PerformancesTuning(target=sys.argv[1], pred_type=sys.argv[2])
Performances_Tuning.load_data()
Performances_Tuning.preprocess_data()
Performances_Tuning.select_models()
Performances_Tuning.save_data()

# Exit
print('Done.')
sys.exit(0)
