import sys
from MI_Classes import PerformancesMerge
    
# Default parameters
if len(sys.argv) != 5:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('val')  # inner_fold
    sys.argv.append('eids')  # pred_type
    sys.argv.append('False')  # ensemble_models. Set False for MI04B and True for MI05B

# Compute results
Performances_Merge = PerformancesMerge(target=sys.argv[1], fold=sys.argv[2], pred_type=sys.argv[3],
                                       ensemble_models=sys.argv[4])
Performances_Merge.merge_performances()
Performances_Merge.save_performances()

# Exit
print('Done.')
sys.exit(0)
