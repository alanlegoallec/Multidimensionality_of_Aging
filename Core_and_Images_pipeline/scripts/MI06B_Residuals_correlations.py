import sys
from MI_Classes import ResidualsCorrelations

# Default parameters
if len(sys.argv) != 4:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('val')  # fold
    sys.argv.append('instances')  # pred_type

# Options
debug_mode = True

# Compute results
Residuals_Correlations = ResidualsCorrelations(target=sys.argv[1], fold=sys.argv[2], pred_type=sys.argv[3],
                                               debug_mode=debug_mode)
Residuals_Correlations.preprocessing()
Residuals_Correlations.generate_correlations()
Residuals_Correlations.save_correlations()

# Exit
print('Done.')
sys.exit(0)
