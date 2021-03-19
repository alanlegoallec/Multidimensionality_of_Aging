import sys
from MI_Classes import SelectBest

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('eids')  # pred_type

# Compute results
Select_Best = SelectBest(target=sys.argv[1], pred_type=sys.argv[2])
Select_Best.select_models()
Select_Best.save_data()

# Exit
print('Done.')
sys.exit(0)
