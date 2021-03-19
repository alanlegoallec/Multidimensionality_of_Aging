import sys
from MI_Classes import SelectCorrelationsNAs

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Compute results
Select_CorrelationsNAs = SelectCorrelationsNAs(target=sys.argv[1])
Select_CorrelationsNAs.load_data()
Select_CorrelationsNAs.fill_na()
Select_CorrelationsNAs.save_correlations()

# Exit
print('Done.')
sys.exit(0)
