import sys
from MI_Classes import GWASPlots

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Compute results
GWAS_Plots = GWASPlots(target=sys.argv[1])
GWAS_Plots.generate_manhattan_and_qq_plots()

# Exit
print('Done.')
sys.exit(0)
