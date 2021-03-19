import sys
from MI_Classes import GWASPreprocessing

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Compute results
GWAS_Preprocessing = GWASPreprocessing(target=sys.argv[1])
GWAS_Preprocessing.compute_gwas_inputs()

# Exit
print('Done.')
sys.exit(0)
