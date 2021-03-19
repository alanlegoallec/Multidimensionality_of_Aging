import sys
from MI_Classes import GWASPostprocessing

# Default parameters
if len(sys.argv) != 2:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target

# Compute results
GWAS_Postprocessing = GWASPostprocessing(target=sys.argv[1])
GWAS_Postprocessing.processing_all_organs()
GWAS_Postprocessing.parse_heritability_scores()
GWAS_Postprocessing.parse_genetic_correlations()

# Exit
print('Done.')
sys.exit(0)
