import sys
from MI_Classes import EnsemblesPredictions

# Options
regenerate_models = False  # False = Only compute ensemble model if it was not already computed

# Default parameters
if len(sys.argv) != 3:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('eids')  # pred_type

# Compute results
Ensembles_Predictions = EnsemblesPredictions(target=sys.argv[1], pred_type=sys.argv[2],
                                             regenerate_models=regenerate_models)
Ensembles_Predictions.load_data()
Ensembles_Predictions.generate_ensemble_predictions()
Ensembles_Predictions.save_predictions()

# Exit
print('Done.')
sys.exit(0)
