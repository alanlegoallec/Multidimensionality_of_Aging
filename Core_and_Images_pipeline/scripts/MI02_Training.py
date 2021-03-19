import sys
from MI_Classes import Training

# Options
# Use a small subset of the data VS. run the actual full data pipeline to get accurate results
# /!\ if True, path to save weights will be automatically modified to avoid rewriting them
debug_mode = False
# Load weights from previous best training results, VS. start from scratch
continue_training = True
# Try to find a similar model among those already trained and evaluated to perform transfer learning
transfer_learning = None  # None, hyperparameters (same dataset, other hyperparameters), datasets (other datasets).
# Compute all the metrics during training VS. only compute loss and main metric (faster)
display_full_metrics = False

# Default parameters
if len(sys.argv) != 14:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Arterial')  # organ
    sys.argv.append('Carotids')  # view
    sys.argv.append('Mixed')  # transformation
    sys.argv.append('InceptionV3')  # architecture
    sys.argv.append('1')  # n_fc_layers
    sys.argv.append('1024')  # n_fc_nodes
    sys.argv.append('Adam')  # optimizer
    sys.argv.append('0.0001')  # learning_rate
    sys.argv.append('0.1')  # weight decay
    sys.argv.append('0.5')  # dropout_rate
    sys.argv.append('1.0')  # data_augmentation_factor
    sys.argv.append('0')  # outer_fold

# Compute results
Model_Training = Training(target=sys.argv[1], organ=sys.argv[2], view=sys.argv[3], transformation=sys.argv[4],
                          architecture=sys.argv[5], n_fc_layers=sys.argv[6], n_fc_nodes=sys.argv[7],
                          optimizer=sys.argv[8], learning_rate=sys.argv[9], weight_decay=sys.argv[10],
                          dropout_rate=sys.argv[11], data_augmentation_factor=sys.argv[12], outer_fold=sys.argv[13],
                          debug_mode=debug_mode, continue_training=continue_training,
                          transfer_learning=transfer_learning, display_full_metrics=display_full_metrics)
Model_Training.data_preprocessing()
Model_Training.build_model()
Model_Training.train_model()
Model_Training.clean_exit()
