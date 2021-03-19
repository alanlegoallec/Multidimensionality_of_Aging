import sys
from MI_Classes import PerformancesGenerate

# Options
# Use a small number for the bootstrapping
debug_mode = True

# Default parameters
if len(sys.argv) != 15:
    print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
    sys.argv = ['']
    sys.argv.append('Age')  # target
    sys.argv.append('Arterial')  # organ
    sys.argv.append('Carotids')  # view
    sys.argv.append('Scalars')  # transformation
    sys.argv.append('ElasticNet')  # architecture
    sys.argv.append('0')  # n_fc_layers
    sys.argv.append('0')  # n_fc_nodes
    sys.argv.append('0')  # optimizer
    sys.argv.append('0')  # learning_rate
    sys.argv.append('0')  # weight decay
    sys.argv.append('0')  # dropout
    sys.argv.append('0')  # data_augmentation_factor
    sys.argv.append('val')  # fold
    sys.argv.append('instances')  # pred_type


# Default parameters for ensemble models
# if len(sys.argv) != 15:
#     print('WRONG NUMBER OF INPUT PARAMETERS! RUNNING WITH DEFAULT SETTINGS!\n')
#     sys.argv = ['']
#     sys.argv.append('Age')  # target
#     sys.argv.append('*')  # organ
#     sys.argv.append('*')  # view
#     sys.argv.append('*')  # transformation
#     sys.argv.append('*')  # architecture
#     sys.argv.append('*')  # n_fc_layers
#     sys.argv.append('*')  # n_fc_nodes
#     sys.argv.append('*')  # optimizer
#     sys.argv.append('*')  # learning_rate
#     sys.argv.append('*')  # weight_decay
#     sys.argv.append('*')  # dropout_rate
#     sys.argv.append('*')  # data_augmentation_factor
#     sys.argv.append('val')  # fold
#     sys.argv.append('instances')  # pred_type


# Compute results
Performances_Generate = PerformancesGenerate(target=sys.argv[1], organ=sys.argv[2], view=sys.argv[3],
                                             transformation=sys.argv[4], architecture=sys.argv[5],
                                             n_fc_layers=sys.argv[6], n_fc_nodes=sys.argv[7], optimizer=sys.argv[8],
                                             learning_rate=sys.argv[9], weight_decay=sys.argv[10],
                                             dropout_rate=sys.argv[11], data_augmentation_factor=sys.argv[12],
                                             fold=sys.argv[13], pred_type=sys.argv[14], debug_mode=False)
Performances_Generate.preprocessing()
Performances_Generate.compute_performances()
Performances_Generate.save_performances()

# Exit
print('Done.')
sys.exit(0)
