IDs_01A=$(sbatch MI01A_Preprocessing_main.sh)
IDs_01B=$(sbatch MI01B_Preprocessing_imagesIDs.sh --dependency=$IDs_01A)
IDs_01C=$(./MI01C_Preprocessing_folds_parallel.sh $IDs_01B)

