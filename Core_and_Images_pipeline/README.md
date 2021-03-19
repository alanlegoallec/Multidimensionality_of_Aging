Predicting Age using different types of medical images from UKB, and deep learning.
This pipeline generates the images-based predictors, then merges them and ensemble them with the predictors build on scalar data (e.g blood biomarkers), time series data (Pulse Wave Analysis, ECG, accelerometer based-Physical Activity ) and videos (Heart MRI). Finally, it performs the postprocessing (e.g GWAS on accelerated aging phenotypes).


There are six directories.

scripts: where the python scripts can be found. This is the core of the pipeline.

bash: Where the bash scripts to submit the jobs using slurm can be found. These are the wrapper that allow the pipeline to be run in parallel. Usually one script is named parallel and is going to call the script with the same name without "\_parallel" to submit a job that will be calling the python scripts with different arguments.

eo: Where the error and output files are stored, to keep track of successful and failed jobs, and to help with debugging.

data: where the results generated by the pipeline are saved.

images: Where the images used to predict the phenotypes are stored. They must be stored in a 4 levels/folders arborescence. The first level is the aging main dimension (called the organ), the second level is aging subdimension (called the view) and the third level is the aging sub-subdimension (called the transformation) (for example for Heart_20204 three different axis views are available).

figures: where the figures generated by the pipeline are saved.


The following are the steps that should be taken to run the pipeline: 01A, 01B, 01C, 01D, 02, 03A, 03B, 03C, 04A, 04B05C, 04C, 05B, 06A, 06B, 06C, 07A, 07B, 08, 09A, 09B, 09C, 09D, 09E, 09G
The function of these different steps is described below:

MI01A_Preprocessing_main.sh
Preprocesses the main dataframe.
 
 MI01B_Preprocessing_imagesIDs.sh
 Splits the different images datasets into folds for the future cross validation.

 MI01C_Preprocessing_folds_parallel.sh
 Splits the data into training, validation and testing sets for all CV folds.

 MI01D_Preprocessing_survival.sh
 Preprocesses the main dataframe for survival purposes.

 MI02_Training_parallel.sh
 Trains the models. This step should be repeated until all models have converged.

 MI03A_Predictions_generate_parallel.sh
 Generates the predictions from all models.

 MI03B_Predictions_concatenate_parallel.sh
 Concatenates the predictions from the different cross-validation folds.

 MI03C_Predictions_merge_parallel.sh
 Merges the predictions from all models into a unified dataframe.

 MI03D_Predictions_eids_parallel.sh
 Computes the average age prediction across samples from different instances for every participant.

 MI04A_Performances_generate_parallel.sh
 Computes the performances for each model.

 MI04B05C_Performances_merge_parallel.sh
 Merges the performances of the different models into a unified dataframe.

 MI04C_Performances_tuning_parallel.sh
 For each model, selects the best hyperparameter combination.

 MI05A_Ensembles_predictions_generate_and_merge_parallel.sh
 Hierarchically builds ensemble models.

 MI05B_Ensembles_performances_generate_parallel.sh
 Computes the performances for the ensemble models.

 MI06A_Residuals_generate_parallel.sh
 Computes accelerated aging phenotypes (Residuals, corrected for residuals bias with respect to age).

 MI06B_Residuals_correlations_parallel.sh
 Computes the phenotypic correlation between aging dimensions.

 MI06C_Performances_survival_parallel.sh
 Computes the performances in terms of survival prediction using biological age phenotypes as survival predictors.

 MI07A_Select_best_parallel.sh
 For each aging main dimension and selected subdimensions, select the best performing model.

 MI07B_Select_correlationsNAs_parallel.sh
 Build a summary correlation matrix: when a correlation cannot be computed in terms of samples ("instances") because the intersection has a small sample size, fill the NA with the correlation computed at the participant's level ("eids").

 MI08_Attentionmaps_parallel.sh
 Computes the attention maps (saliency maps and Grad_RAM maps) for all images.

 MI09A_GWAS_preprocessing_parallel.sh
 Preprocesses the data for the GWASs.

 MI09B_GWAS_remove_parallel.sh
 Runs a first pass of GWASs using BOLT to automatically generate the list of samples to remove for the actual GWASs.

 MI09C_GWAS_bolt_parallel.sh
 Performs the GWAS using BOLT to identify the SNPs, genes et heritability for each accelerated aging dimension.

 MI09D_GWAS_correlations_parallel.sh
 Computes the genetic correlations using BOLT.

 MI09E_GWAS_postprocessing_parallel.sh
 Postprocesses the GWAS results and stores the results in summary files.

 MI09F Annotation of the GWAS results
 /!\ Corresponds to a step in the pipeline that should be performed on local machine, since it must be complemented with researches on the internet for different steps. Refer to the python script MI09F_GWAS_annotate.py /!\ Annotates the hits from the GWAS: names of the genes and gene types.

 MI09G_GWAS_plots_parallel.sh
 Generates Manhattan and QQ plots to summarize the GWASs results.
