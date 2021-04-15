# Heart videos modelling

In this repository, deep learning models are used to do a regression of chronological age on raw MRI videos of the cardiac system.
46 thousands videos are processed through an architecture based mainly on 3D convolutions. The nature of such layer enables not only to manage spatial information (as classical 2D convolution) but also to integrate temporal evolution (third dimension) to the prediction of ageing.

The videos consist of 25 frames (relative to the duration of an heart cycle). The imagery offers two perspectives to observed the heart :
 - 3 chambers (3ch) view : the right atrium and ventricule plus the left atrium are visible.
 - 4 chambers (4ch) view : atria and ventricles of both sides are visible.

A single architecture was used to models both views and each folds

Data was split in a ten folds nested cross validation such that each patient is in a testing set. Hence, unbiased prediction would be used to explore the ageing profiles and their correlation with other predictions, biomarkers and genomic data.

The repo contains the files used for the training, the conmputation of the predcitions and the post processing of the 20 models. An ensemble model is build on top of the predictions for each fold.

**Files** :

_ `Heart_models_scores.ipynb` : notebook gathering all the main information and managing the whole process. Training can be launched and monitored, predictions are gathers and compared, final scores are computed.
_ `Image_preprocessing_heart` : the raw images in `DICOM` format are cleaned, reformatted and stored in a pythonic fashion to optimised storage of the videos and code flexibility.
_ `Image_preprocessing_liver` : cleaning and reformatting of the liver images.
_ `Call_predictions.sh` : main bash script to compute the predictions across all folds and views
_ `Call_train.sh` : main bash script to train models across all folds and views
_ `Get_predictions.py` : python script for computing the predictions
_ `Get_predictions.sh` : bash script calling the `Get_predictions.py` script
_ `Train.py` : python script managing the training
_ `Train.sh` : bash script calling the `Train.py` script
_ 3ch : three chambers view
	|_ Fold0 :
		|_ `testpredictions_augmented.csv`
		|_ `trainpredictions_augmented.csv`
		|_ `valpredictions_augmented.csv`
	|_ Fold1 …
_ 4ch : four chambers view
	|_ Fold0 :
		|_ `testpredictions_augmented.csv`
		|_ `trainpredictions_augmented.csv`
		|_ `valpredictions_augmented.csv`
	|_ Fold1…
_ Helpers : helper functions
	|_ `DataFunctions.py` : contains mainly the DataGenerator class managing the loading and augmentation of the videos
	|_ `EvaluatingFunctions.py` : functions used to monitor training evolution and model's performance.
	|_ `EvaluatingModels.py` : functions used to manage training and retrieve store predictions
    
Author : Jean-Baptiste Prost,
Year 2020