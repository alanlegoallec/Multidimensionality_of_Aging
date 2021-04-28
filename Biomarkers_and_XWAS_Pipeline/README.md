# Multi-Dimensionality of Aging

Source code of the publication **“Analyzing the multidimensionality of biological aging with the tools of deep learning across diverse image-based and physiological indicators yields robust age predictors”**.

The source code contains only the part referring to tabular data : 
- the XWAS pipeline (X-Wide Association Study) 
- the Aging pipeline corresponding to the Tabular Biomarkers (e.g. blood biomarkers, anthropometric measures, ...)

More details and results can be found on the paper # LINK or on the website https://www.multidimensionality-of-aging.net/

## Description of the source code.
The code is separated in two parts : 
- The first part **aging** contains the source code of the models trained for the project. 
- The second part **batch_jobs** groups the scripts of the different experiments. These scripts were run on HMS supercomputer O2 using single core CPUs.

The code is able to run the experiments for the XWAS and also for the Aging project.

## Inputing missing data and Creating Clusters
As described in the paper, for the XWAS project, we needed to input missing data and create clusters so as to use them as our inputs to the models. 

For the inputing part, we used Longitudinal Inputting (more details can be found on the supplemental of the paper). The source code can be found in the file **aging/model/InputtingNans.py** and the scripts of the experiments can be found in the folder **batch_jobs/input_ewas/parallel.sh**.

For the Clustering part, we created clusters using Hierarchical Clustering. The source code of this part can be found in the file **aging/model/HC.py** and the script launching the experiments can be found at **Aging/batch_jobs/ClusterEWAS/parallel.sh**

## Loading and preprocessing 

For both subprojects, we loaded and preprocessed the data. 

The associated python files are located in the folder **aging/processing** or **aging/environment_processing** (depending on whether we want to do XWAS or Aging). 
One can find exactly how we loaded the dataset and how we preprocessed them.


## Building Estimators and Training
Both subprojects XWAS and Aging use the file **general_predictor.py** which is the core of the code. This file creates a base model (BaseModel class in the code) containing several estimators and train them accordingly using a nested cross validation. (More details can be found on the paper).

Then, depending to whether we want to run an XWAS experiment or an Aging experiment, we build associated classes : 

- EnvironmentPredictor in the file **environment_predictor.py** for the XWAS
- SpecificPredictor in the file **specific_predictor.py** for the Aging 

Each model handles the loading and the rescaling of the data, the training of the models (using the nested cross validation) and also the generation of the features importances which are defined differently depending on the model (Neural Networks, Boosted Trees or Linear Regression). This definition of the feature importances can also be found more precisely in the paper.







