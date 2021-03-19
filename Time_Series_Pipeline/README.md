# TS pipeline
Pipeline for time-series analysis.

## TS_ressources.py
Script defining classes to build and train models, and make predictions:
- hyperparameter: contains the set of parameters to build and train models
- TS_model_architecture: creates model architecture
- TS_model: inherits from TS_model_architecture ; builds the model architecture, enables to compile it, fit it to data and make predictions

It also defines the class used for visualization purpose (generating saliency maps).

## TS_preprocessing.py
Script defining classes to load, preprocess and generate data:
- DataFetcher: loads the right set of data according to the given parameters
- DataPreprocessor: preprocesses data according to the chosen model
- DataGenerator: generates data during training
- DataGeneratorForPredictions: generates data for predictions

## TS_training.py
Script to train a model for a given set of parameters stored in sys.argv.

Save model performances in a csv file.

## TS_prediction.py
Script to generate predictions for a given set of parameters stored in sys.argv.


