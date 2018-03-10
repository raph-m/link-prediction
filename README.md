# link-prediction
Predict links in a citation network.

## Feature Engineering
In the feature engineering folder you can find scripts to compute new features from the dataset. The features computed are described at the beginning of the scripts, and you can find more information in our project report.

## Feature selection
Running the feature_selection.py script will print the results of a forward selection algorithm. We chose the set of features that we were going to use for the rest of the project from these results.

## Models
You can find several implementations of models to fit to our data. Running the scripts will give you the results and create the submission file.

## Tuning
Runing the tuning scripts will print the results of a grid search to tune the parameters of our models.

## Main
The main.py script processes all you need (feature engineering and machine learning) in order to create our final submission. It will take a very long time to process so you might want to comment the feature engineering for "shortest_path" and "katz".

