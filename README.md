# link-prediction

Predict links in a citation network.

## Feature Engineering

In the feature engineering folder you can find scripts to compute new features from the dataset. The features computed are described at the beginning of the scripts, and you can find more information in our project report.

## Feature Selection

Running the feature_selection.py script will print the results of a forward selection algorithm. We chose the set of features that we were going to use for the rest of the project from these results.

## Models

You can find several implementations of models to fit to our data. Running the scripts will give you the results and create a submission file.

## Tuning

Running the tuning scripts will output best paramaters resulting from a cross validated grid search on a hand picked parameter grid.

## Main

The main.py script processes all you need (feature engineering and machine learning) in order to create our final submission. The svm fit might take a substantial amount of time.
You may use the generated "stack_sub_rf.csv" as a reproduction of our best submission. If they were to be reproducibility issues with runtimes and what not we left our original submission under the name ("stack_sub_rf_reference.csv")

