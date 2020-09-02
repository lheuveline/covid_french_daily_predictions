# Covid-19 French Daily Predictions

This project is an educational project made to discover p5.js and tensorflow.js environment.

The project contains two main pages :

* /train.html : Train neural network using prepared dataset. Tensorflow-vis allows to follow neural network loss per batch and per epoch.
* /map.html : Get lastest daily propagation data, use trained model to predict for the next and display probabilities on map.

## Notes

* The dataset file (sampled_covid_dataset) has been created with scripts/get_dataset.py. This script get historical data from API (on 120 days) and construct a dataset containing the 14th previous hospitalizations values and the direction of the next step (1 or -1). The dataset needed some resampling because it was largely imbalanced using only 120 days.
* The neural network is equivalent to a simple Logistic Regression using 14 variables, with Gradient Descent Optimization.

## To-do

* Get larger dataset.
* Use LSTM cells / Conv-LSTM architecture to detect long term patterns
* Store propagation data in database
* Use train / test sets for training and store model metrics for optimization
* Display only France map and not all world.