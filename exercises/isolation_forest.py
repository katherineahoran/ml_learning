import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.mixture import BayesianGaussianMixture


# Generate data
n_samples, n_outliers = # TODO
rng = # TODO
covariance = # TODO
cluster_1 = # TODO
cluster_2 = # TODO
outliers = # TODO

X = # TODO
y = # TODO

#  Split data into training and test data
X_train, X_test, y_train, y_test = # TODO


# Fit isolation forest based on training set
clf = # TODO
# TODO - fit model on training set

# Predict for training set
predictions = # TODO


# Compare with examples
expectedPercentageOutliers = # TODO

actualPercentageOutliers = # TODO

print("Expected {expectedPercentageOutliers:.2f}%, got {actualPercentageOutliers:.2f}%".format(
    expectedPercentageOutliers=expectedPercentageOutliers, actualPercentageOutliers=actualPercentageOutliers))


