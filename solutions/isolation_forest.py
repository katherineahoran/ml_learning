import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.mixture import BayesianGaussianMixture

# Generate data
n_samples, n_outliers = 120, 40
rng = np.random.RandomState(0)
covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical
outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

X = np.concatenate([cluster_1, cluster_2, outliers])
y = np.concatenate(
    [np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)]
)

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Fit isolation forest based on training set
clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X_train)

# Predict for training set
predictions = clf.predict(X_test)

# Compare with examples
expectedPercentageOutliers = (len(outliers)/len(X)) * 100

actualPercentageOutliers = (((predictions == -1).sum())/len(predictions)) * 100

print("Expected {expectedPercentageOutliers:.2f}%, got {actualPercentageOutliers:.2f}%".format(
    expectedPercentageOutliers=expectedPercentageOutliers, actualPercentageOutliers=actualPercentageOutliers))


