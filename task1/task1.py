import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *

# Load the datasets from CSV
X_train_data = pd.read_csv('X_train.csv')
y_train_data = pd.read_csv('y_train.csv')
X_test_data = pd.read_csv('X_test.csv')

# Merge X_train_data and y_train_data based on the "id" column
train_data = pd.merge(X_train_data, y_train_data, on='id')

# Split into X and y
X_train = train_data.iloc[:, 1:-1].values  # Excluding the 'id' and 'y' columns
y_train = train_data.iloc[:, -1].values

# Split X_test_data into ID and features
test_ids = X_test_data['id'].values
X_test = X_test_data.iloc[:, 1:].values  # Excluding the 'id' column


# 1. Imputation of Missing Values
# imputer = SimpleImputer(strategy='mean')
imputer = KNNImputer(n_neighbors=5)
X_train = imputer.fit_transform(X_train)
# Applying the same imputation strategy on the test set
X_test = imputer.transform(X_test)

# 2. Outlier Detection using Isolation Forest on training data
iso = IsolationForest(contamination=0.05)
outliers = iso.fit_predict(X_train)
# lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
# outliers = lof.fit_predict(X_train)
# Filter out outliers
X_train = X_train[outliers == 1]
y_train = y_train[outliers == 1]

# clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_train)
# outliers = clustering.labels_ == -1
# print(outliers)
# X_train = X_train[~outliers]
# y_train = y_train[~outliers]




# 3. Feature Selection using Feature Importance from RandomForest on training data
clf = RandomForestRegressor()
clf.fit(X_train, y_train)
# Selecting important features based on threshold
sfm = SelectFromModel(clf, threshold=0.01)
X_train = sfm.fit_transform(X_train, y_train)

# Applying the same feature selection on the test set
X_test = sfm.transform(X_test)


# Regression Process using RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# regressor = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
# regressor = DecisionTreeRegressor(max_depth=5)
# regressor = SVR(kernel='rbf', C=1.0, epsilon=0.1)
# regressor = DecisionTreeRegressor(max_depth=5)
# regressor = Ridge(alpha=1.0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# kernel = Matern(nu=1.5, length_scale=19, length_scale_bounds = "fixed")
# rng = np.random.default_rng(seed=0)
# regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, random_state=rng.integers(0, 1000))
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)

# If you want to save the predictions along with their IDs:
output = pd.DataFrame({'id': test_ids, 'y': y_pred})
output.to_csv('predictions.csv', index=False)

print(f"Predictions saved to predictions.csv")
