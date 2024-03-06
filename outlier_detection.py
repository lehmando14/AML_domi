'''
iterate over the list OUTLIER_DETECTION_FUNCTIONS to try out different functions
'''

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def dbscan(x_train_df: pd.DataFrame, eps=3, min_samples=5):
  x_train = x_train_df.to_numpy()
  clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(x_train)

  labels_of_points = clustering.labels_
  number_of_clusters = len((set(labels_of_points) -{-1}))
  number_of_outliers = np.count_nonzero(labels_of_points == -1)
  print(f'eps:{eps}, min_samples:{min_samples}; \
        Number of cluster = {number_of_clusters},\
        Number of outliers = {number_of_outliers}')

  return x_train_df[labels_of_points != -1]

def local_outlier_factor(x_train_df: pd.DataFrame, n_neighbors=20):
  x_train = x_train_df.to_numpy()
  clf = LocalOutlierFactor(n_neighbors=n_neighbors)
  
  labels_of_points = clf.fit_predict(x_train)
  number_of_outliers = np.count_nonzero(labels_of_points == -1)
  print(f'n_neighbors:{n_neighbors}; Number of outliers = {number_of_outliers}')
  
  return x_train_df[labels_of_points != -1]

def isolation_forest_outliers(x_train_df: pd.DataFrame, y_train_df: pd.DataFrame, contamination=0.05):
    x_train = x_train_df.to_numpy()
    clf = IsolationForest(contamination=contamination) # contamination is the proportion of outliers in the data

    labels_of_points = clf.fit_predict(x_train)
    number_of_outliers = np.count_nonzero(labels_of_points == -1)
    print(f'contamination:{contamination}; Number of outliers = {number_of_outliers}')

    non_outlier_index = labels_of_points == 1
    return x_train_df[non_outlier_index], y_train_df[non_outlier_index]

def iqr_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    filtered_data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]
    return filtered_data


OUTLIER_DETECTION_FUNCTIONS = [
   dbscan,
   local_outlier_factor,
   isolation_forest_outliers,
]