import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression

def feature_importance_selection(X_train_df: pd.DataFrame, y_train_df: pd.DataFrame, lowest_quantile_to_drop: float=0.1):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train_df, y_train_df)

    feature_importances = model.feature_importances_
    cut_off_importance = np.quantile(feature_importances, lowest_quantile_to_drop)
    important_features = X_train_df.columns[feature_importances > cut_off_importance]

    def feature_selector(X_test_df: pd.DataFrame):
        return X_test_df[important_features]
    
    return X_train_df[important_features], feature_selector

def feature_importance_selection_keep_n(X_train_df: pd.DataFrame, y_train_df: pd.DataFrame, n: int=10):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train_df, y_train_df)

    feature_importances = model.feature_importances_
    cut_off_importance = np.sort(feature_importances)[-n -1]
    important_features = X_train_df.columns[feature_importances > cut_off_importance]

    def feature_selector(X_test_df: pd.DataFrame):
        return X_test_df[important_features]
    
    return X_train_df[important_features], feature_selector

def k_best_f_regression(X_train_df: pd.DataFrame, y_train_df: pd.DataFrame, k: int=10):
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X_train_df, y_train_df)

    scores = selector.scores_
    cut_off_importance = np.sort(scores)[-k-1]
    k_best_features = X_train_df.columns[scores > cut_off_importance]

    def feature_selector(X_test_df: pd.DataFrame):
        return X_test_df[k_best_features]

    return X_train_df[k_best_features], feature_selector

def feature_selection_mutual_info_percentile(X_train_df: pd.DataFrame, y_train_df: pd.DataFrame, percentile=10):
    feature_selector = SelectPercentile(mutual_info_regression, percentile=10)
    feature_selector.fit(X_train_df, y_train_df)

    indices = feature_selector.get_support(indices=True)
    important_features = X_train_df.columns[indices]

    def feature_selector(X_test_df: pd.DataFrame):
        return X_test_df[important_features]
    
    return X_train_df[important_features], feature_selector
