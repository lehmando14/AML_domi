import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor, NearestCentroid
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from itertools import product
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.feature_selection import SelectPercentile, f_regression, r_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, ExtraTreesRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import catboost

# ============================ LOAD DATA ==================================

def load_data(train_data_path, target_data_path, test_data_path):
    X_train = pd.read_csv(train_data_path, index_col='id')
    y_train = pd.read_csv(target_data_path, index_col='id')
    X_test = pd.read_csv(test_data_path, index_col='id')
    return X_train, y_train, X_test

# ============================ EXTRA FEATURE PREPROCESSING ======================

def remove_zero_variance_features(data):
    selector = VarianceThreshold()
    return pd.DataFrame(selector.fit_transform(data), columns=data.columns[selector.get_support()], index=data.index)

def normalize_data(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)

# ============================ VALUE IMPUTATION ==========================

def median_imputation(data):
    return impute_missing_values(data, strategy='median')

def mean_imputation(data):
    return impute_missing_values(data, strategy='mean')

def impute_missing_values(data, strategy='median'):
    imputer = SimpleImputer(strategy=strategy)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
    return imputed_data

def knn_imputation(data):
    imputer = KNNImputer(n_neighbors=10)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
    return imputed_data

def mice_imputation(data):
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
    return imputed_data

# ============================ OUTLIER DETECTION =========================

def isolation_forest_outliers(data):
    model = IsolationForest(contamination=0.05, random_state=42) # contamination is the proportion of outliers in the data
    outliers = model.fit_predict(data)
    return data[outliers == 1]

def local_outlier_factor_outliers(data):
    model = LocalOutlierFactor(n_neighbors=500 ,contamination=0.05)
    outliers = model.fit_predict(data)
    return data[outliers == 1]

# def dbscan_outliers(data, eps=0.5, min_samples=10):
#     dbscan = DBSCAN(eps=eps, min_samples=min_samples)
#     clusters = dbscan.fit_predict(data)
#     return data[clusters != -1]

# ============================ FEATURE SELECTION =========================

def feature_importance_selection(X, y, estimator=None, threshold='mean'):
    """
    Select features based on importance weights using SelectFromModel.
    
    :param X: DataFrame, feature matrix.
    :param y: Series or array, target values.
    :param estimator: An estimator object with a feature_importances_ attribute. 
                      Default is RandomForestRegressor if None is provided.
    :param threshold: The threshold value to use for feature selection.
                      Options: 'mean', 'median', a floating point value, or a callable.
    :return: DataFrame with selected features.
    """
    if estimator is None:
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
    
    selector = SelectFromModel(estimator, threshold=threshold)
    selector.fit(X, y)

    # Transform the dataset to keep only selected features
    X_selected = selector.transform(X)

    # Get the selected feature indices and create a DataFrame
    selected_features = selector.get_support(indices=True)
    X_selected_df = pd.DataFrame(X_selected, columns=X.columns[selected_features], index=X.index)

    return X_selected_df

def feature_percentile_selection(X, y):
    model = SelectPercentile(f_regression, percentile=70)
    X_selected = model.fit_transform(X, y)
    # Get the selected feature indices and create a DataFrame
    selected_features = model.get_support(indices=True)
    X_selected_df = pd.DataFrame(X_selected, columns=X.columns[selected_features], index=X.index)
    return X_selected_df


# ============================ REGRESSION MODEL ==========================

def random_forest_regressor(X, y, params):
    regressor = RandomForestRegressor(**params, random_state=42)
    regressor.fit(X, y)
    return regressor


def gradient_boosting_regressor(X, y, params):
    regressor = GradientBoostingRegressor(**params, random_state=42)
    regressor.fit(X, y)
    return regressor

def xgboost_regressor(X, y, params):
    regressor = xgb.XGBRegressor(**params, random_state=42)
    regressor.fit(X, y)
    return regressor

def lightgbm_regressor(X, y, params):
    regressor = lgb.LGBMRegressor(**params, random_state=42)
    regressor.fit(X, y)
    return regressor

def svm_regression(X, y, params):
    regressor = SVR(**params)
    regressor.fit(X, y)
    return regressor

def mlp_regressor(X, y, params):
    regressor = MLPRegressor(**params, random_state=42)
    regressor.fit(X, y)
    return regressor

def adaboost_regressor(X, y, params):
    regressor = AdaBoostRegressor(**params, random_state=42)
    regressor.fit(X, y)
    return regressor

def knn_regressor(X, y, params):
    regressor = KNeighborsRegressor(**params)
    regressor.fit(X, y)
    return regressor

def extra_trees_regressor(X, y, params):
    regressor = ExtraTreesRegressor(**params, random_state=42)
    regressor.fit(X, y)
    return regressor

def nearest_centroid_regressor(X, y, params):
    regressor = NearestCentroid(**params)
    regressor.fit(X, y)
    return regressor

def catboost_regressor(X, y, params):
    regressor = catboost.CatBoostRegressor(**params, random_seed=42)
    regressor.fit(X, y)
    return regressor


# def gp_regression(X, y):
#     kernel = Matern() * RBF()
#     regressor = GaussianProcessRegressor(kernel=kernel,random_state=42)
#     regressor.fit(X, y)
#     return regressor


# def decision_tree_regressor(X, y):
#     regressor = DecisionTreeRegressor(random_state=42)
#     regressor.fit(X, y)
#     return regressor

# def ridge_regression(X, y):
#     regressor = Ridge(alpha=1.0)
#     regressor.fit(X, y)
#     return regressor

# def lasso_regression(X, y):
#     regressor = Lasso(alpha=1.0)
#     regressor.fit(X, y)
#     return regressor



# ============================ MAKE PREDICTIONS ==========================

def make_predictions(regressor, X):
    return regressor.predict(X)

# ============================ SAVE PREDICTIONS ==========================

def save_predictions(ids, y_pred, combination, folder='output', file_name='submission'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    combination_str = "_".join(f"{value}" for value in combination.values())
    file_path = os.path.join(folder, f"{file_name}_{combination_str}.csv")

    submission = pd.DataFrame({
        'id': ids,
        'y': y_pred
    })
    submission.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")


# ============================ K-FOLD CROSS-VALIDATION ====================

def k_fold_cross_validation(X, y, outlier_method, feature_method, imputation_method, regression_method, hyperparams):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = []

    for train_index, val_index in kf.split(X):
        

        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Imputation before outlier detection and feature selection
        X_train_imputed = imputation_method(X_train_fold)
        X_val_imputed = imputation_method(X_val_fold)

        # Removal of zero-variance features
        X_train_no_zero_var = remove_zero_variance_features(X_train_imputed)
        X_val_no_zero_var = X_val_imputed[X_train_no_zero_var.columns]

        # Standardization
        X_train_normalized = normalize_data(X_train_no_zero_var)
        X_val_normalized = pd.DataFrame(StandardScaler().fit(X_train_no_zero_var).transform(X_val_no_zero_var), columns=X_val_no_zero_var.columns, index=X_val_no_zero_var.index)

        # Outlier Detection
        X_train_filtered = outlier_method(X_train_normalized)
        y_train_filtered = y_train_fold.loc[X_train_filtered.index]

        # Feature Selection
        X_train_selected = feature_method(X_train_filtered, y_train_filtered['y'].ravel())
        X_val_selected = X_val_normalized[X_train_selected.columns]

        regressor = regression_method(X_train_selected, y_train_filtered['y'].ravel(), hyperparams)
        y_pred = make_predictions(regressor, X_val_selected)

        score = r2_score(y_val_fold['y'].ravel(), y_pred)
        print("cur score", score)
        scores.append(score)

    return sum(scores) / len(scores)

def main():

    outlier_methods = {
        'isolation_forest': isolation_forest_outliers,
        'lof': local_outlier_factor_outliers,
        # 'dbscan': dbscan_outliers,
        # ...
    }
 
    feature_selection_methods = {
        'rf_feature_importance': lambda X, y: feature_importance_selection(X, y, estimator=RandomForestRegressor(random_state=42), threshold='mean'),
        'et_feature_importance': lambda X, y: feature_importance_selection(X, y, estimator=ExtraTreesRegressor(random_state=42), threshold='mean'),
        # 'en_feature_importance': lambda X, y: feature_importance_selection(X, y, estimator=ElasticNet(random_state=42), threshold='mean'),
        'feature_percentile': feature_percentile_selection,
        # ...
    }

    imputation_methods = {
        'knn': knn_imputation,
        'median': median_imputation,
        'mean': mean_imputation,
        'mice': mice_imputation, #performing poorly
        # 'datawig': datawig_imputation, # datawig handles whole columns with missing values, not individual entries
        # ...
    }

    regression_models = {
        'catboost': catboost_regressor,
        # 'nearest_centroid': nearest_centroid_regressor, #performing poorly
        'extra_trees': extra_trees_regressor,
        'adaboost': adaboost_regressor,
        'knn': knn_regressor,
        'random_forest': random_forest_regressor,
        'gradient_boosting': gradient_boosting_regressor,
        'xgboost': xgboost_regressor,
        'lightgbm': lightgbm_regressor,
        'mlp': mlp_regressor,
        'svm': svm_regression,
        # 'rigde_regression': ridge_regression,
        # 'lasso_regression': lasso_regression,
        # 'decision_tree': decision_tree_regressor,
        # 'gp': gp_regression,
        # ...
    }

    # Hyperparameters for the models
    hyperparameters = {
        'random_forest': [
            {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2},
            {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 4}
        ],
        'gradient_boosting': [
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5}
        ],
        'xgboost': [
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'subsample': 0.8},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'subsample': 0.8},
            {'n_estimators': 500, 'learning_rate': 0.03, 'max_depth': 10, 'subsample': 0.9},
            {'n_estimators': 800, 'learning_rate': 0.01, 'max_depth': 15, 'subsample': 0.9}
        ],
        'lightgbm': [
            {'num_leaves': 31, 'learning_rate': 0.1, 'n_estimators': 100},
            {'num_leaves': 50, 'learning_rate': 0.05, 'n_estimators': 200},
            {'num_leaves': 100, 'learning_rate': 0.03, 'n_estimators': 500},
            {'num_leaves': 150, 'learning_rate': 0.01, 'n_estimators': 800}
        ],
        'svm': [
            {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
            {'C': 1.0, 'kernel': 'rbf', 'gamma': 'auto'},
            {'C': 0.5, 'kernel': 'rbf', 'gamma': 'auto'}
        ],
        'mlp': [
            {'hidden_layer_sizes': (100,), 'alpha': 0.0001},
            {'hidden_layer_sizes': (100,50), 'alpha': 0.0002, 'max_iter': 300},
            {'hidden_layer_sizes': (200,100,50), 'alpha': 0.0005, 'max_iter': 500}
        ],
        'knn': [
            {'n_neighbors': 3, 'weights': 'uniform', 'metric': 'euclidean'},
            {'n_neighbors': 3, 'weights': 'distance', 'metric': 'euclidean'},
            {'n_neighbors': 5, 'weights': 'uniform', 'metric': 'manhattan'},
            {'n_neighbors': 5, 'weights': 'distance', 'metric': 'manhattan'},
            {'n_neighbors': 7, 'weights': 'uniform', 'metric': 'minkowski'},
            {'n_neighbors': 7, 'weights': 'distance', 'metric': 'minkowski'}
        ],
        'adaboost': [
            {'n_estimators': 50, 'learning_rate': 1.0, 'loss': 'linear'},
            {'n_estimators': 100, 'learning_rate': 0.5, 'loss': 'square'},
            {'n_estimators': 150, 'learning_rate': 0.8, 'loss': 'exponential'}
        ],
        'extra_trees': [
            {'n_estimators': 100, 'max_features': 'sqrt', 'max_depth': None},
            {'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 10},
            {'n_estimators': 300, 'max_features': 'log2', 'max_depth': 20}
        ],
        'nearest_centroid': [
            {'metric': 'euclidean', 'shrink_threshold': None},
            {'metric': 'manhattan', 'shrink_threshold': 0.5}
        ],
        'catboost': [
            {'iterations': 500, 'learning_rate': 0.05, 'depth': 6},
            {'iterations': 1000, 'learning_rate': 0.01, 'depth': 8, 'l2_leaf_reg': 3, 'border_count': 32},
            {'iterations': 1500, 'learning_rate': 0.1, 'depth': 10, 'l2_leaf_reg': 5, 'border_count': 20}
        ]
    }

    # Load data
    X_train, y_train, X_test = load_data('X_train.csv', 'y_train.csv', 'X_test.csv')

    # print("Missing values in X_train:", X_train.isnull().sum().sum())
    # print("Missing values in y_train:", y_train.isnull().sum().sum())
    # print("Missing values in X_test:", X_test.isnull().sum().sum())

    best_score = float('-inf')
    best_combination = {}

    combinations = list(product(outlier_methods.items(), feature_selection_methods.items(),
                                imputation_methods.items(), regression_models.items()))
    
    for (outlier_name, outlier_method), (feature_name, feature_method), (imputation_name, imputation_method), (model_name, model_method) in combinations:
        for params in hyperparameters[model_name]:
            print(f"Evaluating combination: {outlier_name} - {feature_name} - {imputation_name} - {model_name} with hyperparameters {params}")

            avg_score = k_fold_cross_validation(X_train, y_train, outlier_method, feature_method, imputation_method, model_method, params)

            if avg_score > best_score:
                best_score = avg_score
                best_combination = {
                    'outlier_method': outlier_name,
                    'feature_method': feature_name,
                    'imputation_method': imputation_name,
                    'model_method': model_name,
                    'hyperparameter_set': params
                }

            print(f"Average R^2 score for this combination: {avg_score}")

    print(f"Best combination: {best_combination} with R^2 score of {best_score}")

    # After identifying best combination, preprocess the entire dataset using these best methods and train the model.
    # Imputation
    X_train = imputation_methods[best_combination['imputation_method']](X_train)
    X_test = imputation_methods[best_combination['imputation_method']](X_test)

    # Removal of zero-variance features
    X_train = remove_zero_variance_features(X_train)
    X_test = X_test[X_train.columns]

    # Standardization
    X_train = normalize_data(X_train)
    X_test = pd.DataFrame(StandardScaler().fit(X_train).transform(X_test), columns=X_test.columns, index=X_test.index)

    # Outlier Detection
    X_train = outlier_methods[best_combination['outlier_method']](X_train)
    y_train = y_train.loc[X_train.index]
    

    # Feature Selection
    X_train = feature_selection_methods[best_combination['feature_method']](X_train, y_train['y'].ravel())
    X_test = X_test[X_train.columns]

    # Model Training
    regressor = regression_models[best_combination['model_method']](X_train, y_train['y'].ravel(), best_combination['hyperparameter_set'])
    y_pred = make_predictions(regressor, X_test)

    # Save Predictions
    save_predictions(X_test.index, y_pred, best_combination, folder='output', file_name='predictions')


def test():
    # Load Data
    X_train, y_train, X_test = load_data('../data/X_train.csv', '../data/y_train.csv', '../data/X_test.csv')
    # print(X_train.head())
    # print(y_train.head())
    # print(X_test.head())
    print(X_train.shape)


    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)

    selector = VarianceThreshold()
    X_train = pd.DataFrame(selector.fit_transform(X_train), columns=X_train.columns[selector.get_support()])

    
    # Imputation
    X_train_imputed = median_imputation(X_train)
    # print(X_train_imputed.head())
    # X_test_imputed = median_imputation(X_test)

    X_train_filtered = isolation_forest_outliers(X_train_imputed)
    y_train_filtered = y_train.loc[X_train_filtered.index] # Dataframe with shape (n_samples, 1)
    # print(X_train_filtered.head())
    # print(y_train_filtered.head())

    # Feature Selection

    # f = lambda X, y: feature_importance_selection(X, y, estimator=ExtraTreesRegressor(random_state=42), threshold='mean')
    # X_train_selected = f(X_train_filtered, y_train_filtered['y'].ravel())
    X_train_selected = feature_percentile_selection(X_train_filtered, y_train_filtered['y'].ravel())
    # print(X_train_selected.shape)
    # X_test_selected = X_test_imputed[X_train_selected.columns]
    # print(X_test_selected.head())

    # Regression
    # print(linear_regression(X_train_selected, y_train_filtered['y'].ravel()).predict(X_test_selected))

if __name__ == '__main__':
    main()
    # test()