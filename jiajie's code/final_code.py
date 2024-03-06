import warnings

# Ignore both FutureWarning and DeprecationWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.feature_selection import SelectFromModel
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, make_scorer
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV, LassoCV

# ============================ LOGGING ==================================

import sys

class DualWriter:
    def __init__(self, *writers):
        self.writers = writers
        
    def write(self, message):
        for w in self.writers:
            w.write(message)
            
    def flush(self):
        for w in self.writers:
            w.flush()

# Usage
file = open('logfile.log', 'a')
sys.stdout = DualWriter(sys.stdout, file)

# Now all print statements will be logged in 'logfile.log' and printed in the terminal
print("This message goes to stdout and the log file.")

# ============================ LOAD DATA ==================================

def load_data(train_data_path, target_data_path, test_data_path):
    X_train = pd.read_csv(train_data_path, index_col='id')
    y_train = pd.read_csv(target_data_path, index_col='id')
    X_test = pd.read_csv(test_data_path, index_col='id')
    return X_train, y_train, X_test

# ============================ VALUE IMPUTATION ==========================

def median_imputation(data):
    return impute_missing_values(data, strategy='median')

def impute_missing_values(data, strategy='median'):
    imputer = SimpleImputer(strategy=strategy)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
    return imputed_data

# ============================ OUTLIER DETECTION =========================

def local_outlier_factor_outliers_2(data):
    model = LocalOutlierFactor(n_neighbors=50 ,contamination=0.05, n_jobs=-1)
    outliers = model.fit_predict(data)
    return data[outliers == 1]


# ============================ FEATURE SELECTION =========================

def feature_percentile_selection_2(X, y):
    model = SelectPercentile(f_regression, percentile=25)
    X_selected = model.fit_transform(X, y)
    # Get the selected feature indices and create a DataFrame
    selected_features = model.get_support(indices=True)
    X_selected_df = pd.DataFrame(X_selected, columns=X.columns[selected_features], index=X.index)
    return X_selected_df


# ============================== Standardization ============================================

def standardize_data(data):
    scaler = StandardScaler()
    standardized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    return standardized_data, scaler

# =============================== Zero-Variance Removal ======================================

def remove_zero_variance_features(data):
    selector = VarianceThreshold()
    return pd.DataFrame(selector.fit_transform(data), columns=data.columns[selector.get_support()], index=data.index)



def main():

    save_dir = 'model_predictions'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data
    X_train, y_train, X_test = load_data('X_train.csv', 'y_train.csv', 'X_test.csv')

    print(f'The shape of X_train is {X_train.shape}')
    print(f'The shape of y_train is {y_train.shape}')
    print(f'The shape of X_test is {X_test.shape}')

    X_train = remove_zero_variance_features(X_train)
    X_test = X_test[X_train.columns]

    print(f'The shape of X_train is {X_train.shape}')
    print(f'The shape of y_train is {y_train.shape}')
    print(f'The shape of X_test is {X_test.shape}')
    # print((X_train.columns == X_test.columns).sum())

    # Define regression models to evaluate
    regression_models = {
        'ExtraTreesRegressor' : ExtraTreesRegressor(random_state=42, n_jobs=-1),
        'RidgeRegressor': Ridge(random_state=42),
        'RandomForestRegressor': RandomForestRegressor(random_state=42, n_jobs=-1),
        'GradientBoostingRegressor': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
    }

    param_grids = { # The hyperparameter setting that gives the best public score but not sure if this overfits on the public test set
         'GradientBoostingRegressor': {
            'n_estimators': [1500],  # Number of boosting stages to perform
            'learning_rate': [0.01],  # Shrinks the contribution of each tree by learning_rate
            'max_depth': [6],  # Maximum depth of the individual regression estimators
            'min_samples_split': [2],  # Minimum number of samples required to split an internal node
            'min_samples_leaf': [4],  # Minimum number of samples required to be at a leaf node
            'max_features': [None],  # The number of features to consider when looking for the best split
            'subsample': [0.7],  # The fraction of samples to be used for fitting the individual base learners
        },
        'SVR': {
            'kernel': ['rbf'],  # Type of kernel to use
            'C': [70],  # Regularization parameter
            'gamma': ['auto'],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            'epsilon': np.logspace(-6, 0, 7)  # Epsilon in the epsilon-SVR model
        },
        'ExtraTreesRegressor': {
            'n_estimators': [1000,1200],  # Fewer estimators may suffice for small datasets
            'max_features': [None, 'sqrt'],  # Try 'auto' and a smaller subset of features
        },
    }

    # Impute data
    # Impute training data
    X_train_imputed = median_imputation(X_train)
    # Impute test data
    X_test_imputed = median_imputation(X_test)
    
    # Remove outleir data (in training data)
    X_train_outliers_removed = local_outlier_factor_outliers_2(X_train_imputed)
    y_train_outliers_removed = y_train.loc[X_train_outliers_removed.index]

    # Standardize data after outlier removal
    X_train_standardized, scaler = standardize_data(X_train_outliers_removed)
    # Standardize test data using the same scaler
    X_test_standardized = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns, index=X_test_imputed.index)

    # Select features
    X_train_feature_selected = feature_percentile_selection_2(X_train_standardized,y_train_outliers_removed['y'].ravel())
    X_test_feature_selected = X_test_standardized[X_train_feature_selected.columns]

    # 5-fold Cross-validation Grid Search on SVR
    print(f"Cross-validating SVR...")
    grid_search1 = GridSearchCV(regression_models['SVR'], param_grids['SVR'], scoring='r2', cv=5, n_jobs=-1)
    grid_search1.fit(X_train_feature_selected,y_train_outliers_removed['y'].ravel())
    print(f"Best parameters for SVR: {grid_search1.best_params_}")
    print(f"Best CV r2_score for SVR: {grid_search1.best_score_}")

    
    # 5-fold Cross-validation Grid Search on ExtraTrees
    print(f"Cross-validating ExtraTrees...")
    grid_search2 = GridSearchCV(regression_models['ExtraTreesRegressor'], param_grids['ExtraTreesRegressor'], scoring='r2', cv=5, n_jobs=-1)
    grid_search2.fit(X_train_feature_selected,y_train_outliers_removed['y'].ravel())
    print(f"Best parameters for ExtraTrees: {grid_search2.best_params_}")
    print(f"Best CV r2_score for ExtraTrees: {grid_search2.best_score_}")

    # 5-fold Cross-validation Grid Search on Gradient Boosting Rgressor
    print(f"Cross-validating GBR...")
    grid_search3 = GridSearchCV(regression_models['GradientBoostingRegressor'], param_grids['GradientBoostingRegressor'], scoring='r2', cv=5, n_jobs=-1)
    grid_search3.fit(X_train_feature_selected,y_train_outliers_removed['y'].ravel())
    print(f"Best parameters for GBR: {grid_search3.best_params_}")
    print(f"Best CV r2_score for GBR: {grid_search3.best_score_}")

    # Stacking Regressor 1
    sr = StackingRegressor(estimators=[('SVR', grid_search1.best_estimator_), ('ExtraTrees', grid_search2.best_estimator_), ('GradientBoostingRegressor', grid_search3.best_estimator_)], final_estimator=RidgeCV(alphas=[200,300,400,500]), n_jobs=-1)
    sr.fit(X_train_feature_selected, y_train_outliers_removed['y'].ravel())
    
    print(f"RidgeCV final alpha: {sr.final_estimator_.alpha_}")

    # Make predictions
    predictions = sr.predict(X_test_feature_selected)

    # Save the predictions
    prediction_filename = f"SVR_ET_GBR_RidgeCV_predictions.csv"
    prediction_filepath = os.path.join(save_dir, prediction_filename)
    prediction_df = pd.DataFrame(predictions, index=X_test_feature_selected.index, columns=['y'])
    prediction_df.to_csv(prediction_filepath)
    


if __name__ == "__main__":
    main()