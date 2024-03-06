import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from itertools import product
from scipy.stats import zscore
from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


# ============================ LOAD DATA ==================================

def load_data(train_data_path, target_data_path, test_data_path):
    X_train = pd.read_csv(train_data_path, index_col='id')
    X_train.index = X_train.index.astype(int)
    y_train = pd.read_csv(target_data_path, index_col='id')
    y_train.index = y_train.index.astype(int)
    X_test = pd.read_csv(test_data_path, index_col='id')
    X_test.index = X_test.index.astype(int)
    return X_train, y_train, X_test

# ============================ REGRESSION MODEL ==========================

def random_forest_regressor(X, y):
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X, y)
    return regressor

def gradient_boosting_regressor(X, y):
    regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
    regressor.fit(X, y)
    return regressor

def linear_regression(X, y):
    regressor = LinearRegression()
    regressor.fit(X, y)
    return regressor

# Add more regression models as needed...

# ============================ MAKE PREDICTIONS ==========================

def make_predictions(regressor, X):
    return regressor.predict(X)

# ============================ SAVE PREDICTIONS ==========================

def save_predictions(ids, y_pred, output_path='submission.csv'):
    submission = pd.DataFrame({
        'id': ids,
        'y': y_pred
    })
    submission.to_csv(output_path, index=False)

# ============================ K-FOLD CROSS-VALIDATION ====================

def k_fold_cross_validation(X, y, outlier_method, feature_method, imputation_method, regression_method):
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    scores = []

    for train_index, val_index in kf.split(X):
        # print(train_index)
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        # Imputation before outlier detection and feature selection
        X_train_imputed = imputation_method(X_train_fold)
        X_val_imputed = imputation_method(X_val_fold)
        

        # Outlier Detection
        X_train_filtered = outlier_method(X_train_imputed)
        # Filter y_train_fold using the indices from the outlier detection method
        y_train_fold_filtered = y_train_fold.loc[X_train_filtered.index]

        X_train_selected = feature_method(X_train_filtered, y_train_fold_filtered['y'])
        X_val_selected = X_val_imputed[X_train_selected.columns]

        regressor = regression_method(X_train_selected, y_train_fold_filtered['y'])
        y_pred = make_predictions(regressor, X_val_selected)

        score = r2_score(y_val_fold['y'], y_pred)
        scores.append(score)

    return sum(scores) / len(scores)

# ============================ MAIN ======================================
def main():

    outlier_methods = {
        # 'iqr': iqr_outliers,
        'isolation_forest': isolation_forest_outliers,
        # ...
    }
 
    feature_selection_methods = {
        # 'lasso': lasso_features,
        'feature_importance': feature_importance_selection,
        # ...
    }

    imputation_methods = {
        'median': median_imputation,
        'mean': mean_imputation,
        'knn': knn_imputation,
        # ...
    }

    regression_models = {
        'random_forest': random_forest_regressor,
        'gradient_boosting': gradient_boosting_regressor,
        'linear_regression': linear_regression,
        # ...
    }

    # Load data
    X_train, y_train, X_test = load_data('X_train.csv', 'y_train.csv', 'X_test.csv')

    # print("Missing values in X_train:", X_train.isnull().sum().sum())
    # print("Missing values in y_train:", y_train.isnull().sum().sum())
    # print("Missing values in X_test:", X_test.isnull().sum().sum())

    # Separate out 'id' columns before any operations.
    train_ids = X_train['id'].copy()
    test_ids = X_test['id'].copy()
    X_train.drop('id', axis=1, inplace=True)
    X_test.drop('id', axis=1, inplace=True)
    
    best_score = float('-inf')
    best_combination = {}

    combinations = list(product(outlier_methods.items(), feature_selection_methods.items(),
                                imputation_methods.items(), regression_models.items()))

    for (outlier_name, outlier_method), (feature_name, feature_method), (imputation_name, imputation_method), (model_name, model_method) in combinations:
        print(f"Evaluating combination: {outlier_name} - {feature_name} - {imputation_name} - {model_name}")
        
        avg_score = k_fold_cross_validation(X_train, y_train, outlier_method, feature_method, imputation_method, model_method)

        if avg_score > best_score:
            best_score = avg_score
            best_combination = {
                'outlier_method': outlier_name,
                'feature_method': feature_name,
                'imputation_method': imputation_name,
                'model_method': model_name
            }
        print(f"Average R^2 score for this combination: {avg_score}")

    print(f"Best combination: {best_combination} with R^2 score of {best_score}")

    # After identifying best combination, preprocess the entire dataset using these best methods and train the model.
    # Imputation
    X_train = imputation_methods[best_combination['imputation_method']](X_train)
    X_test = imputation_methods[best_combination['imputation_method']](X_test)

    # Outlier Detection
    X_train = outlier_methods[best_combination['outlier_method']](X_train)
    y_train = y_train.loc[X_train.index]
    

    # Feature Selection
    X_train = feature_selection_methods[best_combination['feature_method']](X_train, y_train['y'])
    X_test = X_test[X_train.columns]

    # Model Training
    regressor = regression_models[best_combination['model_method']](X_train, y_train['y'])
    y_pred = make_predictions(regressor, X_test)

    # Save the predictions using saved 'test_ids'
    save_predictions(test_ids, y_pred)

if __name__ == "__main__":
    main()



