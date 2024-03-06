from imputation import median_imputation
from outlier_detection import isolation_forest_outliers
from feature_selection import feature_importance_selection
from main import load_data
from sklearn.model_selection import train_test_split

folder = 'data'
X_train_df, y_train_df, X_predict = load_data(
    f'{folder}/X_train.csv', f'{folder}/y_train.csv', f'{folder}/X_test.csv')

imputation_function = median_imputation
outlier_detection_function = isolation_forest_outliers
feature_selection_function = feature_importance_selection

features_train, features_test, target_train, target_test = train_test_split(X_train_df, y_train_df, test_size=0.2)


