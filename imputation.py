import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

def median_imputation(data: pd.DataFrame):
    return impute_missing_values(data, strategy='median')

def mean_imputation(data: pd.DataFrame):
    return impute_missing_values(data, strategy='mean')

def impute_missing_values(df: pd.DataFrame, strategy='median'):
    imputer = SimpleImputer(strategy=strategy)
    imputed_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
    return imputed_df

def knn_imputation(data):
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns, index=data.index)
    return imputed_data

def transform_df(df, transformer):
    # Impute missing values using the provided imputer
    imputed_values = transformer.transform(df)
    
    # Create a new DataFrame with imputed values
    imputed_df = pd.DataFrame(imputed_values, columns=df.columns, index=df.index)
    
    return imputed_df