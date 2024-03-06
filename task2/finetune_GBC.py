import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier

import random as rn
from biosppy.signals import ecg
from tqdm import tqdm_notebook as tqdm

# import specialised modules
from sklearnex import patch_sklearn
patch_sklearn()


TRAIN_FILE_PATH = "X_train.csv"
TARGET_FILE_PATH =  "y_train.csv"
TEST_FILE_PATH = "X_test.csv"
seed=42
NUM_MAX_POINTS = 17807
SAMPLING_RATE=300
my_cols = ["id"] + ["x" + str(i) for i in range(NUM_MAX_POINTS)]
np.random.seed(seed)
rn.seed(seed)

# Load train and test set
train_data = pd.read_csv(TRAIN_FILE_PATH, names=my_cols)[1:]
train_data.drop("id", axis=1, inplace=True)

Y_train = pd.read_csv(TARGET_FILE_PATH)
Y_train.drop("id", axis=1, inplace = True)

test_data =  pd.read_csv(TEST_FILE_PATH, names=my_cols)[1:]
id_test = test_data.columns[0]
test_data.drop("id", axis=1, inplace=True)

def make_submission(filename, predictions):
    test_data =  pd.read_csv(TEST_FILE_PATH, names=my_cols)[1:]
    test_data["y"] = predictions
    test_data[["id", "y"]].to_csv(filename, index= False)

def get_features_from_raw_qrs(signal, sampling_rate):
    X = list()
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal, sampling_rate, show=False)
    rpeaks = ecg.correct_rpeaks(signal=signal, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.1)
    
    peaks = signal[rpeaks]
    if len(heart_rate) < 2:
        heart_rate = [0, 1]
    if len(heart_rate_ts) < 2:
        heart_rate_ts = [0, 1]
    
    X.append(np.mean(peaks))
    X.append(np.min(peaks))
    X.append(np.max(peaks))
    X.append(np.mean(np.diff(rpeaks)))
    X.append(np.min(np.diff(rpeaks)))
    X.append(np.max(np.diff(rpeaks)))
    X.append(np.mean(heart_rate))
    X.append(np.min(heart_rate))
    X.append(np.max(heart_rate))
    X.append(np.mean(np.diff(heart_rate)))
    X.append(np.min(np.diff(heart_rate)))
    X.append(np.max(np.diff(heart_rate)))
    X.append(np.mean(np.diff(heart_rate_ts)))
    X.append(np.min(np.diff(heart_rate_ts)))
    X.append(np.min(np.diff(heart_rate_ts)))
    X.append(np.max(np.diff(heart_rate_ts)))
    X.append(np.sum(filtered-signal))
    
    X += list(np.mean(templates, axis=0))
    X += list(np.min(templates, axis=0))
    X += list(np.max(templates, axis=0))
    X = np.array(X)
    
    X[np.isnan(X)] = 0
    return X


# Compute features from raw signal
features = list()
sampling_rate = float(SAMPLING_RATE)
for id in tqdm(range(train_data.shape[0])):
    signal = np.array(pd.to_numeric(train_data.iloc[id].dropna()))
    features.append(get_features_from_raw_qrs(signal, sampling_rate))
    
    
X = np.array(features)
y = np.ravel(np.array(Y_train.values))

features_test = list()
for id in tqdm(range(test_data.shape[0])):
    signal = np.array(pd.to_numeric(test_data.iloc[id].dropna()))
    features_test.append(get_features_from_raw_qrs(signal, sampling_rate))
    
X_test = np.array(features_test)

# Create model and make submission
scaler = StandardScaler() 
scaler.fit(X)
x_train_scaled = scaler.transform(X)
x_test_scaled = scaler.transform(X_test)

param_grid = {
    'n_estimators': [100],
    'learning_rate': [0.1],
    'max_depth': [5],
}

f1_scorer = make_scorer(f1_score, average='micro')

# Create a Gradient Boosting Classifier
gb = GradientBoostingClassifier(random_state=seed)

# Set up the grid search with cross-validation
grid_search = GridSearchCV(estimator=gb, param_grid=param_grid, cv=5, scoring=f1_scorer, n_jobs=-1)

print("Start fitting")
# Fit the grid search
grid_search.fit(x_train_scaled, y)

# Best hyperparameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Best score
print(f"Best Score: {grid_search.best_score_}")

# The best estimator
best_gb = grid_search.best_estimator_

# Make predictions using the best estimator
prediction = best_gb.predict(x_test_scaled)

make_submission("final.csv", prediction)
print("GBC done")