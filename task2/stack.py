# Required libraries
import pandas as pd
import numpy as np
from biosppy.signals import ecg
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
# Load your data
X_train_df = pd.read_csv("X_train.csv", index_col='id')
y_train_df = pd.read_csv("y_train.csv", index_col='id')
X_test_df = pd.read_csv("X_test.csv", index_col='id')
# Your existing feature extraction functions remain the same
from biosppy.signals import ecg
import numpy as np

from sklearnex import patch_sklearn
patch_sklearn()

NUM_MAX_POINTS = 17807
my_cols = ["id"] + ["x" + str(i) for i in range(NUM_MAX_POINTS)]
TEST_FILE_PATH = "X_test.csv"
def make_submission(filename, predictions):
    test_data =  pd.read_csv(TEST_FILE_PATH, names=my_cols)[1:]
    test_data["y"] = predictions
    test_data[["id", "y"]].to_csv(filename, index= False)

def get_features_from_raw_qrs(signal, sampling_rate):
    X = []
    signal = signal.dropna()
    ts, filtered, rpeaks, templates_ts, templates, heart_rate_ts, heart_rate = ecg.ecg(signal, sampling_rate, show=False)
    rpeaks = ecg.correct_rpeaks(signal=signal, rpeaks=rpeaks, sampling_rate=sampling_rate, tol=0.1)
    
    peaks_voltage = filtered[rpeaks]
    if len(heart_rate) < 2:
        heart_rate = [0, 1]
    if len(heart_rate_ts) < 2:
        heart_rate_ts = [0, 1]
    
    #peak voltage
    X.append(np.mean(peaks_voltage))
    X.append(np.min(peaks_voltage))
    X.append(np.max(peaks_voltage))
    X.append(np.std(peaks_voltage))

    #rpeak timings
    X.append(np.mean(np.diff(rpeaks)))
    X.append(np.min(np.diff(rpeaks)))
    X.append(np.max(np.diff(rpeaks)))
    X.append(np.std(np.diff(rpeaks)))

    #heartrate
    X.append(np.mean(heart_rate))
    X.append(np.min(heart_rate))
    X.append(np.max(heart_rate))
    X.append(np.std(heart_rate))

    #heartrate differences
    X.append(np.mean(np.diff(heart_rate)))
    X.append(np.min(np.diff(heart_rate)))
    X.append(np.max(np.diff(heart_rate)))

    #interval  between heartrate measurements
    X.append(np.mean(np.diff(heart_rate_ts)))
    X.append(np.min(np.diff(heart_rate_ts)))
    X.append(np.max(np.diff(heart_rate_ts)))
    X.append(np.std(np.diff(heart_rate_ts)))
    
    #average behaviour of heart
    X += list(np.mean(templates, axis=0))
    # X += list(np.min(templates, axis=0))
    # X += list(np.max(templates, axis=0))

    X = np.array(X)
    
    X[np.isnan(X)] = 0
    return X

def extract_features_from_df(X_df: pd.DataFrame, sampling_rate=300):
    transformed_rows = []

    for _, row in X_df.iterrows():
        transformed_row = get_features_from_raw_qrs(row, sampling_rate)
        transformed_rows.append(transformed_row)

    # Create a new DataFrame from the list of transformed rows
    transformed_df = pd.DataFrame(transformed_rows, index=X_df.index)

    return transformed_df

print(X_train_df)
# Extract features
new_X_train_df = extract_features_from_df(X_train_df)
new_X_test_df = extract_features_from_df(X_test_df)

# Define base classifiers with their best parameters (adjust these parameters as per your requirements)
extra_trees = ExtraTreesClassifier(max_depth=200, max_features=None, n_estimators=800)
gradient_boosting = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, max_features=None, min_samples_leaf=4, min_samples_split= 2, n_estimators=400, subsample=1.0)
svm = SVC(C=10, gamma='scale', kernel='rbf')

# Define the Stacking Classifier
stacking_classifier = StackingClassifier(
    estimators=[
        ('extra_trees', extra_trees),
        ('gradient_boosting', gradient_boosting),
        ('svm', svm)
    ],
    final_estimator=RidgeClassifier()
)

# Define the parameter grid for GridSearchCV (adjust this grid as needed)
param_grid = {
    'final_estimator__alpha': [1],
    'stack_method': ['auto']
}

# Define scorer
f1_scorer = make_scorer(f1_score, average='micro')
# Set up GridSearchCV
grid_search = GridSearchCV(estimator=stacking_classifier, param_grid=param_grid, cv=5, scoring=f1_scorer, n_jobs=-1)
grid_search.fit(new_X_train_df, y_train_df.squeeze())  # Ensure y_train_df is in the correct format

# Print best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)
# 0.809455 

# The best estimator
best_gb = grid_search.best_estimator_

# Make predictions using the best estimator
prediction = best_gb.predict(new_X_test_df)

make_submission("final.csv", prediction)
print("stack done")



