import pandas as pd
import tpot as tp
import custom_classifier_config_dict as cccd
from sklearn import preprocessing
import numpy as np
from datetime import datetime
import multiprocessing

# =============================================================================
# Constants
# =============================================================================
DATA_PATH = "./data/"
MODELS_PATH = "./models/"
TRAIN_PATH = DATA_PATH + "train.csv"
TEST_PATH = DATA_PATH + "test.csv"
LOSS = 'Survived'
CATEGORICAL = [np.object_]
START_TIME = datetime.utcnow()
N_JOBS = multiprocessing.cpu_count() // 2 + 1

# loss, meta, and high cardinality columns
IGNORE = ['set', 'PassengerId', LOSS, 'Name', 'Ticket']


# =============================================================================
# Load and process data
# =============================================================================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
train['set'] = 'train'
test['set'] = 'test'
combined = pd.concat([train, test])
features = [col for col in combined.columns if col not in IGNORE]
types = combined[features].dtypes

# fill missing data and encode categorical data
# TODO : implement this with an sklearn `pipeline`
for col in features:
    is_categorical = any([types[col].type == cat for cat in CATEGORICAL])
    if is_categorical:
        # fill missing values with a keyword
        combined[col].fillna('(unknown)', inplace=True)
        
        # encode as integers
        le = preprocessing.LabelEncoder()
        le.fit(combined[col])
        combined[col] = le.transform(combined[col])
    else:
        # fill missing values with the 1st mode
        combined[col].fillna(
            combined[col].value_counts().index[0], inplace=True)


# =============================================================================
# Prepare data for train / test
# =============================================================================
X_train = combined[combined.set == 'train'][features]
y_train = combined[combined.set == 'train'][LOSS]
X_test = combined[combined.set == 'test'][features]
y_test = combined[combined.set == 'test'][LOSS]


# =============================================================================
# Train model(s)
# =============================================================================
config_dict = cccd.classifier_config_dict # xgboost removed for now
model = tp.TPOTClassifier(
    generations=100,
    population_size=100,
    cv=5,
    verbosity=2,
    n_jobs=N_JOBS,
    config_dict=config_dict)
model.fit(X_train, y_train)
score = model.score(X_train, y_train)


# =============================================================================
# Predict survival on test set
# =============================================================================
y_test = model.predict(X_test)


# =============================================================================
# Output model and prediction in submission format
# =============================================================================
passenger_id = combined[combined['set'] == 'test']['PassengerId'].values
submission = pd.DataFrame({'PassengerId': passenger_id, LOSS: y_test})
tag = "{}-{:0.4}".format(START_TIME.strftime("%Y_%m_%d_%H_%M_%S"), score)
submission_path = DATA_PATH + "submission_{}.csv".format(tag)
model_path = MODELS_PATH + "model_{}.py".format(tag)

# save outputs
submission.to_csv(submission_path, sep=',', index=False)
model.export(model_path)
