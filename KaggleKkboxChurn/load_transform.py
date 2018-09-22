# https://www.kaggle.com/c/kkbox-churn-prediction-challenge
import pandas as pd
import tpot as tp
import custom_classifier_config_dict as cccd
import ensemble_classifier_config_dict as eccd
import preprocessor_classifier_config_dict as pccd
from sklearn import preprocessing
import numpy as np
from datetime import datetime
import multiprocessing
import re
import xgbhelpers as h
import xgboost as xgb

# =============================================================================
# Constants
# =============================================================================
DATA_PATH = "./data/"
MODELS_PATH = "./models/"
TRAIN_PATH = DATA_PATH + "train.csv"
TEST_PATH = DATA_PATH + "sample_submission_zero.csv"
MEMBERS_PATH = DATA_PATH + "members.csv"
TRANSACTIONS_PATH = DATA_PATH + "transactions.csv"
LOGS_PATH = DATA_PATH + "user_logs.csv"
LOGS_CHUNK_PATH = DATA_PATH + "/user_logs/"
CHUNK_SIZE = 10**6
UNKNOWN = '(unknown)'
LOSS = 'Survived'
ID = 'msno'
CATEGORICAL = [np.object_]
START_TIME = datetime.utcnow()
N_JOBS = round(0.75 * multiprocessing.cpu_count())
N_JOBS = 1 # multiprocessing is broken

# loss, meta, and high cardinality columns
IGNORE = ['set', ID, LOSS, 'Name', 'Ticket']

# =============================================================================
# Load and process data
# =============================================================================
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
members = pd.read_csv(MEMBERS_PATH)
transactions = pd.read_csv(TRANSACTIONS_PATH)

train = train.merge(transactions, how='left', on=ID)
test = test.merge(members, how='left', on=ID)

logs_train = []
logs_test = []
print('Loading user logs and keeping just train/test users...')
for ii, chunk in enumerate(pd.read_csv(LOGS_PATH, chunksize=CHUNK_SIZE)):
    logs_train_chunk = chunk.merge(train, how='inner', on=ID)
    logs_test_chunk = chunk.merge(test, how='inner', on=ID)
    logs_train.append(logs_train_chunk)
    logs_test.append(logs_test_chunk)
    
    print('\nChunks completed: {}'.format(ii))
    print('User rows kept: {} out of {}'.format(
        len(logs_train_chunk) + len(logs_test_chunk),
        len(chunk)))

    
#logs = pd.read_csv(LOGS_PATH)
#train['set'] = 'train'
#test['set'] = 'test'
#combined = pd.concat([train, test]).reset_index(drop=True)

