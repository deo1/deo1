#https://www.kaggle.com/c/titanic/data
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
TEST_PATH = DATA_PATH + "test.csv"
UNKNOWN = '(unknown)'
LOSS = 'Survived'
ID = 'PassengerId'
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
train['set'] = 'train'
test['set'] = 'test'
combined = pd.concat([train, test]).reset_index(drop=True)

# do some preliminary feature engineering
title_regex = '([A-Za-z]+)\.'
surname_regex = '([A-Za-z]+)\,'
deck_regex = '([A-Z]+)'
ticket_regex = '([A-Z\/\.]+)'

get_title = lambda n: re.search(title_regex, n).group(0)[:-1]
get_surname = lambda n: re.search(surname_regex, n).group(0)[:-1]
get_deck = lambda c: re.search(deck_regex, c).group(0) if c == c else UNKNOWN
get_ticket = lambda t: \
    re.search(ticket_regex, t).group(0).replace('/', '').replace('.', '') \
    if re.search(deck_regex, t) != None else UNKNOWN

combined['Title'] = combined['Name'].apply(lambda n: get_title(n))
combined['Surname'] = combined['Name'].apply(lambda n: get_surname(n))
surnames = combined['Surname'].value_counts()
combined['FamilySize'] = combined['Surname'].apply(lambda s: surnames[s])
combined['Deck'] = combined['Cabin'].apply(lambda c: get_deck(c))
combined['TicketLoc'] = combined['Ticket'].apply(lambda t: get_ticket(t))
combined['Fare'] = combined['Fare'].apply(lambda f: np.log(f) if f > 0 else 0)
features = [col for col in combined.columns if col not in IGNORE]
types = combined[features].dtypes

# fill missing data and encode categorical data
# TODO : implement this with an sklearn `pipeline`
for col in features:
    is_categorical = any([types[col].type == cat for cat in CATEGORICAL])
    if is_categorical:
        # fill missing values with a keyword
        combined[col].fillna(UNKNOWN, inplace=True)
        
        # encode as integers
        le = preprocessing.LabelEncoder()
        le.fit(combined[col])
        combined[col] = le.transform(combined[col])
    else:
        # fill missing values with the median
        # TODO : contextual median for age by class / title
        combined[col].fillna(
            combined[col].median(), inplace=True)


# =============================================================================
# Prepare data for train / test
# =============================================================================
X_train = combined[combined.set == 'train'][features].reset_index(drop=True)
y_train = combined[combined.set == 'train'][LOSS].reset_index(drop=True)
X_test = combined[combined.set == 'test'][features].reset_index(drop=True)


# =============================================================================
# Train model(s)
# =============================================================================

# train individual layer 1 models
print('\nTraining layer 1 models...')
config_dict = cccd.classifier_config_dict # xgboost removed for now
ensemble_dict = eccd.classifier_config_dict
preprocessor_dict = pccd.classifier_config_dict
model_features = []
model_layer1 = []
for config in ensemble_dict:
    print('\n')
    this_config = {config: ensemble_dict[config]}
    this_config.update(preprocessor_dict)
    model = tp.TPOTClassifier(
        generations=15,
        population_size=15,
        cv=5,
        verbosity=2,
        n_jobs=N_JOBS,
        config_dict=this_config)
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)
    print(score)
    model_layer1.append(model)
    model_features.append(model.predict(X_train).reshape(-1, 1))

# combine layer 1 predictions as features into the design matrix
new_features = np.hstack(model_features)
new_features = pd.DataFrame(
    new_features,
    columns=['model_' + str(ii) for ii in range(new_features.shape[1])])
for col in new_features.columns:
    X_train[col] = new_features[col]


# train the stacked model on all classifiers
print('\nTraining stacked XGBoost model...')
baseline_params = h.get_params(algorithm='xgb', ptype='start', ver=2)
alg = xgb.XGBClassifier( **baseline_params )
#alg = xgb.XGBRegressor( **baseline_params )

# tune the model
xgb_model, importance = \
    h.fit_model(
        alg,
        X_train,
        y_train,
        useTrainCV=True,
        folds=5,
        metrics=['map','error'],
        chatty=2,
        show_report=True)

# =============================================================================
# Predict survival on test set
# =============================================================================
# predict layer 1 test features
print('\nPredicting layer 1 on test set...')
model_features = []
for model in model_layer1: 
    model_features.append(model.predict(X_test).reshape(-1, 1))\

# combine layer 1 predictions as features into the design matrix
new_features = np.hstack(model_features)
new_features = pd.DataFrame(
    new_features,
    columns=['model_' + str(ii) for ii in range(new_features.shape[1])])
for col in new_features.columns:
    X_test[col] = new_features[col]

# predict y_test using the stacked model
print('\nPredicting final loss on test set...')
if hasattr(xgb_model, 'best_ntree_limit'):
    y_test = xgb_model.predict(X_test, ntree_limit=xgb_model.best_ntree_limit)
else:
    y_test = xgb_model.predict(X_test)

# =============================================================================
# Output model and prediction in submission format
# =============================================================================
passenger_id = combined[combined['set'] == 'test'][ID].values
submission = pd.DataFrame({ID: passenger_id, LOSS: y_test})
submission[LOSS] = submission[LOSS].apply(lambda y: int(y))
tag = "{}-{:0.4}".format(START_TIME.strftime("%Y_%m_%d_%H_%M_%S"), score)
submission_path = DATA_PATH + "submission_{}.csv".format(tag)
model_path = MODELS_PATH + "model_{}.py".format(tag)

# save outputs
submission.to_csv(submission_path, sep=',', index=False)
model.export(model_path)
