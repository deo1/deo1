# reference: https://www.kaggle.com/c/nyc-taxi-trip-duration
# reference: http://www.faqs.org/faqs/ai-faq/neural-nets/part1/preamble.html
# reference: https://www.kaggle.com/mathijs/weather-data-in-new-york-city-2016
# reference: https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm/data
# feature analysis: https://www.kaggle.com/headsortails/nyc-taxi-eda-update-the-fast-the-curious
import os
from taxinet import TaxiNet, TaxiCombinerNet
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from math import sin, cos, sqrt, atan2, radians
from torch.autograd import Variable
from random import random
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import xgbhelpers as h
import xgboost as xgb

# TODO args feature
RUN_FEATURE_EXTRACTION = False
MAX_DISTANCE = 100 * 10**3  # 100 km
MAX_DURATION = 12 * 60 * 60 # 12 hours
ENSEMBLE_COUNT = 2


# ===============================
# Date extraction
# ===============================
    
if (RUN_FEATURE_EXTRACTION):
    
    # read data
    test = pd.read_csv('./data/test.csv')
    train = pd.read_csv('./data/train.csv')
    
    # label data
    test['set'] = 'test'
    train['set'] = 'train'
    
    # instantiate the loss column in the test set so that schemas match
    test['trip_duration'] = np.NaN
    
    # union `join='outer'` the train and test data so that encoding can be done holistically
    # and reset the index to be monotically increasing
    combined = pd.concat([test, train], join='outer')
    combined.set_index([list(range(0, combined.shape[0]))], inplace=True)
    
    
    # ===============================
    # Feature engineering and cleanup
    # ===============================
    
    # drop unneeded column(s)
    # store_and_fwd_flag looks meaningless and dropoff_datetime is only available
    # in the train, and is redundant with trip_duration - pickup_datetime
    combined.drop(['store_and_fwd_flag', 'dropoff_datetime'], axis=1, inplace=True)
    
    # segment datetime into year / month / day / hour columns
    # this should help training differentiate between e.g. weekends vs. weekdays
    # first encode string as datetime object
    combined['pickup_datetime'] = combined['pickup_datetime'].apply(
        lambda dt: datetime.strptime(dt, '%Y-%m-%d %H:%M:%S'))
    
    # then encode the important components as integers (ignoring seconds to keep the training simpler)
    combined['year'] = combined['pickup_datetime'].apply(lambda dt: dt.year)
    combined['month'] = combined['pickup_datetime'].apply(lambda dt: dt.month)
    combined['week'] = combined['pickup_datetime'].apply(lambda dt: dt.week)
    combined['weekday'] = combined['pickup_datetime'].apply(lambda dt: dt.weekday())
    combined['hour'] = combined['pickup_datetime'].apply(lambda dt: dt.hour)
    combined['minute'] = combined['pickup_datetime'].apply(lambda dt: dt.minute)
    
    # finally drop the datetime object which won't be useful for training
    combined.drop('pickup_datetime', axis=1, inplace=True)
    
    # round latitude/longitude to a practical precision. 5 decimal places == 1 meter
    # (should not affect loss estimate but should improve training)
    combined['dropoff_latitude'] = combined['dropoff_latitude'].apply(lambda loc: round(loc, 5))
    combined['dropoff_longitude'] = combined['dropoff_longitude'].apply(lambda loc: round(loc, 5))
    combined['pickup_latitude'] = combined['pickup_latitude'].apply(lambda loc: round(loc, 5))
    combined['pickup_longitude'] = combined['pickup_longitude'].apply(lambda loc: round(loc, 5))

    # 1 km clusters
    combined['dropoff_latitude_cluster'] = combined['dropoff_latitude'].apply(lambda loc: round(loc, 2))
    combined['dropoff_longitude_cluster'] = combined['dropoff_longitude'].apply(lambda loc: round(loc, 2))
    combined['pickup_latitude_cluster'] = combined['pickup_latitude'].apply(lambda loc: round(loc, 2))
    combined['pickup_longitude_cluster'] = combined['pickup_longitude'].apply(lambda loc: round(loc, 2))

    # compute an "as the crow flies" curvature distance to nearest 10th of a meter
    # https://en.wikipedia.org/wiki/Haversine_formula
    R = 6373000 # approximate radius of earth in meters
    rad = lambda coor: radians(abs(coor))
    a = lambda lat1, lon1, lat2, lon2: \
            sin((rad(lat2) - rad(lat1)) / 2)**2 + \
            cos(rad(lat1)) * cos(rad(lat2)) * sin((rad(lon2) - rad(lon1)) / 2)**2
    c = lambda a: 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = lambda lat1, long1, lat2, lon2: round(R * c(a(lat1, long1, lat2, lon2)), 1)
    
    combined['crows_distance'] = combined.apply(
        lambda row: distance(
            row['dropoff_latitude'],
            row['dropoff_longitude'],
            row['pickup_latitude'],
            row['pickup_longitude']),
        axis=1)
    
    # drop suspicious rows from the training data
    combined = combined[
         (combined['set'] == 'test') |
        ((combined['set'] == 'train') &
         (combined['crows_distance'] <= MAX_DISTANCE) &
         (combined['trip_duration'] <= MAX_DURATION))]

    # PCA looks pointless, but who knows.
    coords = np.vstack((combined[['pickup_latitude', 'pickup_longitude']].values,
                        combined[['dropoff_latitude', 'dropoff_longitude']].values))
    
    pca = PCA().fit(coords)
    combined['pickup_pca0'] = pca.transform(combined[['pickup_latitude', 'pickup_longitude']])[:, 0]
    combined['pickup_pca1'] = pca.transform(combined[['pickup_latitude', 'pickup_longitude']])[:, 1]
    combined['dropoff_pca0'] = pca.transform(combined[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
    combined['dropoff_pca1'] = pca.transform(combined[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
    combined['pca_manhattan'] = \
        np.abs(combined['dropoff_pca1'] - combined['pickup_pca1']) + \
        np.abs(combined['dropoff_pca0'] - combined['pickup_pca0'])
    
    # cluster the lat/lon using KMeans
    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])
    combined['pickup_cluster'] = kmeans.predict(combined[['pickup_latitude', 'pickup_longitude']])
    combined['dropoff_cluster'] = kmeans.predict(combined[['dropoff_latitude', 'dropoff_longitude']])

    combined.to_csv('./data/combined.csv', sep=',', index=None)

else:
    # already done pre=processing
    combined = pd.read_csv('./data/combined.csv')

# take the log of the measure. this'll give a normal distribution as well as
# allow us to use RMSE as the loss function instead of RMSLE on the original
combined['trip_duration'] = combined['trip_duration'].apply(lambda t: max(0.01, t))
combined['trip_duration'] = np.log(combined['trip_duration'].values)

# bring in external data about actual distance by road
# source: https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm
usecols = ['id', 'total_distance', 'total_travel_time', 'number_of_steps']
fr1 = pd.read_csv('./data/osrm/fastest_routes_train_part_1.csv', usecols=usecols)
fr2 = pd.read_csv('./data/osrm/fastest_routes_train_part_2.csv', usecols=usecols)
test_street_info = pd.read_csv('./data/osrm/fastest_routes_test.csv', usecols=usecols)
train_street_info = pd.concat([fr1, fr2])

train = combined[combined['set'] == 'train'] # filter back down to train rows
train = train.merge(train_street_info, how='left', on='id')
train.dropna(inplace=True) # there was 1 null row introduced by the join

# ==============================================
# Train XGB model to estimate trip duration
# ==============================================

exclude = ['id', 'set']                      # we won't use these columns for training
loss_column = 'trip_duration'                # this is what we're trying to predict
features = [col for col in train.columns if col not in exclude and col != loss_column]

print('\nTraining and scoring XGBoost...')
baseline_params = h.get_params(algorithm='xgb', ptype='start', ver=3)
#alg = xgb.XGBClassifier( **baseline_params )
alg = xgb.XGBRegressor( **baseline_params )

# tune the model
xgb_model, importance = \
    h.fit_model(
        alg,
        train,
        features=features,
        loss=loss_column,
        useTrainCV=True,
        folds=5,
        metrics=['rmse'],
        chatty=2,
        show_report=True)


# ==============================================
# Produce estimates for XGB
# ==============================================

X = train[features]
if hasattr(xgb_model, 'best_ntree_limit'):
    xgb_ytmp = xgb_model.predict(X, ntree_limit=xgb_model.best_ntree_limit)
else:
    xgb_ytmp = xgb_model.predict(X)

# reshape for tensor input
xgb_ytmp = xgb_ytmp.reshape(xgb_ytmp.shape[0], 1)


# ==============================================
# Train the neural net to estimate trip duration
# ==============================================

epochs = 20                                  # number of passes across the training data
batch_size = 2**9                            # number of samples trained per pass
                                             # (use big batches when using batchnorm)
lr_decay_factor = 0.5
lr_decay_epoch = max(1, round(lr_decay_factor * 0.6 * epochs))
early_stopping_rounds = 26
lr = 0.013
cv = 0.2

feature_count = len(features)

# instantiate the neural net(s)
nets = [
    TaxiNet(
        feature_count,
        learn_rate=lr + (lr * (random() - 0.5) * 0.4), # decays over time (+- 40%)
        cuda=False) for _ in range(ENSEMBLE_COUNT)
    ]

# train each neural net
trained_nets = []
estimates = []
_, train_x, _ = next(nets[0].get_batches(train, loss_column, batch_size=train.shape[0], exclude=exclude))
train_x.volatile=True
for ii, net in enumerate(nets):
    print("Training net {}.".format(ii))
    net.learn_loop(train, loss_column, epochs, batch_size, exclude,
                   lr_decay_factor, lr_decay_epoch, cv, early_stopping_rounds)
    estimate = net.forward(train_x)
    trained_nets.append(net)
    estimates.append(estimate.data.numpy())

# arrange the estimates of the ensemble as features into a new design matrix
estimates.append(xgb_ytmp)
estimates.append(train['crows_distance'].values.reshape(train['crows_distance'].values.shape[0], 1))
estimates.append(train[loss_column].values.reshape(train[loss_column].values.shape[0], 1))
new_features = np.hstack(estimates)
new_features = pd.DataFrame(new_features)

# train the stacked model
print("Training regressor net.")
exclude = []

# the stacked regressor will have the N neural nets + the XGB model as input
stacked_feature_count = (new_features.shape[1] - 1)
stacked_regressor = TaxiCombinerNet(stacked_feature_count, learn_rate=0.004, max_output=MAX_DURATION)
stacked_regressor.learn_loop(
    new_features,
    stacked_feature_count, # the last column is the loss column
    epochs,
    batch_size,
    exclude=[],
    lr_decay_factor=lr_decay_factor,
    lr_decay_epoch=lr_decay_epoch,
    cv=cv,
    early_stopping_rounds=early_stopping_rounds)


# ==============================================
# Produce estimates for the test set
# ==============================================

exclude = ['id', 'set']                      # we won't use these columns for training
test_estimates = []
test = combined[combined['set'] == 'test'] # filter back down to test rows
test = test.merge(test_street_info, how='left', on='id')

_, test_x, test_y = next(stacked_regressor.get_batches(test, loss_column, batch_size=test.shape[0], exclude=exclude))
test_x.volatile = True
test_y.volatile = True
for ii, net in enumerate(trained_nets): # TODO : multiprocess
    print("Evaluating net {}.".format(ii))
    test_estimate = net.forward(test_x)
    test_estimates.append(test_estimate.data.numpy())

# predict model (check for early stopping rounds)
print('Evaluating XGB.')
X = test[features]
if hasattr(xgb_model, 'best_ntree_limit'):
    xgb_ytmp = xgb_model.predict(X, ntree_limit=xgb_model.best_ntree_limit)
else:
    xgb_ytmp = xgb_model.predict(X)

# reshape for tensor input
xgb_ytmp = xgb_ytmp.reshape(xgb_ytmp.shape[0], 1)
test_estimates.append(xgb_ytmp)
test_estimates.append(test['crows_distance'].values.reshape(test['crows_distance'].values.shape[0], 1))

print("Evaluating regressor.")
test_estimates = np.hstack(test_estimates)
test_estimates = Variable(torch.Tensor(test_estimates), volatile=True)
exclude = []
test_output = stacked_regressor(test_estimates)

if stacked_regressor.cuda:
    test[loss_column] = test_output.cpu().numpy()
else:
    test[loss_column] = test_output.data.numpy()

# convert from log space back to linear space for final estimates
test[loss_column] = np.exp(test[loss_column].values)
test['trip_duration_xgb'] = np.exp(test_estimates.data.numpy()[:,3].reshape(test_estimates.data.shape[0], 1))
test_end_time = datetime.utcnow()
test_out = test[['id', loss_column]]
test_out_xgb = test[['id', 'trip_duration_xgb']]
test_out_xgb.columns = ['id', 'trip_duration']
model_path = \
    './models/{}_{:.3}'.format(
        datetime.strftime(test_end_time,"%Y-%m-%d-%H-%M-%S"),
        stacked_regressor.best_cv_loss)
os.mkdir(model_path)
test_out.to_csv('{}/submission.csv'.format(model_path), sep=',', index=None)
test_out_xgb.to_csv('{}/submission_xgbonly.csv'.format(model_path), sep=',', index=None)
torch.save(stacked_regressor.state_dict(), '{}/regressor.nn'.format(model_path))
for ii, n in enumerate(trained_nets):
    torch.save(n.state_dict(), '{}/ensemble_{}.nn'.format(model_path, ii))

# save the XGB model and a standalone model submission
h.save_model(xgb_model, '{}/xgb.model'.format(model_path))
