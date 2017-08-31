# reference: https://www.kaggle.com/c/nyc-taxi-trip-duration
# feature analysis: https://www.kaggle.com/headsortails/nyc-taxi-eda-update-the-fast-the-curious

from taxinet import TaxiNet
import pandas as pd
import numpy as np
from datetime import datetime
from math import sin, cos, sqrt, atan2, radians

# ===============================
# Date extraction
# ===============================

# read data
test = pd.read_csv('./data/test.csv')
train = pd.read_csv('./data/train.csv')

# label data
test['set'] = 'test'
train['set'] = 'train'

# instantiate the loss column in the test set so that schemas match
test['trip_duration'] = np.NaN

# union `join='outer'` the train and test data so that encoding can be done holistically
# and reset the index to be monotically increasing and distinct
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

# trim "id" off the front of the id string column and make it an int type
combined['id'] = combined['id'].apply(lambda id: int(id[2:]))

# round latitude/longitude to a practical precision. 5 decimal places == 1 meter
# (should not affect loss estimate but should improve training)
combined['dropoff_latitude'] = combined['dropoff_latitude'].apply(lambda loc: round(loc, 5))
combined['dropoff_longitude'] = combined['dropoff_longitude'].apply(lambda loc: round(loc, 5))
combined['pickup_latitude'] = combined['pickup_latitude'].apply(lambda loc: round(loc, 5))
combined['pickup_longitude'] = combined['pickup_longitude'].apply(lambda loc: round(loc, 5))

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
max_distance = 100 * 10**3  # 100 km
max_duration = 12 * 60 * 60 # 12 hours
combined = combined[
      (combined['set'] == 'train')
    & (combined['crows_distance'] <= max_distance)
    & (combined['trip_duration'] <= max_duration)]


# ==============================================
# Train the neural net to estimate trip duration
# ==============================================

epochs = 10                                  # number of passes across the training data
train = combined[combined['set'] == 'train'] # filter back down to train rows
exclude = ['id', 'set']                      # we won't use these columns for training
loss_column = 'trip_duration'                # this is what we're trying to predict
batch_size = 1024                            # number of samples trained per pass
feature_count = len([col for col in train.columns if col not in exclude and col != loss_column])
taxi_net = TaxiNet(feature_count, cuda=True) # instantiate the neural net graph

for epoch in range(epochs):
    for batch_idx, batch_x, batch_y in taxi_net.get_batches(train, loss_column, batch_size=1024, exclude=exclude):
        
        # Forward pass then backward pass
        # TODO : custom loss function matching the Kaggle requirements
        # TODO : convolutional layers on the coordinates
        output = taxi_net(batch_x)
        loss = taxi_net.learn(output, batch_y)

        print('\rLoss: {:.3f} after {} batches ({:.1f}%), {} epochs.{}'.format(
                loss.data[0],
                batch_idx,
                100 * batch_idx * batch_size / train.shape[0],
                epoch,
                "       "), end="")

    # shuffle the data so that new batches / orders are used in the next epoch
    train = train.sample(frac=1).reset_index(drop=True)
    print('\nLoss: {:.3f} after {} epochs'.format(loss.data[0], epoch))


# ==============================================
# Produce estimates for the test set
# ==============================================

# TODO