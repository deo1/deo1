# reference: https://www.kaggle.com/c/nyc-taxi-trip-duration
# reference: http://www.faqs.org/faqs/ai-faq/neural-nets/part1/preamble.html
# feature analysis: https://www.kaggle.com/headsortails/nyc-taxi-eda-update-the-fast-the-curious

from taxinet import TaxiNet
import torch
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
     (combined['set'] == 'test') |
    ((combined['set'] == 'train') &
     (combined['crows_distance'] <= max_distance) &
     (combined['trip_duration'] <= max_duration))]

# TODO : see if loss is lognormal, convert for training then back for scoring

# ==============================================
# Train the neural net to estimate trip duration
# ==============================================

epochs = 20                                  # number of passes across the training data
train = combined[combined['set'] == 'train'] # filter back down to train rows
exclude = ['id', 'set']                      # we won't use these columns for training
loss_column = 'trip_duration'                # this is what we're trying to predict
batch_size = 2**12                           # number of samples trained per pass
feature_count = len([col for col in train.columns if col not in exclude and col != loss_column])
taxi_net = TaxiNet(feature_count, learn_rate=0.002, cuda=True) # instantiate the neural net

for epoch in range(epochs):
    for batch_idx, batch_x, batch_y in taxi_net.get_batches(train, loss_column, batch_size=batch_size, exclude=exclude):
        
        # Forward pass then backward pass
        # TODO : custom loss function matching the Kaggle requirements
        # TODO : convolutional layers on the coordinates
        # TODO : cross validation
        output = taxi_net(batch_x)
        loss = taxi_net.learn(output, batch_y)
        
        print('\rLoss: {:.3f} after {} batches ({:.1f}%), {} epochs. (med(y): {:.1f}){}'.format(
                loss.data[0],                                  # iteration loss
                batch_idx,                                     # iteration count
                100 * batch_idx * batch_size / train.shape[0], # % complete within epoch
                epoch,                                         # epoch count
                output.median().data[0],                       # to monitor that the weights haven't saturated to 0
                "       "), end="")

    # score and train on the whole set to see where we're at
    _, all_x, all_y = next(taxi_net.get_batches(train, loss_column, batch_size=train.shape[0], exclude=exclude))
    print('\nLoss: {:.3f} after {} epochs'.format(taxi_net.loss_function(taxi_net(all_x), all_y).data[0], epoch))
    
    # shuffle the data so that new batches / orders are used in the next epoch
    train = train.sample(frac=1).reset_index(drop=True)


# ==============================================
# Produce estimates for the test set
# ==============================================

test = combined[combined['set'] == 'test'] # filter back down to test rows
_, test_x, test_y = next(taxi_net.get_batches(test, loss_column, batch_size=test.shape[0], exclude=exclude))
test[loss_column] = taxi_net(test_x).cpu().numpy()

# make all trip durations >= 0
test[loss_column] = test[loss_column].apply(lambda loss: max(0, loss))

test_out = test[['id', loss_column]]
test_out.to_csv('./data/submission_{}.csv'.format(datetime.strftime(datetime.utcnow(),"%Y-%m-%d-%H-%M-%S")), sep = ',', index = None)
torch.save(taxi_net.state_dict(), './models/submission_{}.nn'.format(datetime.strftime(datetime.utcnow(),"%Y-%m-%d-%H-%M-%S")))
# TODO