# reference: https://www.kaggle.com/c/nyc-taxi-trip-duration
# reference: http://www.faqs.org/faqs/ai-faq/neural-nets/part1/preamble.html
# feature analysis: https://www.kaggle.com/headsortails/nyc-taxi-eda-update-the-fast-the-curious

from taxinet import TaxiNet
import torch
import pandas as pd
import numpy as np
from datetime import datetime
from math import sin, cos, sqrt, atan2, radians

# TODO args feature
RUN_FEATURE_EXTRACTION = False
MAX_DISTANCE = 100 * 10**3  # 100 km
MAX_DURATION = 12 * 60 * 60 # 12 hours

if (RUN_FEATURE_EXTRACTION):
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
    combined = combined[
         (combined['set'] == 'test') |
        ((combined['set'] == 'train') &
         (combined['crows_distance'] <= MAX_DISTANCE) &
         (combined['trip_duration'] <= MAX_DURATION))]
    
    # TODO : see if loss is lognormal, convert for training then back for scoring
    
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
# Train the neural net to estimate trip duration
# ==============================================

epochs = 100                                 # number of passes across the training data
exclude = ['id', 'set']                      # we won't use these columns for training
loss_column = 'trip_duration'                # this is what we're trying to predict
batch_size = 2**13                           # number of samples trained per pass
                                             # (use big batches when using batchnorm)
feature_count = len([col for col in train.columns if col not in exclude and col != loss_column])

# instantiate the neural net
taxi_net = TaxiNet(
    feature_count,
    learn_rate=0.014, # decays over time
    cuda=False,
    max_output=MAX_DURATION)

taxi_net.learn_loop(train, loss_column, epochs, batch_size, exclude, 0.5, 25)

# ==============================================
# Produce estimates for the test set
# ==============================================

test = combined[combined['set'] == 'test'] # filter back down to test rows
test = test.merge(test_street_info, how='left', on='id')
_, test_x, test_y = next(taxi_net.get_batches(test, loss_column, batch_size=test.shape[0], exclude=exclude))

#taxi_net.eval() # test mode (rolling avg for batchnorm) # only apply for single sample
test_output = taxi_net(test_x)

if taxi_net.cuda:
    test[loss_column] = test_output.cpu().numpy()
else:
    test[loss_column] = test_output.data.numpy()

# convert from log space back to linear space for final estimates
test[loss_column] = np.exp(test[loss_column].values)

test_out = test[['id', loss_column]]
test_out.to_csv('./data/submission_{}.csv'.format(datetime.strftime(datetime.utcnow(),"%Y-%m-%d-%H-%M-%S")), sep = ',', index = None)
torch.save(taxi_net.state_dict(), './models/submission_{}.nn'.format(datetime.strftime(datetime.utcnow(),"%Y-%m-%d-%H-%M-%S")))
# TODO