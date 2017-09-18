"""
xgboost param doc: https://github.com/dmlc/xgboost/blob/master/doc/parameter.md
"""

class Colclass(dict):

    def __init__(self):
        self['float64'] = 'continuous'
        self['float32'] = 'continuous'
        self['int64'] = 'discrete'
        self['int32'] = 'discrete'
        self['object'] = 'categorical'

class Xgbparams1(dict):

    def __init__(self):
        self['objective'] = "reg:linear"
        self['learning_rate'] = 0.1
        self['gamma'] = 0.5290
        self['min_child_weight'] = 4.2922
        self['colsample_bytree'] = 1
        self['subsample'] = 0.9930
        self['max_depth'] = 7
        self['max_delta_step'] = 0
        self['silent'] = 0
        self['n_estimators'] = 100 # num_boosting_rounds

class Xgbparams2(dict): # useful for tuning

    __alpha = 30 # rate inversely proportional to number of rounds
    __n_estimators = 3000 # num_boosting_rounds
    __learning_rate = min(1, max(0.1, __alpha / __n_estimators)) # step size

    def __init__(self):
        self['objective'] = "reg:linear"
        self['learning_rate'] = self.__learning_rate
        self['gamma'] = 0.2
        self['min_child_weight'] = 4.2922
        self['colsample_bytree'] = 0.3
        self['subsample'] = 0.5
        self['max_depth'] = 4
        self['max_delta_step'] = 0
        self['silent'] = 0
        self['n_estimators'] = self.__n_estimators

class Xgbparams3(dict): # useful for final run

    __alpha = 30 # rate inversely proportional to number of rounds
    __n_estimators = 1000 # num_boosting_rounds
    __learning_rate = min(1, max(0.1, __alpha / __n_estimators)) # step size

    def __init__(self):
        self['objective'] = "reg:linear"
        self['learning_rate'] = self.__learning_rate
        self['gamma'] = 0.1
        self['min_child_weight'] = 4.2922
        self['colsample_bytree'] = 0.8
        self['subsample'] = 0.8
        self['max_depth'] = 6
        self['max_delta_step'] = 0
        self['silent'] = 0
        self['n_estimators'] = self.__n_estimators

class Xgbparams4(dict): # useful for final run

    __alpha = 30 # rate inversely proportional to number of rounds
    __n_estimators = 2500 # num_boosting_rounds
    __learning_rate = min(1, max(0.1, __alpha / __n_estimators)) # step size

    def __init__(self):
        self['objective'] = "reg:linear"
        self['learning_rate'] = self.__learning_rate
        self['gamma'] = 1.1
        self['min_child_weight'] = 4.2922
        self['colsample_bytree'] = 0.8
        self['subsample'] = 0.95
        self['max_depth'] = 7
        self['max_delta_step'] = 0
        self['silent'] = 0
        self['n_estimators'] = self.__n_estimators
        self['reg_alpha'] = 0.1

class Xgbparams5(dict): # useful for final run

    __alpha = 25 # rate inversely proportional to number of rounds
    __n_estimators = 2500 # num_boosting_rounds
    __learning_rate = min(1, max(0.05, __alpha / __n_estimators)) # step size

    def __init__(self):
        self['objective'] = "reg:linear"
        self['learning_rate'] = self.__learning_rate
        self['gamma'] = 2.0
        self['min_child_weight'] = 6.0
        self['colsample_bytree'] = 1.0
        self['colsample_bylevel'] = 1.0
        self['subsample'] = 0.95
        self['max_depth'] = 8
        self['max_delta_step'] = 0
        self['silent'] = 0
        self['n_estimators'] = self.__n_estimators
        self['reg_alpha'] = 0.1

class Xgbparams6(dict): # useful for final run
    '''
    This is experimenting on the best yet version 5
    '''
    __alpha = 25 # rate inversely proportional to number of rounds
    __n_estimators = 100 # num_boosting_rounds
    __learning_rate = min(1, max(0.05, __alpha / __n_estimators)) # step size

    def __init__(self):
        self['objective'] = "reg:linear"
        self['learning_rate'] = self.__learning_rate
        self['gamma'] = 4.0
        self['min_child_weight'] = 5.5
        self['colsample_bytree'] = 1
        self['colsample_bylevel'] = 1
        self['subsample'] = 0.9920 # good for multiple training
        self['max_depth'] = 5
        self['max_delta_step'] = 0
        self['silent'] = 0
        self['n_estimators'] = self.__n_estimators
        self['reg_alpha'] = 0.5

class CVparams1(list): # test -- small

    def __init__(self):
       super().__init__([
            # round 0 initialization
            # Note: discrete int intervals must be enforced here if necessary
            [{'name':'max_depth',
              'min':2,
              'steps':2,
              'max':3 },

             {'name':'min_child_weight',
              'min':0,
              'steps':2,
              'max':1 }],

            # round 1 initialization
            [{'name':'gamma',
              'min':0,
              'steps':6,
              'max':1 },

             {'name':'colsample_bytree',
              'min':0.5,
              'steps':6,
              'max':1 }] ])

class CVparams2(list):

    def __init__(self):
       super().__init__([
            # round 0 initialization
            [{'name':'max_depth',
              'min':3,
              'steps':2, # Limit depth in the first pass
              'max':4 }, # has to be an int

             {'name':'min_child_weight',
              'min':3,
              'steps':8,
              'max':6 }],

            # round 1 initialization
            [{'name':'gamma',
              'min':0.2,
              'steps':10,
              'max':2 },

             {'name':'colsample_bytree',
              'min':0.6,
              'steps':5,
              'max':1 }],

              # round 2 initialization
            [{'name':'colsample_bylevel',
              'min':0.6,
              'steps':5,
              'max':1 }], # cannot be greater than 1c

              # round 3 initialization
            [{'name':'max_depth',
              'min':3,
              'steps':6, # Limit depth in the first pass
              'max':8 }] ])
