import pandas as pd

def get_params(self, algorithm, ptype, ver):
    import params as p

    if algorithm == 'xgb':
        if ptype == 'start':
            if ver == 1: param = p.Xgbparams1()
            if ver == 2: param = p.Xgbparams2()
            if ver == 3: param = p.Xgbparams3()
            if ver == 4: param = p.Xgbparams4()
            if ver == 5: param = p.Xgbparams5()
            if ver == 6: param = p.Xgbparams6()
        elif ptype == 'search':
            if ver == 1: param = p.CVparams1()
            if ver == 2: param = p.CVparams2()
        else:
            raise RuntimeError("Method Data.get_params : type must be \
                                'start' or 'search'")
    else:
        raise RuntimeError("Method Data.get_params() : Does not currently \
                            support algorithm '{}'".format(algorithm))

    return param

def fit_model(self, alg, df, features, loss, useTrainCV=True, folds=5,
                  early_stopping_rounds=100, metrics='mae',
                  chatty=1, show_report=False):
    import xgboost as xgb # http://xgboost.readthedocs.io/en/latest/python/python_intro.html
    import operator

    X = df[features]
    y = df[ [loss] ] # put loss in list so that a pd.DataFrame is returned instead of pd.Series

    #Fit the algorithm on the data
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=y)
        cvresult = xgb.cv(
            xgb_param,
            xgtrain,
            num_boost_round = alg.get_params()['n_estimators'],
            nfold = folds,
            metrics = metrics,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval = chatty==2)

        # use the best number of rounds from the CV
        best_nrounds = cvresult.shape[0] - 1
        alg.set_params(n_estimators=best_nrounds)
        print("Best num rounds: {}".format(best_nrounds))
        
    alg.fit(X, y, eval_metric=metrics)

    # feature importance
    importance = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)

    #Print model report:
    if show_report:
        import matplotlib.pylab as plt
        from matplotlib.pylab import rcParams
        rcParams['figure.figsize'] = 16, 4
        print("\nModel Report")
        importance.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

    importance = sorted(importance.items(), key=operator.itemgetter(1))
    importance = pd.DataFrame(importance, columns=['feature', 'fscore'])
    importance['relative'] = importance['fscore'] / importance['fscore'].sum()

    return alg, importance
