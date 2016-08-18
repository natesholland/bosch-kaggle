import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split

# import code; code.interact(local=dict(globals(), **locals()))

xtrain = pd.read_csv('data/train_numeric.csv')
xtrain.fillna(value = -99, inplace = True)

import code; code.interact(local=dict(globals(), **locals()))
(train_data, val_data) = train_test_split(xtrain)

id_train = xtrain['Id']
ytrain = train_data['Response']
xtrain = train_data.drop(['Id', 'Response'], axis = 1)

yval = val_data['Response']
xval = val_data.drop(['Id', 'Response'], axis = 1)


dtrain = xgb.DMatrix(xtrain, ytrain)
del xtrain
del ytrain
dvalid = xgb.DMatrix(xval, yval)
del xval
del yval

params = {
    "objective": "binary:logistic",
    "booster": "gblinear",
    # "max_depth": 5,
    # "eta": 0.02,
    # "silent": 1,
    # "alpha": 3,
    # "min_child_weight": 2.7,
    # "gamma": 0,
    # "subsample": 0.5,
    # "colsample_bytree": 0.5
}


watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, 100, evals=watchlist,
                early_stopping_rounds=25, verbose_eval=True)

del dtrain
del dvalid

xtest = pd.read_csv('data/test_numeric.csv')
xtest.fillna(value = -99, inplace = True)

id_test = xtest['Id']
xtest.drop(['Id'], axis = 1, inplace = True)


pred = gbm.predict(xgb.DMatrix(xtest))

del xtest

xsub = pd.read_csv('data/sample_submission.csv')
xsub['Response'] = pred[:,1]
xsub['Response'] = xsub['Response'].round()
xsub['Response'] = xsub['Response'].astype(int)
xsub.to_csv('etrees_quickie.csv', index = False)
