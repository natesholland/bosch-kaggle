import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

xtrain = pd.read_csv('data/train_numeric.csv')
xtrain.fillna(value = -99, inplace = True)

id_train = xtrain['Id']
ytrain = xtrain['Response']
xtrain.drop(['Id', 'Response'], axis = 1, inplace = True)

clf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1,
                            min_samples_leaf = 10, verbose = 1)
clf.fit(xtrain, ytrain)

xtest = pd.read_csv('data/test_numeric.csv')
xtest.fillna(value = -99, inplace = True)

id_test = xtest['Id']
xtest.drop(['Id'], axis = 1, inplace = True)

pred = clf.predict_proba(xtest)

xsub = pd.read_csv('data/sample_submission.csv')
xsub['Response'] = pred[:,1]
xsub['Response'] = xsub['Response'].round()
x['Response'] = x['Response'].astype(int)
xsub.to_csv('etrees_quickie.csv', index = False)
