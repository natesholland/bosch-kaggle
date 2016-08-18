import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# import code; code.interact(local=dict(globals(), **locals()))

xtrain = pd.read_csv('data/train_numeric.csv')
xtrain.fillna(value = -99, inplace = True)

id_train = xtrain['Id']
ytrain = xtrain['Response']
xtrain.drop(['Id', 'Response'], axis = 1, inplace = True)

recognizer = RandomForestClassifier(100, max_depth=20, verbose=1, n_jobs=3)
recognizer.fit(xtrain, ytrain)

del xtrain

xtest = pd.read_csv('data/test_numeric.csv')
xtest.fillna(value = -99, inplace = True)

id_test = xtest['Id']
xtest.drop(['Id'], axis = 1, inplace = True)

pred = recognizer.predict_proba(xtest)

del xtest

xsub = pd.read_csv('data/sample_submission.csv')
xsub['Response'] = pred[:,1]
xsub['Response'] = xsub['Response'].round()
xsub['Response'] = xsub['Response'].astype(int)
xsub.to_csv('etrees_quickie.csv', index = False)
