from sklearn.svm import LinearSVC
import numpy as np
import cPickle as pk
from graphgenerator import GraphGenerator

scope_name = 'GDIF'
with open('/home/data/corpora/HIN/dump/' + scope_name + '.dmp') as f:
    hin = pk.load(f)

X, newIds,e_newIds = GraphGenerator.getTFVectorX(hin,param={'word':False, 'entity':True, 'we_weight':1})
y = GraphGenerator.gety(hin)

trainLabel = np.random.choice(X.shape[0],2700,replace=False)

XTrain = X[trainLabel]
yTrain = y[trainLabel]

# train
clf = LinearSVC(C=0.01)
# clf = LogisticRegression(C=0.01)
clf.fit(XTrain, yTrain)

# examine weight
coef_max = np.max(clf.coef_, axis=0) - np.min(clf.coef_, axis=0)

max_displays = 100
strong_features = coef_max.argsort()[-max_displays:][::-1]

oldIds = hin.Ids
revNewIds = { v:k for k,v in e_newIds.items()}
revOldIds = { v:k for k,v in oldIds.items()}

print len(strong_features)
with open('/home/hejiang/results/' + scope_name + '_features','w') as f:
    pk.dump(strong_features,f)

for idx in strong_features:
    print revOldIds[revNewIds[idx]]