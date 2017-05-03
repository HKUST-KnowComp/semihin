from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import sys
import cPickle as pk
from graphgenerator import GraphGenerator
from datareader import DataReaderWang
from sklearn.semi_supervised import LabelPropagation


diff = ['GSPO','GSCI','GHEA']
sim = ['GSPO','GTOUR','GWELF']
#scope_name = 'GDIF'
scope_name = 'GDIF'


with open('/home/data/corpora/HIN/dump/GDIF.dmp') as f:
    hin = pk.load(f)

X, newIds = GraphGenerator.getTFVectorX(hin,param={'word':True, 'entity':True, 'we_weight':1})
print X.shape
y = GraphGenerator.gety(hin)

lp_cand = range(1, 11)
for lp in lp_cand:
    results = []
    for r in range(50):
        trainLabel = []
        testLabel = []
        with open('/home/hejiang/results/' + scope_name + '_lb' + str(lp).zfill(3) + '_' + str(r).zfill(3) + '_train') as f:
            trainOldLabel = pk.load(f)
        with open('/home/hejiang/results/' + scope_name + '_lb' + str(lp).zfill(3) + '_' + str(r).zfill(3) + '_test') as f:
            testOldLabel = pk.load(f)
        for l in trainOldLabel:
            trainLabel.append(newIds[l])
        for l in testOldLabel:
            testLabel.append(newIds[l])

        XTrain = X[trainLabel]
        XTest = X[testLabel]
        yTrain = y[trainLabel]
        yTest = y[testLabel]

        # train
        #clf = LinearSVC(C=0.01)
        clf = LogisticRegression(C=0.01)
        clf.fit(XTrain, yTrain)

        # test
        pred = clf.predict(XTest)
        results.append(sum(pred == yTest) / float(yTest.shape[0]))
    print str(np.mean(results)) + '\t' + str(np.std(results))



