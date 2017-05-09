from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import numpy as np
import sys
import cPickle as pk
from graphgenerator import GraphGenerator
from datareader import DataReaderWang
from sklearn.semi_supervised import LabelPropagation

DIFF = ['rec.autos', 'comp.os.ms-windows.misc', 'sci.space']
SIM = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc']
GSIM = ['GWEA','GDIS','GENV']
GDIF = ['GENT','GODD','GDEF']

print 'SVM'
lp_cand = range(1, 11)

def run():
    with open('/home/hejiang/results/' + scope_name + '_Ids') as f:
        Ids = pk.load(f)
    with open('/home/data/corpora/HIN/dump/' + scope_name + '_newIds') as f:
        newIds = pk.load(f)
    with open('/home/data/corpora/HIN/dump/' + scope_name + '_DocIds') as f:
        DocIds = pk.load(f)
    mapping = {}
    for k,v in DocIds.items():
        mapping[newIds[Ids[str(k)]]] = v
    with open('/home/hejiang/results/embed/' + scope_name + '_mapping','w') as f:
        pk.dump(mapping,f)
    with open('/home/hejiang/results/embed/' + scope_name + '_WEavg') as f:
        X = pk.load(f)
    with open('/home/hejiang/results/embed/' + scope_name + '_y') as f:
        y = pk.load(f)

    for lp in lp_cand:
        results = []
        for r in range(50):
            with open('/home/hejiang/results/' + scope_name + '_lb' + str(lp).zfill(3) + '_' + str(r).zfill(
                    3) + '_train') as f:
                trainLabel = pk.load(f)
            with open('/home/hejiang/results/' + scope_name + '_lb' + str(lp).zfill(3) + '_' + str(r).zfill(
                    3) + '_test') as f:
                testLabel = pk.load(f)
            train = []
            test = []
            for k in trainLabel:
                train.append(mapping[k])
            for k in testLabel:
                test.append(mapping[k])

            XTrain = X[train]#.toarray()
            XTest = X[test]#.toarray()
            yTrain = y[train]
            yTest = y[test]

            # train
            clf = LinearSVC(C=0.01)
            #clf = LogisticRegression(C=0.01)
            #clf = MultinomialNB()
            clf.fit(XTrain, yTrain)

            # test
            pred = clf.predict(XTest)
            results.append(sum(pred == yTest) / float(yTest.shape[0]))
        print str(np.mean(results)) + '\t' + str(np.std(results))


scope_names = ['SIM','DIFF','GSIM','GDIF']
scopes = [SIM,DIFF,GSIM,GDIF]

for i in range(4):
    scope = scopes[i]
    scope_name = scope_names[i]
    print scope_name
    run()