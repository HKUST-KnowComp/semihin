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
lp_cand = [5]


def run():
    for lp in lp_cand:
        results = []
        for r in range(50):
            with open('/home/hejiang/results/' + scope_name + '_lb' + str(lp).zfill(3) + '_' + str(r).zfill(
                    3) + '_train') as f:
                trainLabel = pk.load(f)
            with open('/home/hejiang/results/' + scope_name + '_lb' + str(lp).zfill(3) + '_' + str(r).zfill(
                    3) + '_test') as f:
                testLabel = pk.load(f)

            XTrain = X[trainLabel.keys()].toarray()
            XTest = X[testLabel.keys()].toarray()
            yTrain = y[trainLabel.keys()]
            yTest = y[testLabel.keys()]

            # train
            clf = LinearSVC(C=0.01)
            #clf = LogisticRegression(C=0.01)
            #clf = MultinomialNB()
            clf.fit(XTrain, yTrain)

            # test
            pred = clf.predict(XTest)
            results.append(sum(pred == yTest) / float(yTest.shape[0]))
        print str(np.mean(results)) + '\t' + str(np.std(results))



word_option = False
entity_option = True
print 'word: ' + str(word_option)
print 'entity:' + str(entity_option)
'''
print '20NG DIFF'
scope_name = 'DIFF'
scope = DIFF
with open('/home/data/corpora/HIN/dump/DIFF.dmp') as f:
    hin = pk.load(f)
X, newIds = GraphGenerator.getTFVectorX(hin,param={'word':word_option, 'entity':entity_option, 'we_weight':0.1})

y = GraphGenerator.gety(hin)
run()

print '20NG SIM'
scope_name = 'SIM'
scope = SIM
with open('/home/data/corpora/HIN/dump/SIM.dmp') as f:
    hin = pk.load(f)
X, newIds = GraphGenerator.getTFVectorX(hin,param={'word':word_option, 'entity':entity_option, 'we_weight':0.1})
print X.shape
y = GraphGenerator.gety(hin)
run()
'''
print 'GCAT DIFF'
scope_name = 'GDIF'
scope = GDIF
with open('/home/data/corpora/HIN/dump/GDIF.dmp') as f:
    hin = pk.load(f)
X, newIds = GraphGenerator.getTFVectorX(hin,param={'word':word_option, 'entity':entity_option, 'we_weight':1})
print X.shape
y = GraphGenerator.gety(hin)
run()

print 'GCAT SIM'
scope_name = 'GSIM'
scope = GSIM
with open('/home/data/corpora/HIN/dump/GSIM.dmp') as f:
    hin = pk.load(f)
X, newIds = GraphGenerator.getTFVectorX(hin,param={'word':word_option, 'entity':entity_option, 'we_weight':1})
print X.shape
y = GraphGenerator.gety(hin)
run()