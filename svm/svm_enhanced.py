from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import normalize
from sklearn import metrics
import numpy as np
import sys
import cPickle as pk
from graphgenerator import GraphGenerator
from datareader import DataReaderWang
from sklearn.semi_supervised import LabelPropagation
import scipy.sparse as sparse

DIFF = ['rec.autos', 'comp.os.ms-windows.misc', 'sci.space']
SIM = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc']
GSIM = ['GWEA','GDIS','GENV']
GDIF = ['GENT','GODD','GDEF']

print 'SVM'
lp_cand = range(1, 11)

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

            XTrain = X[trainLabel.keys()]
            XTest = X[testLabel.keys()]
            yTrain = y[trainLabel.keys()]
            yTest = y[testLabel.keys()]

            # train
            clf = LinearSVC(C=0.01)
            #lf = LogisticRegression(C=0.01)
            #clf = MultinomialNB()
            clf.fit(XTrain, yTrain)

            # test
            pred = clf.predict(XTest)
            results.append(sum(pred == yTest) / float(yTest.shape[0]))
        print str(np.mean(results)) + '\t' + str(np.std(results))


print 'GCAT DIFF'
scope_name = 'GDIF'
scope = GDIF
with open('/home/data/corpora/HIN/dump/GDIF.dmp') as f:
    hin = pk.load(f)
#with open('/home/hejiang/results/GDIF_features') as f:
with open('/home/hejiang/results/GDIF_laplacian_features') as f:
    ent_features = pk.load(f)
X_ent, newIds, newIds_ent = GraphGenerator.getTFVectorX(hin,param={'word':False, 'entity':True, 'we_weight':1})
X_word, newIds_, newIds_word = GraphGenerator.getTFVectorX(hin,param={'word':True, 'entity':False, 'we_weight':1})
#X_feat = X_word[:,ent_features].toarray()
print X_ent.shape
X_feat = X_ent[:,ent_features].toarray()
print X_feat.shape
print X_word.shape
X = np.concatenate([X_word.toarray(), X_feat],axis=1)
#with open('/home/hejiang/int/' + scope_name + '_' + str(X_feat.shape[1]) + '_Xword','w') as f:
#    pk.dump(sparse.csc_matrix(X_feat),f)
# uncomment this if you want word only results
#X = X_word
y = GraphGenerator.gety(hin)
run()
'''
print 'GCAT SIM'
scope_name = 'GSIM'
scope = GSIM
with open('/home/data/corpora/HIN/dump/GSIM.dmp') as f:
    hin = pk.load(f)
X_ent, newIds_ent = GraphGenerator.getTFVectorX(hin,param={'word':False, 'entity':True, 'we_weight':1})
X_word, newIds_word = GraphGenerator.getTFVectorX(hin,param={'word':True, 'entity':False, 'we_weight':1})
print X_ent.shape
y = GraphGenerator.gety(hin)
run()
'''