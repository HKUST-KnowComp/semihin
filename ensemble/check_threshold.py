import numpy as np
import scipy.sparse as sparse
from graphgenerator import GraphGenerator
from classifier import SSLClassifier
import cPickle as pk
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import os
import subprocess

DIFF = ['rec.autos', 'comp.os.ms-windows.misc', 'sci.space']
SIM = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc']
GSIM = ['GWEA','GDIS','GENV']
GDIF = ['GENT','GODD','GDEF']

NG20TypeList = [['astronomy','spaceflight'], ['computer'],['sports','automotive'],['medicine','organization','biology','business'],['computer','astronomy','spaceflight','automotive']]



def NG20DIFFCheckThreshold():
    # this section should be changed between different scopes
    typeList = NG20TypeList
    experiment_path = '/home/hejiang/results/DIFF_'
    scope_name = 'DIFF'
    scope = DIFF

    lb_cand = range(1,11)
    repeats = 50

    maxPreds = {}
    for t in typeList:
        maxPreds[str(t)] = []

    for lb in lb_cand:
        results = []
        for r in range(repeats):
            with open('/home/hejiang/results/' + 'DIFF' + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_train') as f:
                trainLabel = pk.load(f)
            with open('/home/hejiang/results/' + 'DIFF' + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_test') as f:
                testLabel = pk.load(f)

            # check quartile for threshold
            for t in typeList:
                with open(experiment_path + 'lb' + str(lb).zfill(3) + '_' + str(t) + '_' + str(r).zfill(3) + '_train') as f:
                    trainPred = pk.load(f)
                    for i,k in enumerate(trainLabel.keys()):
                        v = scope[np.argmax(trainPred[i,:])]
                        # some potential improvement: set a threshold for random walk number to block
                        # 'unconfident' data points
                        max = np.max(trainPred[i,:])
                        maxPreds[str(t)].append(max)

                with open(experiment_path + 'lb' + str(lb).zfill(3) + '_' + str(t) + '_' + str(r).zfill(3) + '_test') as f:
                    testPred = pk.load(f)
                    for i,k in enumerate(testLabel.keys()):
                        v = scope[np.argmax(testPred[i,:])]
                        # some potential improvement: set a threshold for random walk number to block
                        # 'unconfident' data points
                        max = np.max(testPred[i,:])
                        maxPreds[str(t)].append(max)

    for t in typeList:
        print t
        a = np.array(maxPreds[str(t)])
        print 'mean ' + str(np.mean(a))
        print 'std ' + str(np.std(a))
        print 'max ' + str(np.max(a))
        print '95 percentile ' + str(np.percentile(a,95))
        print '90 percentile ' + str(np.percentile(a,90))
        print 'up quartile ' + str(np.percentile(a,75))
        print 'down quartile ' + str(np.percentile(a,25))
        print 'min ' + str(np.min(a))

NG20DIFFCheckThreshold()