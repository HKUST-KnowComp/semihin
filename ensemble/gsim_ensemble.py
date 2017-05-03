import numpy as np
import scipy.sparse as sparse
from graphgenerator import GraphGenerator
from classifier import SSLClassifier
import cPickle as pk
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os
import subprocess

DIFF = ['rec.autos', 'comp.os.ms-windows.misc', 'sci.space']
SIM = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc']
GSIM = ['GWEA','GDIS','GENV']
GDIF = ['GENT','GODD','GDEF']

GSIMTypeList = [['military','sports'],['medicine'],[],['astronomy','spaceflight','automotive'],['location'],['location','organization'],['organization'],['government'],['location','organization','government']]

DIFF_threshold = {'[\'sports\', \'automotive\']':1e-3,
                  '[\'computer\']':1e-4,
                  '[\'medicine\', \'organization\', \'biology\', \'business\']':1e-3,
                  '[\'computer\', \'astronomy\', \'spaceflight\', \'automotive\']':-1,
                  '[\'astronomy\', \'spaceflight\']':1e-3
                }
GSIM_threshold = {'[\'sports\', \'automotive\']':1e-3,
                  '[\'computer\']':1e-4,
                  '[\'medicine\', \'organization\', \'biology\', \'business\']':1e-3,
                  '[\'computer\', \'astronomy\', \'spaceflight\', \'automotive\']':-1,
                  '[\'astronomy\', \'spaceflight\']':1e-3
                }



def NG20GSIMMetaPaths():
    experiment_path = '/home/hejiang/results/GSIM_'
    scope = GSIM

    with open('/home/data/corpora/HIN/dump/GSIM.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    print 'word option: ' + str(tf_param['word'])
    print 'entity option: ' + str(tf_param['entity'])
    print 'word-entity weight: ' + str(tf_param['we_weight'])

    typeList = GSIMTypeList
    for t in typeList:
        print t
        X, newIds = GraphGenerator.getTFVectorX(hin,tf_param,t)
        n = X.shape[0]
        e = X.shape[1]
        X = X.toarray()
        graph = np.zeros((n+e,n+e))
        graph[0:n,n:n+e] = X
        graph[n:n+e,0:n] = X.transpose()
        graph = sparse.csc_matrix(graph)

        newLabel = GraphGenerator.getNewLabels(hin)
        lp_param = {'alpha':0.98,'normalization_factor':0.01}
    #    lp_param = {'alpha':0.98, 'normalization_factor':5}
        # 3-class classification
        lp_candidate = [1,2,3,4,5,6,7,8,9,10]
        for lp in lp_candidate:
            ssl = SSLClassifier(graph, newLabel, scope, lp_param, repeatTimes=50, trainNumbers=lp)
            ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds,saveProb=True,savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_' + str(t))
            ssl.stats()

def NG20GSIMGALEnsemble():
    # this section should be changed between different scopes
    typeList = GSIMTypeList
    experiment_path = '/home/hejiang/results/SIM_'
    scope_name = 'SIM'
    scope = SIM

    threshold = DIFF_threshold

    lb_cand = range(1,11)
    repeats = 50

    with open('/home/data/corpora/HIN/dump/SIM.dmp') as f:
        hin = pk.load(f)
    #X, newIds = GraphGenerator.getTFVectorX(hin, param={'word': True, 'entity': False, 'we_weight': 0.1})
    y = GraphGenerator.gety(hin)

    command_file = open('/home/hejiang/code/get-another-label/bin/loop.sh','w')

    for lb in lb_cand:
        results = []
        for r in range(repeats):
            with open('/home/hejiang/results/' + scope_name + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_train') as f:
                #trainOldLabel = pk.load(f)
                trainLabel = pk.load(f)
            with open('/home/hejiang/results/' + scope_name + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_test') as f:
                #testOldLabel = pk.load(f)
                testLabel = pk.load(f)

            yTrain = y[trainLabel.keys()]
            yTest = y[testLabel.keys()]

            numTrain = len(trainLabel)
            numTest = len(testLabel)
            XTrain = np.zeros((numTrain, 0))
            XTest = np.zeros((numTest, 0))

            label_file = open('/home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + '_' +
                              str(r).zfill(3) + '_label.txt', 'w')
            gold_file = open('/home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + '_' +
                              str(r).zfill(3) + '_gold.txt', 'w')
            eval_file = open('/home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + '_' +
                             str(r).zfill(3) + '_eval.txt', 'w')

            # write get-another-label gold file
            for k,v in trainLabel.items():
                gold_file.write(str(k) + '\t' + v + '\n')

            # write get-another-label eval file
            for k,v in testLabel.items():
                eval_file.write(str(k) + '\t' + v + '\n')

            # write get-another-label label file
            for t in typeList:
                with open(experiment_path + 'lb' + str(lb).zfill(3) + '_' + str(t) + '_' + str(r).zfill(3) + '_train') as f:
                    trainPred = pk.load(f)
                    for i,k in enumerate(trainLabel.keys()):
                        v = scope[np.argmax(trainPred[i,:])]
                        label_file.write(str(t) + '\t' + str(k) + '\t' + v + '\n')

                with open(experiment_path + 'lb' + str(lb).zfill(3) + '_' + str(t) + '_' + str(r).zfill(3) + '_test') as f:
                    testPred = pk.load(f)
                    for i,k in enumerate(testLabel.keys()):
                        v = scope[np.argmax(testPred[i,:])]
                        # some potential improvement: set a threshold for random walk number to block
                        # 'unconfident' data points

                        max = np.max(testPred[i,:])
                        if max > threshold[str(t)]:
                            label_file.write(str(t) + '\t' + str(k) + '\t' + v + '\n')

            # run get-another-label batch
            command = r'/home/hejiang/code/get-another-label/bin/get-another-label.sh ' + \
                '--categories /home/hejiang/results/gal/' + scope_name + '_categories.txt ' + \
                '--cost /home/hejiang/results/gal/' + scope_name + '_costs.txt ' + \
                '--gold /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + \
                '_' + str(r).zfill(3) + '_gold.txt ' + \
                '--input /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + \
                '_' + str(r).zfill(3) + '_label.txt ' + \
                '--eval /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + \
                '_' + str(r).zfill(3) + '_eval.txt ' + \
                '> /home/hejiang/results/gal/' + scope_name + '_lb' + str(lb).zfill(3) + '_' + \
                str(r).zfill(3) + '_result.txt'

            command_file.write(command + '\n')

NG20GSIMMetaPaths()