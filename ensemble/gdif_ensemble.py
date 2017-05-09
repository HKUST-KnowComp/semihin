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
SIM_count = {'comp.graphics':1000,'comp.sys.mac.hardware':1000,'comp.os.ms-windows.misc':1000}
DIFF_count = {'rec.autos':1000,'comp.os.ms-windows.misc':1000,'sci.space':1000}
GSIM_count = {'GWEA':1014,'GDIS':2083,'GENV':499}
GDIF_count = {'GENT':1062,'GODD':1096,'GDEF':542}


GDIFTypeList = [['military','sports'],
                ['medicine'],
                [],
                ['astronomy','spaceflight','automotive'],
                ['location'],['organization'],
                ['organization.organization_sector'],
                ['government'],
                ['religion','computer','aviation']]

GDIF_weight = {'[\'military\',\'sports\']':0.1,
                '\[\'medicine\'\]':0.1,
                '[]':0.5,
                '[\'astronomy\',\'spaceflight\',\'automotive\']':0.05,
                '[\'location\']':0.02,
                '[\'organization\']':0.03,
                '[\'organization.organization_sector\']':0.05,
                '[\'government\']':0.1,
                '[\'religion\',\'computer\',\'aviation\']':0.1
                }

def NG20GDIFMetaPaths():
    experiment_path = '/home/hejiang/results/GDIF_'
    scope = GDIF

    with open('/home/data/corpora/HIN/dump/GDIF.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    print 'word option: ' + str(tf_param['word'])
    print 'entity option: ' + str(tf_param['entity'])
    print 'word-entity weight: ' + str(tf_param['we_weight'])

    typeList = GDIFTypeList
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

def GCATDIFFCoTrainEnsemble():
    experiment_path = '/home/hejiang/results/GDIF_'

    with open('/home/data/corpora/HIN/dump/GDIF.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    scope = DIFF
    c = len(scope)
    threshold = GDIF_threshold
    weight = GDIF_weight
    lb_cand = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #lb_cand = [10]
    repeats = 50

    # rounds for alternating optimizatrion
    rounds = 20

    typeList = GDIFTypeList


    for rd in range(rounds):

        # step 1:
        # generate output of each meta-path
        for t in typeList:
            print t
            X, newIds,entIds = GraphGenerator.getTFVectorX(hin,tf_param,t)
            n = X.shape[0]
            e = X.shape[1]
            X = X.toarray()
            graph = np.zeros((n+e,n+e))
            graph[0:n,n:n+e] = X
            graph[n:n+e,0:n] = X.transpose()
            graph = sparse.csc_matrix(graph)

            newLabel = GraphGenerator.getNewLabels(hin)
            lp_param = {'alpha':0.99,'normalization_factor':0.01}
        #    lp_param = {'alpha':0.98, 'normalization_factor':5}
            # 3-class classification

            for lb in lb_cand:
                ssl = SSLClassifier(graph, newLabel, GDIF, lp_param, repeatTimes=repeats, trainNumbers=lb,classCount=GDIF_count)
                if rd == 0:
                    ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lb).zfill(3) + '_',
                                                    newIds=newIds,saveProb=True,savePathPrefix=experiment_path + 'lb' +
                                                    str(lb).zfill(3) + '_' + str(t))
                else:
                    with open('/home/hejiang/results/' + 'GDIF' + 'lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                            3) + '_pred_rd_' + str(rd-1).zfill(3)) as f:
                        inputPred = pk.load(f)
                    ssl.repeatedFixedExpeimentwithInput(pathPrefix=experiment_path + 'lb' + str(lb).zfill(3) + '_',
                                                    newIds=newIds,saveProb=True,savePathPrefix=experiment_path + 'lb' +
                                                    str(lb).zfill(3) + '_' + str(t),inputPred=inputPred)
                ssl.stats()

        # step 2:
        # propagate pseudo-label for other path
        for lb in lb_cand:
            results = []
            for r in range(repeats):
                with open('/home/hejiang/results/' + 'GDIF' + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                        3) + '_train') as f:
                    trainLabel = pk.load(f)
                with open('/home/hejiang/results/' + 'GDIF' + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                        3) + '_test') as f:
                    testLabel = pk.load(f)

                numTrain = len(trainLabel)
                numTest = len(testLabel)
                n = numTrain + numTest

                # write get-another-label label file
                outPred = np.zeros((n,c))
                for t in typeList:
                    typePred = np.zeros((n,c))
                    with open(experiment_path + 'lb' + str(lb).zfill(3) + '_' + str(t) + '_' + str(r).zfill(3) + '_train') as f:
                        trainPred = pk.load(f)
                        for i,k in enumerate(trainLabel.keys()):
                            typePred[k,:] = trainPred[i,:]

                    with open(experiment_path + 'lb' + str(lb).zfill(3) + '_' + str(t) + '_' + str(r).zfill(3) + '_test') as f:
                        testPred = pk.load(f)
                        for i,k in enumerate(testLabel.keys()):
                            typePred[k,:] = testPred[i,:]

                            # some potential improvement: set a threshold for random walk number to block
                            # 'unconfident' data points
                        '''
                            max = np.max(testPred[i,:])
                            if max > threshold[str(t)]:
                                label_file.write(str(t) + '\t' + str(k) + '\t' + v + '\n')
                        '''
                    # add meta-path probability to global probability
                    outPred += typePred * weight[str(t)]

                with open('/home/hejiang/results/' + 'GDIF' + 'lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_pred_rd_' + str(rd).zfill(3), 'w') as f:
                    pk.dump(outPred,f)

GCATDIFFCoTrainEnsemble()