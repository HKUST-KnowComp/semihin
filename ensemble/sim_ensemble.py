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

NG20TypeList = [['organization'],['business'],['astronomy','spaceflight'], ['computer'],['sports','automotive'],['medicine','organization','biology','business'],['computer','astronomy','spaceflight','automotive']]

SIM_threshold = {'[\'sports\', \'automotive\']':1e-3,
                  '[\'computer\']':1e-4,
                  '[\'medicine\', \'organization\', \'biology\', \'business\']':1e-3,
                  '[\'computer\', \'astronomy\', \'spaceflight\', \'automotive\']':-1,
                  '[\'astronomy\', \'spaceflight\']':1e-3,
                 '[\'organization\']':1e-3,
                 '[\'business\']':1e-3
                }


SIM_weight = {'[\'sports\', \'automotive\']':0.1,
                  '[\'computer\']':0.5,
                  '[\'medicine\', \'organization\', \'biology\', \'business\']':0.2,
                  '[\'computer\', \'astronomy\', \'spaceflight\', \'automotive\']':0.5,
                  '[\'astronomy\', \'spaceflight\']':0.1,
                  '[\'organization\']':0.05,
                  '[\'business\']': 0.05
              }

def NG20SIMMetaGraphs():
    experiment_path = 'data/local/metagraph/'

    with open('/home/data/corpora/HIN/dump/SIM.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    print 'word option: ' + str(tf_param['word'])
    print 'entity option: ' + str(tf_param['entity'])
    print 'word-entity weight: ' + str(tf_param['we_weight'])

    typeList = NG20TypeList
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
        lp_param = {'alpha':0.99,'normalization_factor':0.01}
    #    lp_param = {'alpha':0.98, 'normalization_factor':5}
        # 3-class classification
        lp_candidate = [1,2,3,4,5,6,7,8,9,10]
        for lp in lp_candidate:
            ssl = SSLClassifier(graph, newLabel, SIM, lp_param, repeatTimes=50, trainNumbers=lp)
            ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds,saveProb=True,savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_' + str(t))
            ssl.stats()

def NG20SIMSVMEnsemble():

    # this section should be changed between different scopes
    typeList = NG20TypeList
    experiment_path = '/home/hejiang/results/SIM_'
    scope_name = 'SIM'
    threshold = SIM_threshold


    lb_cand = range(1,11)
    repeats = 50

    with open('/home/data/corpora/HIN/dump/SIM.dmp') as f:
        hin = pk.load(f)
    #X, newIds = GraphGenerator.getTFVectorX(hin, param={'word': True, 'entity': False, 'we_weight': 0.1})
    y = GraphGenerator.gety(hin)

    for lb in lb_cand:
        results = []
        for r in range(repeats):
            with open('/home/hejiang/results/' + 'SIM' + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_train') as f:
                #trainOldLabel = pk.load(f)
                trainLabel = pk.load(f)
            with open('/home/hejiang/results/' + 'SIM' + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_test') as f:
                #testOldLabel = pk.load(f)
                testLabel = pk.load(f)

            '''
            trainLabel = []
            testLabel = []
            print trainOldLabel
            for l in trainOldLabel:
                trainLabel.append(newIds[l])
            for l in testOldLabel:
                testLabel.append(newIds[l])
            '''

            yTrain = y[trainLabel.keys()]
            yTest = y[testLabel.keys()]

            numTrain = len(trainLabel)
            numTest = len(testLabel)
            XTrain = np.zeros((numTrain,0))
            XTest = np.zeros((numTest,0))

            for t in typeList:
                with open(experiment_path + 'lb' + str(lb).zfill(3) + '_' + str(t) + '_' + str(r).zfill(3) + '_train') as f:
                    trainPred = pk.load(f)
                with open(experiment_path + 'lb' + str(lb).zfill(3) + '_' + str(t) + '_' + str(r).zfill(3) + '_test') as f:
                    testPred = pk.load(f)

                # threshold each meta-graph
                XTraint = np.zeros((numTrain,3))
                XTestt = np.zeros((numTest,3))
                for i,k in enumerate(trainLabel.items()):
                    v = np.argmax(trainPred[i, :])
                    max = np.max(trainPred[i, :])
                    if max > threshold[str(t)]:
                        # zero-one prediction
                        XTraint[i, v] = 1
                        # raw prediction
                        #XTraint[i, :] = trainPred[i, :]

                for i,k in enumerate(testLabel.items()):
                    v = np.argmax(testPred[i, :])
                    max = np.max(testPred[i, :])
                    if max > threshold[str(t)]:
                        # zero-one prediction
                        XTestt[i, v] = 1
                        # raw prediction
                        #XTestt[i, :] = testPred[i, :]


                XTrain = np.concatenate((XTrain,XTraint),axis=1)
                XTest = np.concatenate((XTest,XTestt),axis=1)

                # use raw input
                #XTrain = np.concatenate((XTrain,trainPred),axis=1)
                #XTest = np.concatenate((XTest,testPred),axis=1)

            # train
            #clf = LinearSVC(C=0.001)
            #clf = DecisionTreeClassifier()
            clf = LogisticRegression(C=0.01)
            clf.fit(XTrain, yTrain)

            # test
            pred = clf.predict(XTest)
            results.append(sum(pred == yTest) / float(yTest.shape[0]))
        print str(np.mean(results)) + '\t' + str(np.std(results))

NG20SIMSVMEnsemble()

def NG20SIMGALEnsemble():
    # this section should be changed between different scopes
    typeList = NG20TypeList
    experiment_path = '/home/hejiang/results/SIM_'
    scope_name = 'SIM'
    scope = SIM

    threshold = SIM_threshold

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

#NG20SIMMetaPaths()
#NG20SIMGALEnsemble()

def NG20SIMCoTrainEnsemble():
    experiment_path = '/home/hejiang/results/SIM_'

    with open('/home/data/corpora/HIN/dump/SIM.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    scope = SIM
    c = len(scope)
    threshold = SIM_threshold
    weight = SIM_weight
    lb_cand = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #lb_cand = [10]
    repeats = 50

    # rounds for alternating optimizatrion
    rounds = 20

    typeList = NG20TypeList


    for rd in range(rounds):

        # step 1:
        # generate output of each meta-path
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
            lp_param = {'alpha':0.99,'normalization_factor':0.01}
        #    lp_param = {'alpha':0.98, 'normalization_factor':5}
            # 3-class classification

            for lb in lb_cand:
                ssl = SSLClassifier(graph, newLabel, scope, lp_param, repeatTimes=repeats, trainNumbers=lb)
                if rd == 0:
                    ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lb).zfill(3) + '_',
                                                    newIds=newIds,saveProb=True,savePathPrefix=experiment_path + 'lb' +
                                                    str(lb).zfill(3) + '_' + str(t))
                else:
                    with open('/home/hejiang/results/' + 'SIM' + 'lb' + str(lb).zfill(3) + '_' + str(r).zfill(
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
                with open('/home/hejiang/results/' + 'SIM' + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                        3) + '_train') as f:
                    trainLabel = pk.load(f)
                with open('/home/hejiang/results/' + 'SIM' + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
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

                with open('/home/hejiang/results/' + 'SIM' + 'lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_pred_rd_' + str(rd).zfill(3), 'w') as f:
                    pk.dump(outPred,f)

def NG20SIMSDMEnsemble():
    experiment_path = '/home/hejiang/results/SIM_'

    with open('/home/data/corpora/HIN/dump/SIM.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    scope = DIFF
    c = len(scope)
    threshold = SIM_threshold
    weight = SIM_weight
    lb_cand = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #lb_cand =  [10]
    repeats = 50

    # rounds for alternating optimizatrion
    rounds = 20

    typeList = NG20TypeList


    for rd in range(rounds):

        # step 1:
        # generate output of each meta-path
        for t in typeList:
            print t
            graph, newIds = GraphGenerator.getMetaPathGraph(hin,tf_param,t)

            newLabel = GraphGenerator.getNewLabels(hin)
            lp_param = {'alpha':0.99,'normalization_factor':0.01}
        #    lp_param = {'alpha':0.98, 'normalization_factor':5}
            # 3-class classification

            for lb in lb_cand:
                ssl = SSLClassifier(graph, newLabel, SIM, lp_param, repeatTimes=repeats, trainNumbers=lb,classCount=SIM_count)
                if rd == 0:
                    ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lb).zfill(3) + '_',
                                                    newIds=newIds,saveProb=True,savePathPrefix=experiment_path + 'lb' +
                                                    str(lb).zfill(3) + '_' + str(t))
                else:
                    with open('/home/hejiang/results/' + 'SIM' + 'lb' + str(lb).zfill(3) + '_' + str(r).zfill(
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
                with open('/home/hejiang/results/' + 'SIM' + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                        3) + '_train') as f:
                    trainLabel = pk.load(f)
                with open('/home/hejiang/results/' + 'SIM' + '_lb' + str(lb).zfill(3) + '_' + str(r).zfill(
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

                with open('/home/hejiang/results/' + 'SIM' + 'lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_pred_rd_' + str(rd).zfill(3), 'w') as f:
                    pk.dump(outPred,f)

NG20SIMSDMEnsemble()
