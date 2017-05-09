from datareader import DataReaderWang
from classifier import SSLClassifier
from graphgenerator import GraphGenerator
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
import scipy.sparse as sparse
import numpy as np
import cPickle as pk
import os
import time
import sys
from features.feature_grid_search import grid_search

DIFF = ['rec.autos', 'comp.os.ms-windows.misc', 'sci.space']
SIM = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc']
GSIM = ['GWEA','GDIS','GENV']
GDIF = ['GENT','GODD','GDEF']
SIM_count = {'comp.graphics':1000,'comp.sys.mac.hardware':1000,'comp.os.ms-windows.misc':1000}
DIFF_count = {'rec.autos':1000,'comp.os.ms-windows.misc':1000,'sci.space':1000}
GSIM_count = {'GWEA':1014,'GDIS':2083,'GENV':499}
GDIF_count = {'GENT':1062,'GODD':1096,'GDEF':542}

# meta-graph type list
NG20TypeList = [['organization'],['business'],['astronomy','spaceflight'], ['computer'],['sports','automotive'],['medicine','organization','biology','business'],['computer','astronomy','spaceflight','automotive']]
GDIFTypeList = [['military','sports'],['medicine'],[],['astronomy','spaceflight','automotive'],['location'],['organization'],['government'],['religion','computer','aviation']]
GSIMTypeList = [['military','sports'],['medicine'],[],['astronomy','spaceflight','automotive'],['location'],['location','organization'],['organization'],['government']]
GCATTypeList = [GSIMTypeList, GDIFTypeList]
# threshold and path weights
NG20_threshold = {'[\'sports\', \'automotive\']':1e-3,'[\'computer\']':1e-4,'[\'medicine\', \'organization\', \'biology\', \'business\']':1e-3,'[\'computer\', \'astronomy\', \'spaceflight\', \'automotive\']':-1,'[\'astronomy\', \'spaceflight\']':1e-3,'[\'organization\']':1e-3,'[\'business\']':1e-3}
GCAT_threshold = {'[\'military\', \'sports\']':1e-4,'[\'medicine\']':1e-3,'[]':1e-4,'[\'astronomy\', \'spaceflight\', \'automotive\']':0,'[\'location\']':1e-3,'[\'location\', \'organization\']':1e-2,'[\'organization\']':1e-2,'[\'government\']':1e-2, '[\'religion\', \'computer\', \'aviation\']':1e-6}
NG20_weight = {'[\'sports\', \'automotive\']':0.1,'[\'computer\']':0.5,'[\'medicine\', \'organization\', \'biology\', \'business\']':0.2,'[\'computer\', \'astronomy\', \'spaceflight\', \'automotive\']':0.5,'[\'astronomy\', \'spaceflight\']':0.1,'[\'organization\']':0.05,'[\'business\']': 0.05}
GSIM_weight = {'[\'military\', \'sports\']':0.01,'[\'medicine\']':0.1,'[]':0.02,'[\'astronomy\', \'spaceflight\', \'automotive\']':0.3,'[\'location\']':0.02,'[\'location\', \'organization\']':0.01,'[\'organization\']':0.01,'[\'government\']':0.01}
GDIF_weight = {'[\'military\', \'sports\']':0.01,'[\'medicine\']':0.1,'[]':0.02,'[\'astronomy\', \'spaceflight\', \'automotive\']':0.3,'[\'location\']':0.02,'[\'location\', \'organization\']':0.01,'[\'organization\']':0.01,'[\'government\']':0.01, '[\'religion\', \'computer\', \'aviation\']':0.1}
GCAT_weight = [GSIM_weight, GDIF_weight]
gcat_scope_names = ['GSIM', 'GDIF']
gcat_scopes = [GSIM, GDIF]
ng20_scope_names = ['SIM', 'DIFF']
ng20_scopes = [SIM, DIFF]
ng20_counts = [SIM_count, DIFF_count]
gcat_counts = [GSIM_count, GDIF_count]
lp_cand = [5]
result = np.zeros((4,11))


def run_dump_gcat(scope_name, scope):
    if not os.path.exists('data/local'):
        os.makedirs('data/local')

    reader = DataReaderWang(path='data/' + scope_name + '.hin', scope=scope, data_set='gcat')
    reader.readfile()
    hin = reader.getHIN()
    with open('data/local/' + scope_name + '.dmp', 'w') as f:
        pk.dump(hin, f)


def run_dump_20ng(scope_name, scope):
    if not os.path.exists('data/local'):
        os.makedirs('data/local')

    reader = DataReaderWang(path='data/' + scope_name + '.hin', scope=scope, data_set='20ng')
    reader.readfile()
    hin = reader.getHIN()
    with open('data/local/' + scope_name + '.dmp', 'w') as f:
        pk.dump(hin, f)
    with open('data/local/' + scope_name + '_Ids', 'w') as f:
        pk.dump(hin.Ids, f)
    with open('data/local/' + scope_name + '_DocIds', 'w') as f:
        pk.dump(hin.DocIds, f)


def run_laplacian_feature_search():
    kNeighbor_list = range(0, 1000, 100)
    feature_list = range(0, 50, 10)
    for name in gcat_scope_names:
        print name + ' feature search start'
        tic = time.time()
        grid_search(kNeighbor_list, feature_list, name)
        toc = time.time() - tic
        print name + ' feature search done in %.2f secs.' % toc


def dump_hin():
    print 'Read and preprocess raw text file of all data sets'
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        run_dump_20ng(scope_name, scope)

    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        run_dump_gcat(scope_name, scope)
    print ''


def generate_train_test_split():
    # generate random train-test split for 2 data set * 2 scopes
    repeat_times = 50


    lp_candidate = [5]

    # 20ng
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        count = ng20_counts[i]
        experiment_path = 'data/local/split/' + scope_name + '/'
        if not os.path.exists('data/local/split/' + scope_name):
            os.makedirs('data/local/split/' + scope_name)
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)
        tf_param = {'word': True, 'entity': True, 'we_weight': 0.1}
        lp_param = {'alpha': 0.99, 'normalization_factor': 0.01}
        graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin, kNeighbors=10, tf_param=tf_param)
        new_label = GraphGenerator.getNewLabels(hin)
        for lp in lp_candidate:
            ssl = SSLClassifier(graph, new_label, scope, lp_param, repeatTimes=repeat_times, trainNumbers=lp, classCount=count)
            ssl.repeatedExperiment(savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')

    # gcat
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        count = gcat_counts[i]
        if not os.path.exists('data/local/split/' + scope_name):
            os.makedirs('data/local/split/' + scope_name)
        experiment_path = 'data/local/split/' + scope_name + '/'
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)
        tf_param = {'word': True, 'entity': True, 'we_weight': 0.1}
        lp_param = {'alpha': 0.99, 'normalization_factor': 0.01}
        graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin, kNeighbors=10, tf_param=tf_param)
        new_label = GraphGenerator.getNewLabels(hin)
        for lp in lp_candidate:
            ssl = SSLClassifier(graph, new_label, scope, lp_param, repeatTimes=repeat_times, trainNumbers=lp, classCount=count)
            ssl.repeatedExperiment(savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')


def svm_experiment(scope_name, X, y):
    for lp in lp_cand:
        results = []
        for r in range(50):
            with open('data/local/split/' + scope_name + '/lb' + str(lp).zfill(3) + '_' + str(r).zfill(
                    3) + '_train') as f:
                trainLabel = pk.load(f)
            with open('data/local/split/' + scope_name + '/lb' + str(lp).zfill(3) + '_' + str(r).zfill(
                    3) + '_test') as f:
                testLabel = pk.load(f)

            XTrain = X[trainLabel.keys()]
            XTest = X[testLabel.keys()]
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
        return np.mean(results)


def nb_experiment(scope_name, X, y):
    for lp in lp_cand:
        results = []
        for r in range(50):
            with open('data/local/split/' + scope_name + '/lb' + str(lp).zfill(3) + '_' + str(r).zfill(
                    3) + '_train') as f:
                trainLabel = pk.load(f)
            with open('data/local/split/' + scope_name + '/lb' + str(lp).zfill(3) + '_' + str(r).zfill(
                    3) + '_test') as f:
                testLabel = pk.load(f)

            XTrain = X[trainLabel.keys()].toarray()
            XTest = X[testLabel.keys()].toarray()
            yTrain = y[trainLabel.keys()]
            yTest = y[testLabel.keys()]

            # train
            clf = GaussianNB()
            clf.fit(XTrain, yTrain)

            # test
            pred = clf.predict(XTest)
            results.append(sum(pred == yTest) / float(yTest.shape[0]))
        return np.mean(results)


def lp_experiment(scope, scope_name, count, graph, labels, newIds):
    experiment_path = 'data/local/split/' + scope_name + '/'
    lp_param = {'alpha': 0.99, 'normalization_factor': 0.01}
    #    lp_param = {'alpha':0.98, 'normalization_factor':5}
    lp = 5
    ssl = SSLClassifier(graph, labels, scope, lp_param, repeatTimes=50, trainNumbers=lp, classCount=count)
    ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_', newIds=newIds)
    return ssl.get_mean()


def lp_meta_experiment(scope, scope_name, type_list, threshold, weight, count):

    pred_path = 'data/local/lpmeta/' + scope_name + '/'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)
    split_path = 'data/local/split/' + scope_name + '/'

    with open('data/local/' + scope_name + '.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    c = len(scope)
    lb_cand = [5]
    repeats = 50

    # rounds for alternating optimization
    rounds = 2

    best_res = 0

    for rd in range(rounds):

        # step 1:
        # generate output of each meta-path
        for t in type_list:
            if not os.path.exists(pred_path + str(t)):
                os.makedirs(pred_path + str(t))
            graph, newIds = GraphGenerator.getMetaPathGraph(hin,tf_param,t)

            newLabel = GraphGenerator.getNewLabels(hin)
            lp_param = {'alpha':0.99,'normalization_factor':0.01}
        #    lp_param = {'alpha':0.98, 'normalization_factor':5}
            # 3-class classification

            lb = 5
            ssl = SSLClassifier(graph, newLabel, scope, lp_param, repeatTimes=repeats, trainNumbers=lb,classCount=count)
            if rd == 0:
                ssl.repeatedFixedExperimentwithNewIds(pathPrefix=split_path + 'lb' + str(lb).zfill(3) + '_',
                                                      newIds=newIds, saveProb=True,
                                                      savePathPrefix=pred_path + str(t) + '/lb' +str(lb).zfill(3))
            else:
                inputPredPath = 'data/local/lpmeta/' + scope_name + '/lb' + str(lb).zfill(3) + '_pred_rd_' + str(rd-1).zfill(3)
                ssl.repeatedFixedExpeimentwithInput(pathPrefix=split_path + 'lb' + str(lb).zfill(3) + '_',
                                                newIds=newIds,saveProb=True,savePathPrefix=pred_path + str(t) + '/lb' +
                                                str(lb).zfill(3),inputPredPath=inputPredPath)
            res = ssl.get_mean()
            if res > best_res:
                best_res = res

        # step 2:
        # propagate pseudo-label for other path
        for lb in lb_cand:
            results = []
            for r in range(repeats):
                with open(split_path + 'lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                        3) + '_train') as f:
                    trainLabel = pk.load(f)
                with open(split_path + 'lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                        3) + '_test') as f:
                    testLabel = pk.load(f)

                numTrain = len(trainLabel)
                numTest = len(testLabel)
                n = numTrain + numTest

                # write get-another-label label file
                outPred = np.zeros((n,c))
                for t in type_list:
                    typePred = np.zeros((n,c))
                    with open(pred_path + str(t) + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(3) + '_train') as f:
                        trainPred = pk.load(f)
                        for i,k in enumerate(trainLabel.keys()):
                            typePred[k,:] = trainPred[i,:]

                    with open(pred_path + str(t) + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(3) + '_test') as f:
                        testPred = pk.load(f)
                        for i,k in enumerate(testLabel.keys()):
                            typePred[k,:] = testPred[i,:]

                    # add meta-path probability to global probability
                    outPred += typePred * weight[str(t)]

                with open('data/local/lpmeta/' + scope_name + '/lb' + str(lb).zfill(3) + '_pred_rd_' + str(rd).zfill(3)
                                  + '_' + str(r).zfill(3), 'w') as f:
                    pk.dump(outPred,f)
    return best_res


def generate_meta_graph(scope, scope_name, type_list, count):
    split_path = 'data/local/split/' + scope_name + '/'
    pred_path = 'data/local/metagraph/' + scope_name + '/'
    if not os.path.exists('data/local/metagraph/' + scope_name + '/'):
        os.makedirs('data/local/metagraph/' + scope_name + '/')

    with open('data/local/' + scope_name + '.dmp') as f:
        hin = pk.load(f)
    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}

    for t in type_list:
        #print t
        X, newIds, entitynewIds = GraphGenerator.getTFVectorX(hin, tf_param, t)
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
        lp_candidate = [5]
        for lp in lp_candidate:
            ssl = SSLClassifier(graph, newLabel, scope, lp_param, repeatTimes=50, trainNumbers=lp, classCount=count)
            if not os.path.exists(pred_path + str(t) + '/'):
                os.makedirs(pred_path + str(t) + '/')
            ssl.repeatedFixedExperimentwithNewIds(pathPrefix=split_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds, saveProb=True,savePathPrefix=pred_path + str(t) + '/' + 'lb' + str(lp).zfill(3))
            #ssl.stats()


def ensemble_svm_experiment(scope, scope_name, type_list, threshold):
    # this section should be changed between different scopes
    experiment_path = 'data/local/metagraph/' + scope_name + '/'

    lb_cand = [5]
    repeats = 50

    with open('data/local/' + scope_name + '.dmp') as f:
        hin = pk.load(f)
    #X, newIds = GraphGenerator.getTFVectorX(hin, param={'word': True, 'entity': False, 'we_weight': 0.1})
    y = GraphGenerator.gety(hin)

    for lb in lb_cand:
        results = []
        for r in range(repeats):
            with open('data/local/split/' + scope_name + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_train') as f:
                trainLabel = pk.load(f)
            with open('data/local/split/' + scope_name + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_test') as f:
                testLabel = pk.load(f)

            yTrain = y[trainLabel.keys()]
            yTest = y[testLabel.keys()]

            numTrain = len(trainLabel)
            numTest = len(testLabel)
            XTrain = np.zeros((numTrain,0))
            XTest = np.zeros((numTest,0))

            for t in type_list:
                with open(experiment_path + str(t) + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(3) + '_train') as f:
                    trainPred = pk.load(f)
                with open(experiment_path + str(t) + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(3) + '_test') as f:
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
            clf = LinearSVC(C=0.1)
            clf.fit(XTrain, yTrain)

            # test
            pred = clf.predict(XTest)
            results.append(sum(pred == yTest) / float(yTest.shape[0]))
        return np.mean(results)


def ensemble_gal_experiment(scope, scope_name, type_list, threshold):
    # this section should be changed between different scopes
    pred_path = 'data/local/metagraph/' + scope_name + '/'

    lb_cand = [5]
    repeats = 50

    with open('data/local/' + scope_name + '.dmp') as f:
        hin = pk.load(f)
    #X, newIds = GraphGenerator.getTFVectorX(hin, param={'word': True, 'entity': False, 'we_weight': 0.1})
    y = GraphGenerator.gety(hin)

    if sys.platform == 'win32':
        command_file = open('galm.bat', 'a')
    else:
        command_file = open('galm.sh', 'a')

    for lb in lb_cand:
        results = []
        for r in range(repeats):
            with open('data/local/split/' + scope_name + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_train') as f:
                trainLabel = pk.load(f)
            with open('data/local/split/' + scope_name + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                    3) + '_test') as f:
                testLabel = pk.load(f)

            if not os.path.exists('data/local/gal/' + scope_name + '/'):
                os.makedirs('data/local/gal/' + scope_name + '/')
            label_file = open('data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + '_' +
                              str(r).zfill(3) + '_label.txt', 'w')
            gold_file = open('data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + '_' +
                              str(r).zfill(3) + '_gold.txt', 'w')
            eval_file = open('data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + '_' +
                             str(r).zfill(3) + '_eval.txt', 'w')

            # write get-another-label gold file
            for k,v in trainLabel.items():
                gold_file.write(str(k) + '\t' + v + '\n')

            # write get-another-label eval file
            for k,v in testLabel.items():
                eval_file.write(str(k) + '\t' + v + '\n')

            # write get-another-label label file
            for t in type_list:
                with open(pred_path + str(t) + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(3) + '_train') as f:
                    trainPred = pk.load(f)
                    for i,k in enumerate(trainLabel.keys()):
                        v = scope[np.argmax(trainPred[i,:])]
                        label_file.write(str(t) + '\t' + str(k) + '\t' + v + '\n')

                with open(pred_path + str(t) + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(3) + '_test') as f:
                    testPred = pk.load(f)
                    for i,k in enumerate(testLabel.keys()):
                        v = scope[np.argmax(testPred[i,:])]
                        max = np.max(testPred[i,:])
                        if max > threshold[str(t)]:
                            label_file.write(str(t) + '\t' + str(k) + '\t' + v + '\n')

            # run get-another-label batch
            if sys.platform == 'win32':
                command = r'call galm/bin/get-another-label.bat ' + \
                    '--categories galm/settings/' + scope_name + '_categories.txt ' + \
                    '--cost galm/settings/' + scope_name + '_costs.txt ' + \
                    '--gold data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + \
                    '_' + str(r).zfill(3) + '_gold.txt ' + \
                    '--input data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + \
                    '_' + str(r).zfill(3) + '_label.txt ' + \
                    '--eval data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + \
                    '_' + str(r).zfill(3) + '_eval.txt ' + \
                    '> data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + '_' + \
                    str(r).zfill(3) + '_result.txt'
            else:
                command = r'galm/bin/get-another-label.sh ' + \
                    '--categories /home/hejiang/results/gal/' + scope_name + '_categories.txt ' + \
                    '--cost /home/hejiang/results/gal/' + scope_name + '_costs.txt ' + \
                    '--gold data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + \
                    '_' + str(r).zfill(3) + '_gold.txt ' + \
                    '--input data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + \
                    '_' + str(r).zfill(3) + '_label.txt ' + \
                    '--eval data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + \
                    '_' + str(r).zfill(3) + '_eval.txt ' + \
                    '> data/local/gal/' + scope_name + '/lb' + str(lb).zfill(3) + '_' + \
                    str(r).zfill(3) + '_result.txt'

            command_file.write(command + '\r\n')


def ensemble_cotrain_experiment(scope, scope_name, type_list, threshold, weight, count):

    pred_path = 'data/local/cotrain/' + scope_name + '/'
    split_path = 'data/local/split/' + scope_name + '/'
    if not os.path.exists(pred_path):
        os.makedirs(pred_path)

    with open('data/local/' + scope_name + '.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    c = len(scope)
    lb_cand = [5]
    repeats = 50

    # rounds for alternating optimization
    rounds = 3

    best_res = 0
    best_t = ''
    for rd in range(rounds):
        round_best_res = 0
        round_best_t = ''

        # step 1:
        # generate output of each meta-path
        for t in type_list:
            if not os.path.exists(pred_path + str(t) + '/'):
                os.makedirs(pred_path + str(t) + '/')

            X, newIds, entityIds = GraphGenerator.getTFVectorX(hin,tf_param,t)
            n = X.shape[0]
            e = X.shape[1]
            X = X.toarray()
            graph = np.zeros((n+e,n+e))
            graph[0:n,n:n+e] = X
            graph[n:n+e,0:n] = X.transpose()
            graph = sparse.csc_matrix(graph)

            newLabel = GraphGenerator.getNewLabels(hin)
            lp_param = {'alpha':0.99,'normalization_factor':0.01}
            # lp_param = {'alpha':0.98, 'normalization_factor':5}

            lb = 5
            ssl = SSLClassifier(graph, newLabel, scope, lp_param, repeatTimes=repeats, trainNumbers=lb, classCount=count)
            if rd == 0:
                ssl.repeatedFixedExperimentwithNewIds(pathPrefix=split_path + 'lb' + str(lb).zfill(3) + '_',
                    newIds=newIds,saveProb=True,savePathPrefix=pred_path + str(t) + '/lb' + str(lb).zfill(3))
            else:
                inputPredPath = 'data/local/cotrain/' + scope_name + '/lb' + str(lb).zfill(3) + '_pred_rd_' + str(rd-1).zfill(3)
                ssl.repeatedFixedExpeimentwithInput(pathPrefix=split_path + 'lb' + str(lb).zfill(3) + '_',
                    newIds=newIds,saveProb=True,savePathPrefix=pred_path + 'lb' + str(lb).zfill(3) + '_' + str(t), inputPredPath=inputPredPath)
            res = ssl.get_mean()
            if res > best_res:
                best_res = res
                best_t = t
            if res > round_best_res:
                round_best_res = res
                round_best_t = t
        print 'Round %d\t%.4f\t%s' % (rd, round_best_res, str(round_best_t))

        # step 2:
        # propagate pseudo-label for other path
        for lb in lb_cand:
            results = []
            for r in range(repeats):
                with open('data/local/split/' + scope_name + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                        3) + '_train') as f:
                    trainLabel = pk.load(f)
                with open('data/local/split/' + scope_name + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(
                        3) + '_test') as f:
                    testLabel = pk.load(f)

                numTrain = len(trainLabel)
                numTest = len(testLabel)
                n = numTrain + numTest

                # write output probability
                outPred = np.zeros((n,c))
                for t in type_list:
                    typePred = np.zeros((n,c))
                    with open(pred_path + str(t) + '/lb' + str(lb).zfill(3) + '_'  + str(r).zfill(3) + '_train') as f:
                        trainPred = pk.load(f)
                        for i,k in enumerate(trainLabel.keys()):
                            typePred[k,:] = trainPred[i,:]

                    with open(pred_path + str(t) + '/lb' + str(lb).zfill(3) + '_' + str(r).zfill(3) + '_test') as f:
                        testPred = pk.load(f)
                        for i,k in enumerate(testLabel.keys()):
                            #typePred[k,:] = testPred[i,:]

                            # some potential improvement: set a threshold for random walk number to block
                            # 'unconfident' data points
                            max = np.max(testPred[i,:])
                            if max > threshold[str(t)]:
                                typePred[k, :] = testPred[i, :]
                    # add meta-path probability to global probability
                    outPred += typePred * weight[str(t)]

                with open('data/local/cotrain/' + scope_name + '/lb' + str(lb).zfill(3) + '_pred_rd_' + str(rd).zfill(3)
                                  + '_' + str(r).zfill(3), 'w') as f:
                    pk.dump(outPred,f)
    return best_res


def run_lp_meta():
    # 20NG
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        count = ng20_counts[i]
        print scope_name + ' lp meta'
        result[i, 5] = lp_meta_experiment(scope, scope_name, NG20TypeList, NG20_threshold, NG20_weight, count)

    # GCAT
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        count = gcat_counts[i]
        weight = GCAT_weight[i]
        type_list = GCATTypeList[i]
        print scope_name + ' lp meta'
        result[i+2, 5] = lp_meta_experiment(scope, scope_name, type_list, GCAT_threshold, weight, count)


def run_meta_graph_ensemble_svm():
    # 20NG
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        count = ng20_counts[i]
        print scope_name + ' svm ensemble'
        result[i, 8] = ensemble_svm_experiment(scope, scope_name, NG20TypeList, NG20_threshold)


    # GCAT
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        count = gcat_counts[i]
        type_list = GCATTypeList[i]
        print scope_name + ' svm ensemble'
        result[i+2, 8] = ensemble_svm_experiment(scope, scope_name, type_list, GCAT_threshold)


def run_meta_graph_ensemble_gal():
    if os.path.isfile('galm.bat'):
        os.remove('galm.bat')
    if os.path.isfile('galm.sh'):
        os.remove('galm.sh')
    # 20NG
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        count = ng20_counts[i]
        print scope_name + ' gal ensemble'
        ensemble_gal_experiment(scope, scope_name, NG20TypeList, NG20_threshold)

    # GCAT
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        count = gcat_counts[i]
        type_list = GCATTypeList[i]
        print scope_name + ' gal ensemble'
        ensemble_gal_experiment(scope, scope_name, type_list, GCAT_threshold)

    if sys.platform == 'win32':
        print 'Script generation done. Please run galm.bat to execute the get-another-label package.'
    else:
        print 'Script generation done. Please run galm.sh to execute the get-another-label package.'


def run_meta_graph_ensemble_cotrain():
    # 20NG
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        count = ng20_counts[i]
        print scope_name + ' cotrain ensemble'
        result[i, 10] = ensemble_cotrain_experiment(scope, scope_name, NG20TypeList, NG20_threshold, NG20_weight, count)

    # GCAT
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        count = gcat_counts[i]
        weight = GCAT_weight[i]
        type_list = GCATTypeList[i]
        print scope_name + ' cotrain ensemble'
        result[i+2, 10] = ensemble_cotrain_experiment(scope, scope_name, type_list, GCAT_threshold, weight, count)


def run_generate_meta_graph():
    # 20NG
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        count = ng20_counts[i]
        generate_meta_graph(scope, scope_name, NG20TypeList, count)

    # GCAT
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        count = gcat_counts[i]
        type_list = GCATTypeList[i]
        generate_meta_graph(scope, scope_name, type_list, count)


def run_svm():
    # 20NG
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)

        print scope_name + ' svm'
        tf_param = {'word': True, 'entity': False, 'we_weight': 0.1}
        X, doc_new_ids, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        y = GraphGenerator.gety(hin)
        result[i, 2] = svm_experiment(scope_name, X, y)


        print scope_name + ' svm+entity'
        tf_param = {'word': True, 'entity': True, 'we_weight': 0.1}
        X, doc_new_ids, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        y = GraphGenerator.gety(hin)
        result[i, 3] = svm_experiment(scope_name, X, y)

    # GCAT
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)

        print scope_name + ' svm'
        tf_param = {'word': True, 'entity': False, 'we_weight': 0.1}
        X, doc_new_ids, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        y = GraphGenerator.gety(hin)
        result[i+2, 2] = svm_experiment(scope_name, X, y)

        print scope_name + ' svm+entity'
        with open('data/local/laplacian/' + scope_name +'.x') as f:
            X = pk.load(f)
        y = GraphGenerator.gety(hin)
        result[i+2, 3] = svm_experiment(scope_name, X, y)


def run_nb():
    # 20NG
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)

        print scope_name + ' naive bayes'
        tf_param = {'word': True, 'entity': False, 'we_weight': 0.1}
        X, doc_new_ids, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        y = GraphGenerator.gety(hin)
        result[i, 0] = nb_experiment(scope_name, X, y)


        print scope_name + ' naive bayes+entity'
        tf_param = {'word': True, 'entity': True, 'we_weight': 0.1}
        X, doc_new_ids, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        y = GraphGenerator.gety(hin)
        result[i, 1] = nb_experiment(scope_name, X, y)

    # GCAT
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)

        print scope_name + ' naive bayes'
        tf_param = {'word': True, 'entity': False, 'we_weight': 0.1}
        X, doc_new_ids, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        y = GraphGenerator.gety(hin)
        result[i+2, 0] = nb_experiment(scope_name, X, y)

        print scope_name + ' naive bayes+entity'
        with open('data/local/laplacian/' + scope_name +'.x') as f:
            X = pk.load(f)
        y = GraphGenerator.gety(hin)
        result[i+2, 1] = svm_experiment(scope_name, X, y)


def run_lp():
    # 20NG
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        count = ng20_counts[i]
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)
        newLabels = GraphGenerator.getNewLabels(hin)

        tf_param = {'word': True, 'entity': True, 'we_weight': 0.112}
        graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin, 10, tf_param)
        print scope_name + ' lp+entity'
        result[i, 4] = lp_experiment(scope, scope_name, count, graph, newLabels, newIds)

    # GCAT
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        count = gcat_counts[i]
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)

        with open('data/local/laplacian/' + scope_name + '.x') as f:
            X = pk.load(f)
        graph = GraphGenerator.generateCosineNeighborGraphfromX(X)
        print scope_name + ' lp+entity'
        result[i+2, 4] = lp_experiment(scope, scope_name, count, graph, newLabels, newIds)

def semihin_experiment(scope, scope_name, count, X, newIds):
    experiment_path = 'data/local/split/' + scope_name + '/'

    with open('data/local/' + scope_name + '.dmp') as f:
        hin = pk.load(f)

    n = X.shape[0]
    e = X.shape[1]
    if not type(X) is np.ndarray:
        X = X.toarray()

    graph = np.zeros((n + e, n + e))
    graph[0:n, n:n + e] = X
    graph[n:n + e, 0:n] = X.transpose()
    graph = sparse.csc_matrix(graph)

    newLabel = GraphGenerator.getNewLabels(hin)
    lp_param = {'alpha': 0.98, 'normalization_factor': 5, 'method': 'variant'}

    #    lp_param = {'alpha':0.98, 'normalization_factor':5}
    # 3-class classification
    lp = 5
    ssl = SSLClassifier(graph, newLabel, scope, lp_param, repeatTimes=50,
                        trainNumbers=lp, classCount=count)
    ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_', newIds=newIds)
    return ssl.get_mean()


def run_semihin():
    # 20NG
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        count = ng20_counts[i]

        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)
        newLabels = GraphGenerator.getNewLabels(hin)

        tf_param = {'word': True, 'entity': False, 'we_weight': 0.112}
        X, newIds, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        print scope_name + ' semihin'
        result[i, 6] = semihin_experiment(scope, scope_name, count, X, newIds)

        tf_param = {'word': True, 'entity': True, 'we_weight': 0.112}
        X, newIds, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        print scope_name + ' semihin+entity'
        result[i, 7] = semihin_experiment(scope, scope_name, count, X, newIds)

    # GCAT
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        count = gcat_counts[i]
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)
        newLabels = GraphGenerator.getNewLabels(hin)

        tf_param = {'word': True, 'entity': False, 'we_weight': 0.112}
        X, newIds, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        print scope_name + ' semihin'
        result[i+2, 6] = semihin_experiment(scope, scope_name, count, X, newIds)

        with open('data/local/laplacian/' + scope_name + '.x') as f:
            X = pk.load(f)
        print scope_name + ' semihin+entity'
        result[i+2, 7] = semihin_experiment(scope, scope_name, count, X, newIds)


def dump_result():
    with open('result', 'w') as f:
        pk.dump(result, f)


def print_result():
    print 'This is result in Latex format. Please run galm.sh/galm.bat and gal_result.py to get result for get-another-label ensemble.'
    s_list = ['20NG-SIM', '20NG-DIFF', 'GCAT-SIM', 'GCAT-DIFF']
    print 'dataset & NB & NB+entity & SVM & SVM + entity & LP+entity & LP+metapath & Semihin & Semihin+entity & SVM ens & GAL ens & Cotrain ens'
    for j in range(4):
        s = s_list[j]
        for i in range(11):
            s += ' & $%.2f\\%%$' % (result[j, i] * 100)
        s += '  \\\\'
        print s


def run_all_experiments():
    run_semihin()
    run_lp()
    run_lp_meta()
    run_svm()
    run_nb()
    run_meta_graph_ensemble_svm()
    run_meta_graph_ensemble_gal()
    run_meta_graph_ensemble_cotrain()


def run_all():
    dump_hin()
    generate_train_test_split()
    run_laplacian_feature_search()
    run_generate_meta_graph()
    run_all_experiments()
    print_result()
    dump_result()

run_all()