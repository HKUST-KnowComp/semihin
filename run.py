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
from features.feature_grid_search import grid_search

DIFF = ['rec.autos', 'comp.os.ms-windows.misc', 'sci.space']
SIM = ['comp.graphics', 'comp.sys.mac.hardware', 'comp.os.ms-windows.misc']
GSIM = ['GWEA','GDIS','GENV']
GDIF = ['GENT','GODD','GDEF']
SIM_count = {'comp.graphics':1000,'comp.sys.mac.hardware':1000,'comp.os.ms-windows.misc':1000}
DIFF_count = {'rec.autos':1000,'comp.os.ms-windows.misc':1000,'sci.space':1000}
GSIM_count = {'GWEA':1014,'GDIS':2083,'GENV':499}
GDIF_count = {'GENT':1062,'GODD':1096,'GDEF':542}
gcat_scope_names = ['GSIM', 'GDIF']
gcat_scopes = [GSIM, GDIF]
ng20_scope_names = ['SIM', 'DIFF']
ng20_scopes = [SIM, DIFF]
ng20_counts = [SIM_count, DIFF_count]
gcat_counts = [GSIM_count, GDIF_count]
lp_cand = [5]

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
        print str(np.mean(results)) + '\t' + str(np.std(results))

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

            XTrain = X[trainLabel.keys()]
            XTest = X[testLabel.keys()]
            yTrain = y[trainLabel.keys()]
            yTest = y[testLabel.keys()]

            # train
            clf = GaussianNB()
            clf.fit(XTrain, yTrain)

            # test
            pred = clf.predict(XTest)
            results.append(sum(pred == yTest) / float(yTest.shape[0]))
        print str(np.mean(results)) + '\t' + str(np.std(results))

def lp_experiment(scope, scope_name, count, graph, labels, newIds):
    experiment_path = 'data/local/split/' + scope_name + '/'
#    print 'word: ' + str(tf_param['word'])
#    print 'entity: ' + str(tf_param['entity'])

    lp_param = {'alpha': 0.99, 'normalization_factor': 0.01}
    #    lp_param = {'alpha':0.98, 'normalization_factor':5}
    # 3-class classification
    lp_candidate = [5]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, labels, scope, lp_param, repeatTimes=50, trainNumbers=lp, classCount=count)
        # first-time run or refresh label
        # ssl.repeatedExperiment(savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_', newIds=newIds)
        ssl.stats()


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
        svm_experiment(scope_name, X, y)


        print scope_name + ' svm+entity'
        tf_param = {'word': True, 'entity': True, 'we_weight': 0.1}
        X, doc_new_ids, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        y = GraphGenerator.gety(hin)
        svm_experiment(scope_name, X, y)

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
        svm_experiment(scope_name, X, y)

        print scope_name + ' svm+entity'
        with open('data/local/laplacian/' + scope_name +'.x') as f:
            X = pk.load(f)
        y = GraphGenerator.gety(hin)
        svm_experiment(scope_name, X, y)

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
        svm_experiment(scope_name, X, y)


        print scope_name + ' naive bayes+entity'
        tf_param = {'word': True, 'entity': True, 'we_weight': 0.1}
        X, doc_new_ids, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        y = GraphGenerator.gety(hin)
        svm_experiment(scope_name, X, y)

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
        svm_experiment(scope_name, X, y)

        print scope_name + ' naive bayes+entity'
        with open('data/local/laplacian/' + scope_name +'.x') as f:
            X = pk.load(f)
        y = GraphGenerator.gety(hin)
        svm_experiment(scope_name, X, y)

def run_lp():
    # 20NG
    for i in range(2):
        scope_name = ng20_scope_names[i]
        scope = ng20_scopes[i]
        count = ng20_counts[i]
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)
        newLabels = GraphGenerator.getNewLabels(hin)

        tf_param = {'word': True, 'entity': False, 'we_weight': 0.112}
        graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin, 10, tf_param)
        print scope_name + ' lp'
        lp_experiment(scope, scope_name, count, graph, newLabels, newIds)

        tf_param = {'word': True, 'entity': True, 'we_weight': 0.112}
        graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin, 10, tf_param)
        print scope_name + ' lp+entity'
        lp_experiment(scope, scope_name, count, graph, newLabels, newIds)

    # GCAT
    for i in range(2):
        scope_name = gcat_scope_names[i]
        scope = gcat_scopes[i]
        count = gcat_counts[i]
        with open('data/local/' + scope_name + '.dmp') as f:
            hin = pk.load(f)

        tf_param = {'word': True, 'entity': False, 'we_weight': 0.112}
        graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin, 10, tf_param)
        print scope_name + ' lp'
        lp_experiment(scope, scope_name, count, graph, newLabels, newIds)

        with open('data/local/laplacian/' + scope_name + '.x') as f:
            X = pk.load(f)
        graph = GraphGenerator.generateCosineNeighborGraphfromX(X)
        print scope_name + ' lp+entity'
        lp_experiment(scope, scope_name, count, graph, newLabels, newIds)

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
    lp_candidate = [5]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, newLabel, scope, lp_param, repeatTimes=50,
                            trainNumbers=lp, classCount=count)
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_', newIds=newIds)
        ssl.stats()

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
        semihin_experiment(scope, scope_name, count, X, newIds)

        tf_param = {'word': True, 'entity': True, 'we_weight': 0.112}
        X, newIds, entity_new_ids = GraphGenerator.getTFVectorX(hin, param=tf_param, entity_types=None)
        print scope_name + ' semihin+entity'
        semihin_experiment(scope, scope_name, count, X, newIds)

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
        semihin_experiment(scope, scope_name, count, X, newIds)

        with open('data/local/laplacian/' + scope_name + '.x') as f:
            X = pk.load(f)
        print scope_name + ' semihin+entity'
        semihin_experiment(scope, scope_name, count, X, newIds)


def run_all_experiments():
    run_semihin()
    run_lp()
    run_svm()
    run_nb()

def run_all():
    dump_hin()
    generate_train_test_split()
    run_laplacian_feature_search()
    run_all_experiments()

run_all_experiments()


def all_in_one_run():
    dump_hin()
    generate_train_test_split()