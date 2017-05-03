import numpy as np
import scipy.sparse as sparse
from datareader import DataReaderWang
from graphgenerator import GraphGenerator
from classifier import SSLClassifier
import cPickle as pk
import time
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sparse

# Scopes
SIM = ['comp.graphics','comp.sys.mac.hardware','comp.os.ms-windows.misc']
DIFF = ['rec.autos','comp.os.ms-windows.misc','sci.space']
GSIM = ['GWEA','GDIS','GENV']
GDIF = ['GENT','GODD','GDEF']

def NG20DIFFLPClassification():
    print 'NG20 DIFF LP'
    experiment_path = '/home/hejiang/results/DIFF_'

    with open('/home/data/corpora/HIN/dump/DIFF.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin,kNeighbors=10,tf_param=tf_param)
    print 'word: ' + str(tf_param['word'])
    print 'entity: ' + str(tf_param['entity'])

    newLabel = GraphGenerator.getNewLabels(hin)
    lp_param = {'alpha':0.99,'normalization_factor':0.01}
#    lp_param = {'alpha':0.98, 'normalization_factor':5}
    # 3-class classification
    lp_candidate = [1,2,3,4,5,6,7,8,9,10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, newLabel, DIFF, lp_param, repeatTimes=50, trainNumbers=lp,classCount=DIFF_count)
        # first-time run or refresh label
        #ssl.repeatedExperiment(savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds)
        ssl.stats()

def NG20SIMLPClassification():
    print '20NG SIM LP'
    experiment_path = '/home/hejiang/results/SIM_'

    with open('/home/data/corpora/HIN/dump/SIM.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin,kNeighbors=10,tf_param=tf_param)
    print 'word: ' + str(tf_param['word'])
    print 'entity: ' + str(tf_param['entity'])

    newLabel = GraphGenerator.getNewLabels(hin)
    lp_param = {'alpha':0.99,'normalization_factor':0.01}
#    lp_param = {'alpha':0.98, 'normalization_factor':5}
    # 3-class classification
    lp_candidate = [1,2,3,4,5,6,7,8,9,10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, newLabel, SIM, lp_param, repeatTimes=50, trainNumbers=lp,classCount=SIM_count)
        # first-time run or refresh label
        #ssl.repeatedExperiment(savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds)
        ssl.stats()

def GCATSIMLPClassification():
    print 'GCAT SIM LP'
    experiment_path = '/home/hejiang/results/GSIM_'

    with open('/home/data/corpora/HIN/dump/GSIM.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word': True, 'entity': True, 'we_weight': 0.1}
    print 'word: ' + str(tf_param['word'])
    print 'entity: ' + str(tf_param['entity'])

    graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin, kNeighbors=10, tf_param=tf_param)

    newLabel = GraphGenerator.getNewLabels(hin)
    lp_param = {'alpha': 0.99, 'normalization_factor': 0.01}
    #    lp_param = {'alpha':0.98, 'normalization_factor':5}
    # 3-class classification
    lp_candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, newLabel, GSIM, lp_param,
                            repeatTimes=50, trainNumbers=lp,classCount=GSIM_count)
        # first-time run or refresh label
        #ssl.repeatedExperiment(savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds)
        ssl.stats()

def GCATDIFFLPClassification():
    print 'GCAT DIFF LP'
    experiment_path = '/home/hejiang/results/GDIF_'

    with open('/home/data/corpora/HIN/dump/GDIF.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word': True, 'entity': False, 'we_weight': 0.1}
    graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin, kNeighbors=10, tf_param=tf_param)
    graph, newIds = GraphGenerator.generateCosineNeighborGraph(hin, kNeighbors=10, tf_param=tf_param)
    print 'word: ' + str(tf_param['word'])
    print 'entity: ' + str(tf_param['entity'])

    newLabel = GraphGenerator.getNewLabels(hin)
    lp_param = {'alpha': 0.99, 'normalization_factor': 0.01}
    #    lp_param = {'alpha':0.98, 'normalization_factor':5}
    # 3-class classification
    lp_candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, newLabel, GDIF, lp_param,
                            repeatTimes=50, trainNumbers=lp,classCount=GDIF_count)
        # first-time run or refresh label
        #ssl.repeatedExperiment(savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds)
        ssl.stats()

def NG20DIFFFullHINClassification():
    experiment_path = '/home/hejiang/results/DIFF_'

    with open('/home/data/corpora/HIN/dump/DIFF.dmp') as f:
        hin = pk.load(f)

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    print 'word option: ' + str(tf_param['word'])
    print 'entity option: ' + str(tf_param['entity'])
    print 'word-entity weight: ' + str(tf_param['we_weight'])

    print 'The 20NG-DIFF data has ' + str(len(hin.Links)) + ' links.'

    X, newIds = GraphGenerator.getTFVectorX(hin,tf_param)
    n = X.shape[0]
    e = X.shape[1]
    X = X.toarray()
    graph = np.zeros((n+e,n+e))
    graph[0:n,n:n+e] = X
    graph[n:n+e,0:n] = X.transpose()
    graph = sparse.csc_matrix(graph)

    newLabel = GraphGenerator.getNewLabels(hin)
    lp_param = {'alpha':0.98,'normalization_factor':5,'method':'variant'}

#    lp_param = {'alpha':0.98, 'normalization_factor':5}
    # 3-class classification
    lp_candidate = [1,2,3,4,5,6,7,8,9,10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, newLabel, DIFF,lp_param, repeatTimes=50,
                            trainNumbers=lp,classCount=DIFF_count)
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds)
        ssl.stats()

def NG20SIMFullHINClassification():
    experiment_path = '/home/hejiang/results/SIM_'

    with open('/home/data/corpora/HIN/dump/SIM.dmp') as f:
        hin = pk.load(f)

    print 'The 20NG-SIM data has ' + str(len(hin.Links)) + ' links.'

    tf_param = {'word':True, 'entity':True, 'we_weight':0.1}
    print 'word option: ' + str(tf_param['word'])
    print 'entity option: ' + str(tf_param['entity'])
    print 'word-entity weight: ' + str(tf_param['we_weight'])

    X, newIds = GraphGenerator.getTFVectorX(hin,tf_param)
    n = X.shape[0]
    e = X.shape[1]
    X = X.toarray()
    graph = np.zeros((n+e,n+e))
    graph[0:n,n:n+e] = X
    graph[n:n+e,0:n] = X.transpose()
    graph = sparse.csc_matrix(graph)

    newLabel = GraphGenerator.getNewLabels(hin)
    lp_param = {'alpha':0.98,'normalization_factor':5,'method':'variant'}
#    lp_param = {'alpha':0.98, 'normalization_factor':5}
    # 3-class classification
    lp_candidate = [1,2,3,4,5,6,7,8,9,10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, newLabel, SIM,lp_param, repeatTimes=50,
                            trainNumbers=lp, classCount=SIM_count)
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds)
        ssl.stats()

def GCATSIMFullHINClassification():
    print 'GCAT SIM HIN'
    experiment_path = '/home/hejiang/results/GSIM_'

    with open('/home/data/corpora/HIN/dump/GSIM.dmp') as f:
        hin = pk.load(f)

    print 'The GCAT-SIM data has ' + str(len(hin.Links)) + ' links.'

    tf_param = {'word':True, 'entity':True, 'we_weight':1}
    print 'word option: ' + str(tf_param['word'])
    print 'entity option: ' + str(tf_param['entity'])
    print 'word-entity weight: ' + str(tf_param['we_weight'])

    X, newIds = GraphGenerator.getTFVectorX(hin,tf_param)
    n = X.shape[0]
    e = X.shape[1]
    X = X.toarray()
    graph = np.zeros((n+e,n+e))
    graph[0:n,n:n+e] = X
    graph[n:n+e,0:n] = X.transpose()
    graph = sparse.csc_matrix(graph)

    newLabel = GraphGenerator.getNewLabels(hin)
    lp_param = {'alpha':0.98,'normalization_factor':0.01,'method':'variant'}
    print 'alpha: ' + str(lp_param['alpha'])
    print 'normalization: ' + str(lp_param['normalization_factor'])
#    lp_param = {'alpha':0.98, 'normalization_factor':5}
    # 3-class classification
    lp_candidate = [1,2,3,4,5,6,7,8,9,10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, newLabel, GSIM, lp_param, repeatTimes=50,
                            trainNumbers=lp, classCount=GSIM_count)
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds)
        ssl.stats()

def GCATDIFFFullHINClassification():
    print 'GCAT DIFF HIN'
    experiment_path = '/home/hejiang/results/GDIF_'

    with open('/home/data/corpora/HIN/dump/GDIF.dmp') as f:
        hin = pk.load(f)

    print 'The GCAT-DIFF data has ' + str(len(hin.Links)) + ' links.'

    tf_param = {'word':True, 'entity':False, 'we_weight':1}
    print 'word option: ' + str(tf_param['word'])
    print 'entity option: ' + str(tf_param['entity'])
    print 'word-entity weight: ' + str(tf_param['we_weight'])

    X, newIds, entIds = GraphGenerator.getTFVectorX(hin,tf_param)
    with open('/home/hejiang/int/GDIF_300_Xword') as f:
        X_word = pk.load(f)
    with open('/home/hejiang/int/GDIF_100_X') as f:
        X_ent = pk.load(f)
    print X_word.shape
    print X_ent.shape
    X = np.concatenate([X_word.toarray(), X_ent.toarray()],axis=1)
    #X = X.toarray()

    n = X.shape[0]
    e = X.shape[1]
    graph = np.zeros((n+e,n+e))
    graph[0:n,n:n+e] = X
    graph[n:n+e,0:n] = X.transpose()
    graph = sparse.csc_matrix(graph)

    newLabel = GraphGenerator.getNewLabels(hin)
#    lp_param = {'alpha':0.99,'normalization_factor':0.01}
    lp_param = {'alpha':0.98, 'normalization_factor':0.01,'method':'ordinary'}
    print 'alpha: ' + str(lp_param['alpha'])
    print 'normalization: ' + str(lp_param['normalization_factor'])
    # 3-class classification
    lp_candidate = [1,2,3,4,5,6,7,8,9,10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, newLabel, GDIF, lp_param, repeatTimes=50,
                            trainNumbers=lp,classCount=GDIF_count)
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_',newIds=newIds)
        ssl.stats()

def EmbedLPClassification(scope,scope_name,count):
    kNeighbors = 40

    experiment_path = '/home/hejiang/results/' + scope_name + '_'

    with open('/home/data/corpora/HIN/dump/' + scope_name + '.dmp') as f:
        hin = pk.load(f)

    newLabel = GraphGenerator.getNewLabels(hin)

    with open('/home/hejiang/results/embed/' + scope_name + '_WEavg') as f:
        X_old = pk.load(f)
        X = np.array(X_old)
    with open('/home/hejiang/results/embed/' + scope_name + '_mapping') as f:
        mapping = pk.load(f)
    for k,v in mapping.items():
        X[k,:] = X_old[v,:]
    with open('/home/data/corpora/HIN/dump/' + scope_name + '_newIds') as f:
        newIds = pk.load(f)

    cosX = cosine_similarity(X)
    lp_param = {'alpha': 0.98, 'normalization_factor': 5, 'method': 'variant'}

    n = cosX.shape[0]
    graph = np.zeros((n, n))
    for i in range(n):
        for j in np.argpartition(-cosX[i], kNeighbors)[:kNeighbors]:
            if j == i:
                continue
            graph[i, j] += cosX[i, j]
            graph[j, i] += cosX[i, j]
            #graph[i, j] += 1
            #graph[j, i] += 1
    graph = sparse.csr_matrix(graph)

    #    lp_param = {'alpha':0.98, 'normalization_factor':5}
    # 3-class classification
    lp_candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, newLabel, scope, lp_param, repeatTimes=50,
                            trainNumbers=lp, classCount=count)
        ssl.repeatedFixedExperimentwithNewIds(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_', newIds=newIds)
        ssl.stats()
