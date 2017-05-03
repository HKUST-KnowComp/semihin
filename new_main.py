import cPickle as pk
import scipy.sparse as sparse
import numpy as np
import scipy as sp
from classifier import SSLClassifier
from datareader import DBLPReader
from datareader import DataReaderWang
from datareader import FB15KReader
from graphgenerator import GraphGenerator
from linkfilter import LinkFilter

def NG20DIFFClassificationExperiment():

    experiment_path = 'data/results/DIFF_'

    rd_diff = DataReaderWang(path='/home/data/corpora/HIN/20news.hin',scope=['rec.autos','comp.os.ms-windows.misc','sci.space'])
    rd_diff.readfile()
    hin = rd_diff.getHIN()
    print 'The 20NG-DIFF data has ' + str(len(hin.Links)) + ' links.'

    with open(experiment_path + 'Ids','w') as f:
        pk.dump(hin.Ids,f)

    graph_param = {'method': '', 'type':'', 'word':True, 'entity':True, 'word_tfidf':False, 'we_weight':0.112, 'external_links':False}
    graph = GraphGenerator.generateMetaGraph(hin, graph_param)

    #lp_param = {'alpha':0.01,'normalize_factor':10}
    lp_param = {'alpha':0.98, 'normalize_factor':5}
    # 3-class classification
    lp_candidate = [1,2,3,4,5,6,7,8,9,10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, hin.DocLabels, ['rec.autos','comp.os.ms-windows.misc','sci.space'], lp_param, repeatTimes=50, trainNumbers=lp)
        ssl.repeatedFixedExperiment(pathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')
        ssl.stats()

def NG20SIMClassificationExperiment():
    experiment_path = '/home/hejiang/results/SIM_'

    rd_sim = DataReaderWang(path='/home/data/corpora/HIN/20news.hin',scope=['comp.graphics','comp.sys.mac.hardware','comp.os.ms-windows.misc'])
    rd_sim.readfile()
    hin = rd_sim.getHIN()
    print 'The 20NG-SIM data has ' + str(len(hin.Links)) + ' links.'

    with open(experiment_path + 'Ids','w') as f:
        pk.dump(hin.Ids,f)

    graph_param = {'method': '', 'type':'', 'word':True, 'entity':True, 'word_tfidf':True, 'we_weight':0.112, 'external_links':False}
    graph = GraphGenerator.generateMetaGraph(hin, graph_param)

    lp_param = {}
    # 3-class classification
    lp_candidate = [1,2,3,4,5,6,7,8,9,10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, hin.DocLabels,
                ['comp.graphics','comp.sys.mac.hardware','comp.os.ms-windows.misc'],
                lp_param, repeatTimes=50, trainNumbers=lp)
        ssl.repeatedExperiment(savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')
        ssl.stats()

def GCATDIFFClassificationExperiment():
    experiment_path = '/home/hejiang/results/GDIF_'

    rd_diff = DataReaderWang(path='/home/data/corpora/HIN/GCAT_modified.hin',scope=['GSPO','GSCI','GHEA'])
    rd_diff.readfile()
    hin = rd_diff.getHIN()
    print 'The GCAT-DIFF data has ' + str(len(hin.Links)) + ' links.'

    graph_param = {'method': '', 'type':'', 'word':True, 'entity':True, 'word_tfidf':True, 'we_weight':0.112, 'external_links':False}
    graph = GraphGenerator.generateMetaGraph(hin, graph_param)

    lp_param = {}
    # 3-class classification
    lp_candidate = [1,2,3,4,5,6,7,8,9,10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, hin.DocLabels, ['GSPO','GSCI','GHEA'],lp_param, repeatTimes=50, trainNumbers=lp)
        ssl.repeatedExperiment(savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')
        ssl.stats()

def GCATSIMClassificationExperiment():
    experiment_path = '/home/hejiang/results/GSIM_'
    
    rd_sim = DataReaderWang(path='/home/data/corpora/HIN/GCAT_modified.hin', scope=['GSPO','GTOUR','GWELF'])
    rd_sim.readfile()
    hin = rd_sim.getHIN()
    print 'The GCAT-SIM data has ' + str(len(hin.Links)) + ' links.'

    graph_param = {'method': '', 'type': '', 'word': True, 'entity': True, 'word_tfidf': True, 'we_weight': 0.112,
                   'external_links': False}
    graph = GraphGenerator.generateMetaGraph(hin, graph_param)

    lp_param = {}
    # 3-class classification
    lp_candidate = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for lp in lp_candidate:
        ssl = SSLClassifier(graph, hin.DocLabels, ['GSPO','GTOUR','GWELF'],lp_param, repeatTimes=50, trainNumbers=lp)
        ssl.repeatedExperiment(savePathPrefix=experiment_path + 'lb' + str(lp).zfill(3) + '_')
        ssl.stats()

#GCATSIMClassificationExperiment()
#GCATDIFFClassificationExperiment()

#DBLPClassificationExperiment()
NG20DIFFClassificationExperiment()
#NG20SIMClassificationExperiment()
