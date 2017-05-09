import numpy as np
import pickle as pk
from labelpropagation import LabelPropagation

class SSLClassifier:
    def __init__(self, graph, label, classNames, LPParam, repeatTimes=10, trainNumbers=1,classCount={}):
        # init
        self.label = label
        self.graph = graph
        self.classNames = classNames
        self.LPParam = LPParam
        self.repeatTimes = repeatTimes
        self.classCount = classCount
        self.totalNum = 0.0
        for k,v in self.classCount.items():
            self.totalNum += v

        # number of training data for each class
        self.trainNumbers = trainNumbers

        # default
        self.trainLabel = {}
        self.testLabel = {}
        self.pred = {}
        self.results = []

        # definition
        self.n = self.graph.shape[0]
        self.c = len(classNames)
        self.y = np.zeros((self.n,self.c))

    def randomTrainTestSplit(self):
        self.trainLabel = {}
        self.testLabel = {}
        train_labels = set()
        for c in self.classNames:
            c_train = [k for k,v in self.label.iteritems() if v == c]
            c_label = np.random.permutation(len(c_train))[0:self.trainNumbers]
            for l in c_label:
                train_labels.add(c_train[l])

        self.y = np.zeros((self.n,self.c))
        for (k,v) in self.label.items():
            if k not in train_labels:
                self.testLabel[k] = v
            else:
                self.trainLabel[k] = v
                cid = self.classNames.index(v)
                self.y[k,cid] = self.totalNum / self.classCount[v]

    def makey(self):
        self.y = np.zeros((self.n,self.c))
        for (k,v) in self.trainLabel.items():
            cid = self.classNames.index(v)
            self.y[k, cid] = 1 #/ self.classCount[v]

    def runExperiment(self):

        # run LP algorithm
        lp = LabelPropagation(self.graph,self.y,self.LPParam)
        lp.walk()

        # convert pred probability to pred labels
        pred = {}
        trainProb = np.zeros((len(self.trainLabel),self.c))
        testProb = np.zeros((len(self.testLabel),self.c))
        for i,t in enumerate(self.testLabel):
            pred[t] = self.classNames[np.argmax(lp.PredictedProbs[t, :])]
            testProb[i, :] = lp.PredictedProbs[t, :]

        for i,t in enumerate(self.trainLabel):
            trainProb[i, :] = lp.PredictedProbs[t, :]

        # calculate error rate
        error_rate = self.precision(pred, self.testLabel)

        return error_rate,trainProb,testProb,pred

    '''
    def precision(self, pred, test):
        correct = 0.0
        wrong = 0.0
        for (d, l) in test.items():
            if pred[d] == l:
                correct += 1
            else:
                wrong += 1
        return correct / (correct + wrong)
    '''
    def precision(self, pred, test):
        correct = 0.0
        wrong = 0.0
        for (d, l) in test.items():
            if pred[d] == l:
                correct += 1.0 / self.classCount[l]
            else:
                wrong += 1.0 / self.classCount[l]
        return correct / (correct + wrong)

    def repeatedExperiment(self,savePathPrefix=None):
        self.results = []
        for r in range(self.repeatTimes):
            self.randomTrainTestSplit()
            if savePathPrefix:
                self.saveFixedExperiment(savePathPrefix + str(r).zfill(3))
            e, trainPred, testPred, pred = self.runExperiment()
            self.results.append(e)

    def saveFixedExperiment(self,pathPrefix):
        with open(pathPrefix + '_train','w') as f:
            pk.dump(self.trainLabel,f)
        with open(pathPrefix + '_test', 'w') as f:
            pk.dump(self.testLabel,f)

    def repeatedFixedExperiment(self,pathPrefix):
        self.results = []
        for r in range(self.repeatTimes):
            self.loadFixedExperiment(pathPrefix+str(r).zfill(3))
            self.makey()
            e, trainPred, testPred, pred = self.runExperiment()
            self.results.append(e)

    def loadFixedExperiment(self,pathPrefix):
        with open(pathPrefix + '_train') as f:
            self.trainLabel = pk.load(f)
        with open(pathPrefix + '_test') as f:
            self.testLabel = pk.load(f)

    # These two methods fits when feature extraction change the Id (vectorize, cosine-graph)
    def repeatedFixedExperimentwithNewIds(self,pathPrefix,newIds,saveProb=False,savePathPrefix='',savePred=False):
        self.results = []
        for r in range(self.repeatTimes):
            self.loadFixedExperimentwithNewIds(pathPrefix+str(r).zfill(3),newIds)
            self.makey()
            e,trainProb,testProb,pred = self.runExperiment()
            if saveProb:
                with open(savePathPrefix + '_' + str(r).zfill(3) + '_train', 'w') as f:
                    pk.dump(trainProb,f)
                with open(savePathPrefix + '_' + str(r).zfill(3) + '_test', 'w') as f:
                    pk.dump(testProb,f)
            if savePred:
                with open(savePathPrefix + '_' + str(r).zfill(3) + '_pred', 'w') as f:
                    pk.dump(pred, f)
            self.results.append(e)

    def repeatedFixedExpeimentwithInput(self,pathPrefix,newIds,saveProb=True,savePathPrefix='',savePred=False,inputPredPath=None):
        beta = 0.9
        self.results = []
        for r in range(self.repeatTimes):
            self.loadFixedExperimentwithNewIds(pathPrefix+str(r).zfill(3),newIds)

            with open(inputPredPath + '_' + str(r).zfill(3)) as f:
                inputPred = pk.load(f)
            # generate y
            self.MakeyAndPred(inputPred, beta)

            e,trainProb,testProb,pred = self.runExperiment()
            if saveProb:
                with open(savePathPrefix + '_' + str(r).zfill(3) + '_train', 'w') as f:
                    pk.dump(trainProb,f)
                with open(savePathPrefix + '_' + str(r).zfill(3) + '_test', 'w') as f:
                    pk.dump(testProb,f)
            if savePred:
                with open(savePathPrefix + '_' + str(r).zfill(3) + '_pred', 'w') as f:
                    pk.dump(pred, f)
            self.results.append(e)

    def loadPred(self,loadPath):
        with open(loadPath) as f:
            pred = pk.load(f)
        return pred

    def MakeyAndPred(self,pred, beta):

        self.y = np.zeros((self.n,self.c))
        for (k,v) in self.trainLabel.items():
            cid = self.classNames.index(v)
            self.y[k, cid] = 1 #/ self.classCount[v]
        m = len(self.trainLabel)+len(self.testLabel)
        self.y[0:m,:] = beta * self.y[0:m,:] + (1 - beta) * pred

    def loadFixedExperimentwithNewIds(self,pathPrefix,newIds):
        self.trainLabel = {}
        self.testLabel = {}
        with open(pathPrefix + '_train') as f:
            self.trainLabel = pk.load(f)
        with open(pathPrefix + '_test') as f:
            self.testLabel = pk.load(f)

    def stats(self):
        mean = np.mean(self.results)
        std = np.std(self.results)
        print str(mean) + '\t' + str(std)

    def get_mean(self):
        return np.mean(self.results)