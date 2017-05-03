import time
import cPickle as pk
import scipy.sparse as sparse
import scipy.sparse.linalg as slin
import scipy.linalg as lin
import numpy as np


class LabelPropagation:
    '''Label Propagation Algorithm

    Parameters
    ----------

    alpha : float, default=0.98

    method : str, default='ordinary'


    '''

    def __init__(self,graph,Y,param):
        self.graph = graph
        self.Y = Y
        self.param = param

        # default values
        if 'alpha' not in param:
            self.param['alpha'] = 0.98
        if 'method' not in param:
            self.param['method'] = 'ordinary'
        if 'normalize_factor' not in param:
            self.param['normalize_factor'] = 5

        # definitions

        # predicted classification probability
        self.PredictedProbs = {}
        # experiment running time
        self.ElapsedTime = 0

    def walk(self):
        if self.param['method'] == 'ordinary':
            self.ordinaryWalk()
        elif self.param['method'] == 'variant':
            self.variantWalk()

    def ordinaryWalk(self):
        tick = time.time()

        alpha = self.param['alpha']
        n = self.graph.shape[0]
        c = self.Y.shape[1]
        nf = self.param['normalize_factor']

        #self.graph = self.graph + 3000 * sparse.eye(n,n)

        Di = sparse.diags([np.sqrt(1 / (self.graph.sum(axis=0) + nf * np.ones(n))).getA1()], [0])
        S = Di.dot(self.graph.dot(Di))
        S_iter = (sparse.eye(n) - alpha * S).tocsc()

        F = np.zeros((n,c))
        for i in range(c):
            F[:, i], info = slin.cg(S_iter, self.Y[:, i], tol=1e-12)
        toc = time.time()

        #print np.where(F > 0)
        self.ElapsedTime = toc - tick
        self.PredictedProbs = F

    def variantWalk(self):
        tick = time.time()

        alpha = self.param['alpha']
        n = self.graph.shape[0]
        c = self.Y.shape[1]
        nf = self.param['normalize_factor']

        data = (self.graph.sum(axis=0) + nf * np.ones(n)).ravel()
        Di = sparse.spdiags(data,0,n,n).tocsc()
        S_iter = (Di - alpha * self.graph).tocsc()

        F = np.zeros((n, c))
        for i in range(c):
            F[:, i], info = slin.cg(S_iter, self.Y[:, i], tol=1e-10)

        toc = time.time()

        self.ElapsedTime = toc - tick
        self.PredictedProbs = F
