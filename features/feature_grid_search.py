from sklearn.svm import LinearSVC
import numpy as np
import cPickle as pk
from graphgenerator import GraphGenerator
from sklearn.metrics.pairwise import cosine_similarity
import time
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
from sklearn import metrics
import numpy as np
import sys
import cPickle as pk
from graphgenerator import GraphGenerator
import os

kNeighbor_List = range(0, 1000, 100)
feature_list = range(0,50,10)


def grid_search(kNeighbor_list, feature_list, scope_name):
    # Load File
    with open('data/local/' + scope_name + '.dmp') as f:
        hin = pk.load(f)

    X_word, newIds, e_newIds = GraphGenerator.getTFIDFVectorX(hin,
                                                              param={'word': True, 'entity': False, 'we_weight': 1})
    X_ent, newIds, e_newIds = GraphGenerator.getTFIDFVectorX(hin, param={'word': False, 'entity': True, 'we_weight': 1})
    y = GraphGenerator.gety(hin)

#    print X_ent.sum()
#    print X_word.sum()

    best = 0.0
    best_k = 0
    best_f = 0
    best_features = None
    for k in kNeighbor_list:
        l_score = generate_laplacian_score(X_ent, X_word, k)
        for f in feature_list:
            filtered_f = filter_features(l_score, f)
            res = test_grid(X_ent, X_word, y, filtered_f, scope_name)
            if res > best:
                best = res
                best_k = k
                best_f = f
                best_features = filtered_f
            print 'Feature %d\tNeighbor %d\tRes %f\tBest %f' %(f, k, res, best)
    print 'Best K=%d Best F=%d' %(best_k, best_f)
    if not os.path.exists('data/local/laplacian'):
        os.makedirs('data/local/laplacian')
    with open('data/local/laplacian/' + scope_name + '.x', 'w') as f:
        X_feat = X_ent[:, best_features].toarray()
        X = np.concatenate([X_word.toarray(), X_feat], axis=1)
        pk.dump(X, f)



def generate_laplacian_score(X_ent, X_word, kNeighbors):
    # Generate cosine similarity graph
    n = X_ent.shape[0]
    m = X_ent.shape[1]
    cosX = cosine_similarity(X_word)
    graph = np.zeros((n, n))
    t = cosX.sum().sum() / n/n
    for i in range(n):
        for j in np.argpartition(cosX[i], -kNeighbors)[-kNeighbors:]:
            if j == i:
                continue
            # diff = (X_word[i, :] - X_word[j, :]).toarray().flatten()

            # dist = np.exp(np.dot(diff, diff) / t)
            graph[i, j] = cosX[i, j] #np.exp(- (1 - cosX[i, j]) / 0.03) #
            graph[j, i] = cosX[i, j] #np.exp(- (1 - cosX[i, j]) / 0.03) #

    D = sparse.diags([graph.sum(axis=0)], [0])
    L = D - graph

    laplacian_score = np.zeros(m)
    for i in range(m):
        f_tilde = X_ent[:, i] - (float(X_ent[:, i].transpose() * D * np.ones((n, 1))) / D.sum().sum()) * np.ones(
            (n, 1))
        score = float(f_tilde.transpose() * L * f_tilde) / float(f_tilde.transpose() * D * f_tilde + 1e-10)
        laplacian_score[i] = score


    return (laplacian_score)


def filter_features(laplacian_score, num_features):

    # Select Feature with lowest Laplacian Score
    strong_features = laplacian_score.argsort()[:num_features]
    # Select Feature with the highest Laplacian Score
    # Notice : Theoretically Wrong
    # strong_features = laplacian_score.argsort()[-filtered_features:]
    return strong_features


def test_grid(X_ent, X_word, y, features, scope_name):
    X_feat = X_ent[:, features].toarray()
    X = np.concatenate([X_word.toarray(), X_feat], axis=1)

    results = []
    for r in range(50):
        with open('data/local/split/' + scope_name + '/lb' + str(5).zfill(3) + '_' + str(r).zfill(
                3) + '_train') as f:
            trainLabel = pk.load(f)
        with open('data/local/split/' + scope_name + '/lb' + str(5).zfill(3) + '_' + str(r).zfill(
                3) + '_test') as f:
            testLabel = pk.load(f)

        XTrain = X[trainLabel.keys()]
        XTest = X[testLabel.keys()]
        yTrain = y[trainLabel.keys()]
        yTest = y[testLabel.keys()]

        # train
        clf = LinearSVC(C=0.01)
        # lf = LogisticRegression(C=0.01)
        # clf = MultinomialNB()
        clf.fit(XTrain, yTrain)

        # test
        pred = clf.predict(XTest)
        results.append(sum(pred == yTest) / float(yTest.shape[0]))
    return np.mean(results)
