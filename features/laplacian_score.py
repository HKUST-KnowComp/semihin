from sklearn.svm import LinearSVC
import numpy as np
import cPickle as pk
from graphgenerator import GraphGenerator
from sklearn.metrics.pairwise import cosine_similarity
import time
import scipy.sparse as sparse

''' Laplacian Score Feature selection for GCAT data set'''


''' Parameter
    kNeighbors : k neighbors of feature-graph
    t : tunable parameter of graph connection
    filtered_features : amount of output features
'''
kNeighbors = 10
t = 200.0
filtered_features = 100

# Load File
scope_name = 'GDIF'
with open('/home/data/corpora/HIN/dump/' + scope_name + '.dmp') as f:
    hin = pk.load(f)

X_word, newIds,e_newIds = GraphGenerator.getTFVectorX(hin,param={'word':True, 'entity':False, 'we_weight':1})
X_ent, newIds,e_newIds = GraphGenerator.getTFVectorX(hin,param={'word':False, 'entity':True, 'we_weight':1})
y = GraphGenerator.gety(hin)

# Generate cosine similarity graph
n = X_ent.shape[0]
m = X_ent.shape[1]
print m
cosX = cosine_similarity(X_word)
print cosX.shape
graph = np.zeros((n,n))
tic = time.time()
for i in range(n):
    for j in np.argpartition(cosX[i],-kNeighbors)[-kNeighbors:]:
        if j == i:
            continue
        #diff = (X_word[i, :] - X_word[j, :]).toarray().flatten()

        #dist = np.exp(np.dot(diff, diff) / t)
        graph[i, j] += cosX[i, j]
        graph[j, i] += cosX[j, i]

toc = time.time() - tic
print 'graph generation done in %f seconds.' % toc

D = sparse.diags([graph.sum(axis=0)], [0])
L = D - graph

print m
laplacian_score = np.zeros(m)
for i in range(m):
    f_tilde = X_ent[:, i] - (float(X_ent[:, i].transpose() * D * np.ones((n,1))) / np.sum(np.sum(D))) * np.ones((n,1))
    score = float(f_tilde.transpose() * L * f_tilde) / float(f_tilde.transpose() * D * f_tilde)
    laplacian_score[i] = score

# Select Feature with lowest Laplacian Score
strong_features = laplacian_score.argsort()[:filtered_features]
print strong_features
# Select Feature with the highest Laplacian Score
# Notice : Theoretically Wrong
#strong_features = laplacian_score.argsort()[-filtered_features:]


oldIds = hin.Ids
revNewIds = { v:k for k,v in e_newIds.items()}
revOldIds = { v:k for k,v in oldIds.items()}

print len(strong_features)
with open('/home/hejiang/results/' + scope_name + '_laplacian_features','w') as f:
    pk.dump(strong_features,f)

for idx in strong_features:
    print revOldIds[revNewIds[idx]]