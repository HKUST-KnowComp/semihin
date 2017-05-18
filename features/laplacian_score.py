from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sparse


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

