import scipy.sparse as sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import pickle as pk

class GraphGenerator:

    @staticmethod
    def emptyGraph(hin):
        n = len(hin.Ids)
        return sparse.diags(np.zeros(n))

    @staticmethod
    def addLinks(graph,Links,weight=1):
        for link in Links:
            head, relation, tail = link
            graph[head, tail] += weight
            graph[tail, head] += weight
        return graph

    def addWeightedLinks(graph,Links,exteralWeight=1):
        for link in Links:
            head, relation, tail, weight = link
            graph[head, tail] += weight * exteralWeight
            graph[tail, head] += weight * exteralWeight

    # generate Meta-Graph
    @staticmethod
    def generateMetaGraph(hin, param={'method': '', 'type': '', 'word':True, 'entity':True, 'word_tfidf':False, 'we_weight':0.112, 'external_links':False}, externalLinks=[]):
        n = len(hin.Ids)
        D = float(len(hin.DocIds))
        graph = sparse.lil_matrix((n,n))

        # source of the links
        if param['external_links']:
            links = externalLinks
        else:
            links = hin.Links

        # generate hin
        for link in links:
            head = link[0]
            relation = link[1]
            tail = link[2]

            head_type = hin.NodeTypes[head]
            tail_type = hin.NodeTypes[tail]
            # dwd links
            if param['word'] and tail_type == hin.TypeIds['word']:
                if param['word_tfidf']:
                    graph[head, tail] += D / hin.WordDFs[tail] * param['we_weight']
                    graph[tail, head] += D / hin.WordDFs[tail] * param['we_weight']
                else:
                    graph[head, tail] = param['we_weight']
                    graph[tail, head] = param['we_weight']
            # document-entity and entity-entity link
            if param['entity'] and tail_type != hin.TypeIds['word']:
                    graph[head, tail] = 1
                    graph[tail, head] = 1

        return graph

    @staticmethod
    def generateRelationTypeBasedGraph(hin, param={'method': '', 'type': '', 'word':True, 'entity':True, 'word_tfidf':False, 'we_weight':0.112, 'external_links':False}):
        n = len(hin.Ids)
        D = float(len(hin.DocIds))
        graph = sparse.lil_matrix((n, n))
        for link in hin.Links:
            head = link[0]
            relation = link[1]
            tail = link[2]

            head_type = hin.NodeTypes[head]
            tail_type = hin.NodeTypes[tail]
            # dwd links
            if param['word'] and tail_type == hin.TypeIds['word']:
                if param['word_tfidf']:
                    graph[head, tail] += D / hin.WordDFs[tail] * param['we_weight']
                    graph[tail, head] += D / hin.WordDFs[tail] * param['we_weight']
                else:
                    graph[head, tail] += param['we_weight']
                    graph[tail, head] += param['we_weight']
            # document-entity and entity-entity link
            if param['entity'] and tail_type != hin.TypeIds['word']:
                graph[head, tail] += 1
                graph[tail, head] += 1

        return graph

    # generate TF vector for sklearn
    # output all X for further processing
    @staticmethod
    def getTFVectorX(hin, param={'word':True, 'entity':True, 'we_weight':0.112},entity_types=None):
        # get new IDs for documents
        doc_new_ids = {}
        for doc in hin.DocIds:
            doc_old_id = hin.DocIds[doc]
            doc_new_ids[doc_old_id] = len(doc_new_ids)
        # get new IDs for words and entities
        entity_new_ids = {}

        word_flag = param['word']
        entity_flag = param['entity']
        type_flag = True if entity_types else False
        selected_types = []
        if type_flag:
            for t in hin.TypeIds:
                for entity_type in entity_types:
                    if t.startswith(entity_type):
                        selected_types.append(hin.TypeIds[t])
                        break

        entity_count = {}
        for link in hin.Links:
            head = link[0]
            tail = link[2]
            entity_count[head] = entity_count.get(head, 0) + 1
            entity_count[tail] = entity_count.get(tail, 0) + 1

        entity_count_low_threshold = 10
        entity_count_high_threshold = 10000

        for entity in hin.Ids:
            entity_old_id = hin.Ids[entity]
            node_type = hin.NodeTypes[entity_old_id]
            if node_type != hin.TypeIds['document']:
                if word_flag and node_type == hin.TypeIds['word']:
                    entity_new_ids[entity_old_id] = len(entity_new_ids)
                elif type_flag and node_type in selected_types:
                    if entity_count[entity_old_id] >= entity_count_low_threshold and \
                        entity_count[entity_old_id] <= entity_count_high_threshold:
                        entity_new_ids[entity_old_id] = len(entity_new_ids)
                elif not type_flag and entity_flag and node_type != hin.TypeIds['word']:
                    if entity_count[entity_old_id] >= entity_count_low_threshold and \
                        entity_count[entity_old_id] <= entity_count_high_threshold:
                        entity_new_ids[entity_old_id] = len(entity_new_ids)

        n = len(doc_new_ids)
        m = len(entity_new_ids)
        X = sparse.lil_matrix((n,m))

        for link in hin.Links:
            head = link[0]
            relation = link[1]
            tail = link[2]
            # not a document-entity or document-word link
            if hin.NodeTypes[head] != hin.TypeIds['document']:
                continue
            elif param['word'] and hin.NodeTypes[tail] == hin.TypeIds['word']:
                X[doc_new_ids[head], entity_new_ids[tail]] += param['we_weight']

            elif param['entity']:
                if type_flag and hin.NodeTypes[tail] in selected_types and tail in entity_new_ids:
                    X[doc_new_ids[head], entity_new_ids[tail]] += 1
                elif not type_flag and  hin.NodeTypes[tail] != hin.TypeIds['word'] and tail in entity_new_ids:
                    X[doc_new_ids[head], entity_new_ids[tail]] += 1

        return X.tocsc(), doc_new_ids, entity_new_ids


    # generate TFIDF vector for sklearn
    # output X
    @staticmethod
    def getTFIDFVectorX(hin, param={'word':True, 'entity':True,'we_weight':0.112},entity_types=None):
        # get new IDs for documents
        doc_new_ids = {}
        for doc in hin.DocIds:
            doc_old_id = hin.DocIds[doc]
            doc_new_ids[doc_old_id] = len(doc_new_ids)
        # get new IDs for words and entities
        entity_new_ids = {}

        word_flag = param['word']
        entity_flag = param['entity']
        type_flag = True if entity_types else False
        selected_types = []
        if type_flag:
            for t in hin.TypeIds:
                for entity_type in entity_types:
                    if t.startswith(entity_type):
                        selected_types.append(hin.TypeIds[t])
                        break

        entity_count = {}
        for link in hin.Links:
            head = link[0]
            tail = link[2]
            entity_count[head] = entity_count.get(head, 0) + 1
            entity_count[tail] = entity_count.get(tail, 0) + 1

        entity_count_low_threshold = 10
        entity_count_high_threshold = 10000

        for entity in hin.Ids:
            entity_old_id = hin.Ids[entity]
            node_type = hin.NodeTypes[entity_old_id]
            if node_type != hin.TypeIds['document']:
                if word_flag and node_type == hin.TypeIds['word']:
                    entity_new_ids[entity_old_id] = len(entity_new_ids)
                elif type_flag and node_type in selected_types:
                    if entity_count[entity_old_id] >= entity_count_low_threshold and \
                        entity_count[entity_old_id] <= entity_count_high_threshold:
                        entity_new_ids[entity_old_id] = len(entity_new_ids)
                elif not type_flag and entity_flag and node_type != hin.TypeIds['word']:
                    if entity_count[entity_old_id] >= entity_count_low_threshold and \
                        entity_count[entity_old_id] <= entity_count_high_threshold:
                        entity_new_ids[entity_old_id] = len(entity_new_ids)

        n = len(doc_new_ids)
        m = len(entity_new_ids)
        X = sparse.lil_matrix((n,m))
        print n,m

        we_weight = param['we_weight']
        for link in hin.Links:
            head = link[0]
            relation = link[1]
            tail = link[2]
            # not a document-entity or document-word link
            if hin.NodeTypes[head] != hin.TypeIds['document']:
                continue
            elif param['word'] and hin.NodeTypes[tail] == hin.TypeIds['word']:
                X[doc_new_ids[head], entity_new_ids[tail]] += 1.0 * we_weight

            elif param['entity']:
                if type_flag and hin.NodeTypes[tail] in selected_types and tail in entity_new_ids:
                    X[doc_new_ids[head], entity_new_ids[tail]] += 1.0
                elif not type_flag and  hin.NodeTypes[tail] != hin.TypeIds['word'] and tail in entity_new_ids:
                    X[doc_new_ids[head], entity_new_ids[tail]] += 1.0

            elif param['entity'] and hin.NodeTypes[tail] != hin.TypeIds['word']:
                X[doc_new_ids[head], entity_new_ids[tail]] += 1.0

        X.tocsc()
        jIndex = range(m)
        for j in jIndex:
            nnzXj = X[:,j].nonzero()[0]
            df = len(nnzXj)
            idf = np.log(float(n) / float(df))
            for i in nnzXj:
                if X[i, j] > 0:
                    X[i, j] = (1 + np.log(X[i, j])) * idf

        return X.tocsc(), doc_new_ids, entity_new_ids

    @staticmethod
    def getY(hin):
        # get new IDs for documents
        label_types = {}
        for doc in hin.DocIds:
            doc_label = hin.DocLabels[doc]
            if doc_label not in label_types:
                label_types[doc_label] = len(label_types)

        Y = np.array((len(hin.DocLabels),len(label_types)))
        for doc in hin.DocLabels:
            Y[hin.DocIds[doc], label_types[hin.DocLabels[doc]]] = 1

        return Y

    @staticmethod
    def gety(hin):
        # get new IDs for documents
        doc_new_ids = {}
        label_types = {}
        for doc in hin.DocIds:
            doc_old_id = hin.DocIds[doc]
            doc_new_ids[doc_old_id] = len(doc_new_ids)
            doc_label = hin.DocLabels[doc_old_id]
            if doc_label not in label_types:
                label_types[doc_label] = len(label_types)

        y = np.zeros(len(doc_new_ids))
        for doc_old_id in hin.DocLabels:
            y[doc_new_ids[doc_old_id]] = label_types[hin.DocLabels[doc_old_id]]

        return y

    @staticmethod
    def getNewLabels(hin):
        doc_new_ids = {}
        docNewLabel = {}
        for doc in hin.DocIds:
            doc_old_id = hin.DocIds[doc]
            doc_new_ids[doc_old_id] = len(doc_new_ids)
            new_id = doc_new_ids[doc_old_id]
            docNewLabel[new_id] = hin.DocLabels[doc_old_id]
        return docNewLabel

    @staticmethod
    def generateCosineNeighborGraph(hin,kNeighbors=10,tf_param={'word':True, 'entity':False, 'we_weight':1}):
        X, newIds, entIds = GraphGenerator.getTFVectorX(hin,param=tf_param)
        cosX = cosine_similarity(X)
        #return sparse.csc_matrix(X.dot(X.transpose())),newIds
        n = cosX.shape[0]
        graph = np.zeros((n,n))
        tic = time.time()
        for i in range(n):
            for j in np.argpartition(-cosX[i],kNeighbors)[:kNeighbors]:
                if j == i:
                    continue
                #graph[i, j] += cosX[i, j]
                #graph[j, i] += cosX[i, j]
                graph[i, j] += 1
                graph[j, i] += 1
        toc = time.time() - tic

        return sparse.csc_matrix(graph), newIds

    @staticmethod
    def generateCosineNeighborGraphfromX(X, kNeighbors=10):
        cosX = cosine_similarity(X)
        # return sparse.csc_matrix(X.dot(X.transpose())),newIds
        #print cosX.shape
        n = cosX.shape[0]
        graph = np.zeros((n, n))
        tic = time.time()
        for i in range(n):
            for j in np.argpartition(-cosX[i], kNeighbors)[:kNeighbors]:
                if j == i:
                    continue
                # graph[i, j] += cosX[i, j]
                # graph[j, i] += cosX[i, j]
                graph[i, j] += 1
                graph[j, i] += 1
        toc = time.time() - tic
        #print 'graph generation done in %f seconds.' % toc
        return sparse.csc_matrix(graph)

    @staticmethod
    def getMetaPathGraph(hin,param={'word':True, 'entity':True, 'we_weight':0.112},entity_types=None):
        # get new IDs for documents
        doc_new_ids = {}
        for doc in hin.DocIds:
            doc_old_id = hin.DocIds[doc]
            doc_new_ids[doc_old_id] = len(doc_new_ids)
        # get new IDs for words and entities
        entity_new_ids = {}

        word_flag = param['word']
        entity_flag = param['entity']
        type_flag = True if entity_types else False
        selected_types = []
        if type_flag:
            for t in hin.TypeIds:
                for entity_type in entity_types:
                    if t.startswith(entity_type):
                        selected_types.append(hin.TypeIds[t])
                        break

        entity_count = {}
        for link in hin.Links:
            head = link[0]
            tail = link[2]
            entity_count[head] = entity_count.get(head, 0) + 1
            entity_count[tail] = entity_count.get(tail, 0) + 1

        entity_count_low_threshold = 10
        entity_count_high_threshold = 10000

        for entity in hin.Ids:
            entity_old_id = hin.Ids[entity]
            node_type = hin.NodeTypes[entity_old_id]
            if node_type != hin.TypeIds['document']:
                if word_flag and node_type == hin.TypeIds['word']:
                    entity_new_ids[entity_old_id] = len(entity_new_ids)
                elif type_flag and node_type in selected_types:
                    if entity_count[entity_old_id] >= entity_count_low_threshold and \
                        entity_count[entity_old_id] <= entity_count_high_threshold:
                        entity_new_ids[entity_old_id] = len(entity_new_ids)
                elif not type_flag and entity_flag and node_type != hin.TypeIds['word']:
                    if entity_count[entity_old_id] >= entity_count_low_threshold and \
                        entity_count[entity_old_id] <= entity_count_high_threshold:
                        entity_new_ids[entity_old_id] = len(entity_new_ids)

        n = len(doc_new_ids)
        m = len(entity_new_ids)
        X = sparse.lil_matrix((n,m))

        for link in hin.Links:
            head = link[0]
            relation = link[1]
            tail = link[2]
            # not a document-entity or document-word link
            if hin.NodeTypes[head] != hin.TypeIds['document']:
                continue
            elif param['word'] and hin.NodeTypes[tail] == hin.TypeIds['word']:
                X[doc_new_ids[head], entity_new_ids[tail]] += param['we_weight']

            elif param['entity']:
                if type_flag and hin.NodeTypes[tail] in selected_types and tail in entity_new_ids:
                    X[doc_new_ids[head], entity_new_ids[tail]] += 1
                elif not type_flag and  hin.NodeTypes[tail] != hin.TypeIds['word'] and tail in entity_new_ids:
                    X[doc_new_ids[head], entity_new_ids[tail]] += 1

        Xc = X.tocsc()
        return Xc * Xc.transpose(), doc_new_ids