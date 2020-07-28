import codecs

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


def log(message):
    result_file_name = FLAGS.log_file
    log_file = result_file_name
    codecs.open(log_file, mode='a', encoding='utf-8').write(message + "\n")
    print(message)


def load_data(knn=False, nnSize=0):
    adjs = []
    pos_edge, neg_edge = load_SL_matrix()
    adjs.append(load_dense_feature('../data/Human_GOsim.txt', knn=knn, nnSize=nnSize))
    adjs.append(load_dense_feature('../data/Human_GOsim_CC.txt', knn=knn, nnSize=nnSize))
    adjs.append(load_sparse_features('../data/biogrid_ppi_sparse.txt'))

    return pos_edge, neg_edge, adjs


def load_cmf_data():
    print("loading sl data...")
    adjs = []
    adjs.append(sp.coo_matrix(np.loadtxt('../cmfdata/F1_F2_coexpr_for_train')))
    adjs.append(sp.coo_matrix(np.loadtxt('../cmfdata/F1_F2_me_for_train')))
    adjs.append(sp.coo_matrix(np.loadtxt('../cmfdata/F1_F2_pathway_for_train')))
    adjs.append(sp.coo_matrix(np.loadtxt('../cmfdata/F1_F2_proteincomplex_for_train')))
    adjs.append(sp.coo_matrix(np.loadtxt('../cmfdata/F1_F2_ppi_for_train')))
    pos_edge = np.load('../cmfdata/pos_edge_binary.npy').astype(np.int32)
    neg_edge = np.load('../cmfdata/neg_edge_binary.npy').astype(np.int32)
    return pos_edge, neg_edge, adjs


def load_SL_matrix():
    print("Loading SL matrix...")

    slMapping, NumNodes = dict(), 0
    with open('../data/List_Proteins_in_SL.txt', 'r') as inf:
        id = 0
        for line in inf:
            slMapping[line.replace('\n', '')] = id
            id += 1
        NumNodes = id

    row, col = [], []
    with open("../data/SL_Human_Approved.txt", "r") as inf:
        for line in inf:
            id = line.rstrip().split()
            row.append(slMapping[id[0]])
            col.append(slMapping[id[1]])

    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(NumNodes, NumNodes))
    adj = adj + adj.T
    adj = adj.toarray()
    x, y = np.triu_indices(NumNodes, k=1)
    pos_edge, neg_edge = [], []

    for e in zip(x, y):
        if adj[e[0], e[1]] == 0:
            neg_edge.append(e)
        else:
            pos_edge.append(e)

    pos_edge = np.array(pos_edge, dtype=np.int32)
    neg_edge = np.array(neg_edge, dtype=np.int32)
    return pos_edge, neg_edge


def load_nonPred_SL_matrix():
    print("Loading SL matrix...")

    slMapping, NumNodes = dict(), 0
    with open('../data/List_Proteins_in_SL.txt', 'r') as inf:
        id = 0
        for line in inf:
            slMapping[line.replace('\n', '')] = id
            id += 1
        NumNodes = id

    computational_pairs = []
    with open("../data/computational_pairs.txt", "r") as inf:
        for line in inf:
            name1, name2 = line.rstrip().split()
            computational_pairs.append({name1, name2})

    row, col = [], []
    with open("../data/SL_Human_Approved.txt", "r") as inf:
        for line in inf:
            name1, name2, _ = line.rstrip().split()
            if {name1, name2} not in computational_pairs:
                row.append(slMapping[name1])
                col.append(slMapping[name2])

    adj = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(NumNodes, NumNodes))
    adj = adj.toarray()
    x, y = np.triu_indices(NumNodes, k=1)
    pos_edge, neg_edge = [], []

    for e in zip(x, y):
        if adj[e[0], e[1]] == 0:
            neg_edge.append(e)
        else:
            pos_edge.append(e)

    pos_edge = np.array(pos_edge, dtype=np.int32)
    neg_edge = np.array(neg_edge, dtype=np.int32)
    return pos_edge, neg_edge


def build_KNN_mateix(S, nn_size):
    m, n = S.shape
    X = np.zeros((m, n))
    for i in range(m):
        ii = np.argsort(S[i, :])[::-1][:min(nn_size, n)]
        X[i, ii] = S[i, ii]
    return X


def array2coo(S, t=0):
    m, n = np.shape(S)
    row, col = [], []
    for i in range(len(S)):
        for j in range(i, len(S[i])):
            if S[i][j] > t:
                row.append(i)
                col.append(j)

    coo_matrix = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(m, n))
    coo_matrix = coo_matrix + coo_matrix.T

    return coo_matrix


def load_dense_feature(fileName, knn=False, nnSize=0):
    print('Loading ' + fileName)
    featureFile = open(fileName, 'r')
    line = featureFile.readlines()
    featureMatrix = np.zeros((6375, 6375))

    for i in range(len(line)):
        s = line[i].replace('\n', '').split('\t')
        for j in range(i, len(s)):
            if s[j] == '': break
            featureMatrix[i][j + 1] = float(s[j])

    featureMatrix = featureMatrix + featureMatrix.T
    if knn == True:
        featureMatrix = build_KNN_mateix(featureMatrix, nn_size=nnSize)

    featureMatrix = featureMatrix + featureMatrix.T
    features = array2coo(featureMatrix)

    return features


def load_sparse_features(fileName):
    print("Loading " + fileName)
    row, col = [], []
    with open(fileName, 'r') as inf:
        for line in inf:
            r, c = line.replace('\n', '').split('\t')
            row.append(r)
            col.append(c)

    features = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=(6375, 6375))
    features = features + features.T

    return features


def mean_confidence_interval(data, confidence=0.95):
    import scipy as sp
    import scipy.stats
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_normalized = degree_mat_inv_sqrt.dot(adj_)
    adj_normalized = adj_normalized.dot(degree_mat_inv_sqrt).tocoo()

    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(support, features, adj_orig, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj_orig']: adj_orig})
    return feed_dict


if __name__ == '__main__':
    # a = load_dense_feature('../data/Human_GOsim.txt', knn=True, nnSize=45)
    # logits = np.array([10989.211, 10995.484, 10979.361, 10991.63, 10990.312])
    logits = np.array([2441.0, 2382.5, 2296.5])
    logits /= 1000
    softmax = np.exp(logits) / np.sum(np.exp(logits))
    print(softmax)
