import numpy as np
import scipy.sparse as sp
import tensorflow as tf

import random as rn
import os
from keras import backend as K

#--------------------------------------------------------------------------------------------------------
def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

#--------------------------------------------------------------------------------------------------------
def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    degree_mat_inv_sqrt = sp.diags(np.power(np.array(adj_.sum(1)), -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt)
    return sparse_to_tuple(adj_normalized)

#--------------------------------------------------------------------------------------------------------
def construct_feed_dict(adj_normalized, adj, features, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

#--------------------------------------------------------------------------------------------------------
def construct_feed_dict_synthesis(adj_normalized, adj, features, features_branch,placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['features_branch']: features_branch})
    feed_dict.update({placeholders['adj']: adj_normalized})
    feed_dict.update({placeholders['adj_orig']: adj})
    return feed_dict

#--------------------------------------------------------------------------------------------------------
def mask_test_edges(adj, seed,test_percent=10., val_percent=5.):
    

    seed=seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    rn.seed(seed)
    session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    tf.compat.v1.set_random_seed(seed)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
    adj = adj - sp.dia_matrix((adj.diagonal()[None, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    assert adj.diagonal().sum() == 0

    edges_positive, _, _ = sparse_to_tuple(adj)
    edges_positive = edges_positive[edges_positive[:,1] > edges_positive[:,0],:]

    num_test = int(np.floor(edges_positive.shape[0] / (100. / test_percent)))
    num_val = int(np.floor(edges_positive.shape[0] / (100. / val_percent)))

    edges_positive_idx = np.arange(edges_positive.shape[0])
    np.random.shuffle(edges_positive_idx)
    val_edge_idx = edges_positive_idx[:num_val]
    test_edge_idx = edges_positive_idx[num_val:(num_val + num_test)]
    test_edges = edges_positive[test_edge_idx]
    val_edges = edges_positive[val_edge_idx]
    train_edges = np.delete(edges_positive, np.hstack([test_edge_idx, val_edge_idx]), axis = 0)
    positive_idx, _, _ = sparse_to_tuple(adj)
    positive_idx = positive_idx[:,0]*adj.shape[0] + positive_idx[:,1]
    test_edges_false = np.empty((0,2),dtype='int64')
    idx_test_edges_false = np.empty((0,),dtype='int64')

    while len(test_edges_false) < len(test_edges):

        idx = np.random.choice(adj.shape[0]**2, 2*(num_test - len(test_edges_false)), replace = True)

        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]

        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()

        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]

        coords = np.unique(coords, axis = 0)
        np.random.shuffle(coords)

        coords = coords[coords[:,0] != coords[:,1]]

        coords = coords[:min(num_test, len(idx))]
        test_edges_false = np.append(test_edges_false, coords, axis = 0)
        idx = idx[:min(num_test, len(idx))]
        idx_test_edges_false = np.append(idx_test_edges_false, idx)

    val_edges_false = np.empty((0,2), dtype = 'int64')
    idx_val_edges_false = np.empty((0,), dtype = 'int64')
    while len(val_edges_false) < len(val_edges):

        idx = np.random.choice(adj.shape[0]**2, 2*(num_val - len(val_edges_false)), replace = True)

        idx = idx[~np.in1d(idx, positive_idx, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_test_edges_false, assume_unique = True)]
        idx = idx[~np.in1d(idx, idx_val_edges_false, assume_unique = True)]

        rowidx = idx // adj.shape[0]
        colidx = idx % adj.shape[0]
        coords = np.vstack((rowidx,colidx)).transpose()

        lowertrimask = coords[:,0] > coords[:,1]
        coords[lowertrimask] = coords[lowertrimask][:,::-1]

        coords = np.unique(coords, axis = 0)
        np.random.shuffle(coords)

        coords = coords[coords[:,0] != coords[:,1]]

        coords = coords[:min(num_val, len(idx))]
        val_edges_false = np.append(val_edges_false, coords, axis = 0)
        idx = idx[:min(num_val, len(idx))]
        idx_val_edges_false = np.append(idx_val_edges_false, idx)

    train_edges_linear = train_edges[:,0]*adj.shape[0] + train_edges[:,1]
    test_edges_linear = test_edges[:,0]*adj.shape[0] + test_edges[:,1]
    assert not np.any(np.in1d(idx_test_edges_false, positive_idx))
    assert not np.any(np.in1d(idx_val_edges_false, positive_idx))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0]+val_edges[:,1], train_edges_linear))
    assert not np.any(np.in1d(test_edges_linear, train_edges_linear))
    assert not np.any(np.in1d(val_edges[:,0]*adj.shape[0]+val_edges[:,1], test_edges_linear))

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, val_edges, val_edges_false, test_edges, test_edges_false