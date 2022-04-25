import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys

#--------------------------------------------------------------------------------------------------------
def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

#--------------------------------------------------------------------------------------------------------
def load_data(dataset):
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        with open("/home1/07913/xwu236/GAE/Datasets/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("/home1/07913/xwu236/GAE/Datasets/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    graph = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(graph)
    return adj, features

#--------------------------------------------------------------------------------------------------------
def load_label(dataset):
    names = ['ty', 'ally']
    objects = []
    for i in range(len(names)):
        with open("/home1/07913/xwu236/GAE/Datasets/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    ty, ally = tuple(objects)
    test_idx_reorder = parse_index_file("/home1/07913/xwu236/GAE/Datasets/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    label = sp.vstack((ally, ty)).tolil()
    label[test_idx_reorder, :] = label[test_idx_range, :]
    label = np.argmax(label.toarray(), axis = 1)
    return label
