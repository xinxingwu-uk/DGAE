
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import  tensorflow.compat.v1  as tf
tf.disable_v2_behavior() 

#--------------------------------------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#--------------------------------------------------------------------------------------------------------
def get_roc_score(edges_pos, edges_neg, emb):
    preds = []
    preds_neg = []
    for e in edges_pos:
        preds.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))
    for e in edges_neg:
        preds_neg.append(sigmoid(emb[e[0],:].dot(emb[e[1],:].T)))
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    return roc_score, ap_score
