import numpy as np
import os
import scipy.sparse as sp
import tensorflow as tf
import time
import pandas as pd
import networkx as nx

# Import defined methods
import sys
sys.path.append(r'/home1/07913/xwu236/GAE')

from linear_gae.evaluation import get_roc_score
from linear_gae.input_data import load_data, load_label
from linear_gae.model import *
from linear_gae.optimizer import OptimizerAE, OptimizerVAE
from linear_gae.preprocessing import *

def write_to_csv(p_data,p_path):
    dataframe = pd.DataFrame(p_data)
    dataframe.to_csv(p_path, mode='a',header=False,index=False,sep=',')
    del dataframe

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataset='cora'
task='link_prediction'
model_name='deep18_gcn_ae'
dropout=0.
epochs=200
features_used=False
learning_rate=0.01
nb_run=10 # Number of model run + test
prop_val=5 # Proportion of edges in validation set (for Link Prediction task)
prop_test=10 # Proportion of edges in test set (for Link Prediction task)
validation=True # Whether to report validation results at each epoch (for Link Prediction task)
verbose=True # Whether to print comments details
kcore=False # Whether to run k-core decomposition and use the framework. False = model will be trained on the entire graph
task='link_prediction'

p_model_name=model_name

mean_time = []
if verbose:
    print("Loading data...")
adj_init, features_init = load_data(dataset)

# Lists to collect average results
if task == 'link_prediction':
    mean_roc = []
    mean_ap = []

mean_time = []

# Load graph dataset
if verbose:
    print("Loading data...")
adj_init, features_init = load_data(dataset)

# The entire training+test process is repeated nb_run times
for seed_i in np.arange(nb_run):
    seed=seed_i
    if task == 'link_prediction' :
        if verbose:
            print("Masking test edges...")
        adj, val_edges, val_edges_false, test_edges, test_edges_false = \
        mask_test_edges(adj_init, seed,prop_test, prop_val)
        
    # Start computation of running times
    t_start = time.time()

    if features_used:
        features = features_init
        
    # Preprocessing and initialization
    if verbose:
        print("Preprocessing and Initializing...")
        
    # Compute number of nodes
    num_nodes = adj.shape[0]
    # If features are not used, replace feature matrix by identity matrix
    if not features_used:
        features = sp.identity(adj.shape[0])
    # Preprocessing on node features
    features = sparse_to_tuple(features)
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    
    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape = ())
    }

    # Create model
    if model_name == 'gcn_ae':
        # Standard Graph Autoencoder
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_name == 'gcn_vae':
        # Standard Graph Variational Autoencoder
        model = GCNModelVAE(placeholders, num_features, num_nodes,
                            features_nonzero)
    elif model_name == 'linear_ae':
        # Linear Graph Autoencoder
        model = LinearModelAE(placeholders, num_features, features_nonzero)
    elif model_name == 'linear_vae':
        # Linear Graph Variational Autoencoder
        model = LinearModelVAE(placeholders, num_features, num_nodes,
                               features_nonzero)
    elif model_name == 'deep18_gcn_ae':
        # Deep (3-layer GCN) Graph Autoencoder
        model = Deep18GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_name == 'deep_gcn_vae':
        # Deep (3-layer GCN) Graph Variational Autoencoder
        model = DeepGCNModelVAE(placeholders, num_features, num_nodes,
                                features_nonzero)
    else:
        raise ValueError('Undefined model!')

    # Optimizer
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0]
                                                - adj.sum()) * 2)
    with tf.name_scope('optimizer'):
        # Optimizer for Non-Variational Autoencoders
        if model_name in ('gcn_ae', 'linear_ae', 'deep_gcn_ae','deep18_gcn_ae'):
            opt = OptimizerAE(preds = model.reconstructions,
                              labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                            validate_indices = False), [-1]),
                              pos_weight = pos_weight,
                              norm = norm)

            # Optimizer for Variational Autoencoders
        elif model_name in ('gcn_vae', 'linear_vae', 'deep_gcn_vae','deep18_gcn_vae'):
            opt = OptimizerVAE(preds = model.reconstructions,
                               labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                             validate_indices = False), [-1]),
                               model = model,
                               num_nodes = num_nodes,
                               pos_weight = pos_weight,
                               norm = norm)

    # Normalization and preprocessing on adjacency matrix
    adj_norm = preprocess_graph(adj)
    adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

    # Initialize TF session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Model training
    if verbose:
        print("Training...")

    for epoch in range(epochs):
        # Flag to compute running time for each epoch
        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm, adj_label, features,
                                        placeholders)
        feed_dict.update({placeholders['dropout']: dropout})
        # Weights update
        outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],
                        feed_dict = feed_dict)
        # Compute average loss
        avg_cost = outs[1]
        if verbose:
            # Display epoch information
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
                  "time=", "{:.5f}".format(time.time() - t))
            # Validation, for Link Prediction
            if not kcore and validation and task == 'link_prediction':
                feed_dict.update({placeholders['dropout']: 0})
                emb = sess.run(model.z_mean, feed_dict = feed_dict)
                feed_dict.update({placeholders['dropout']: dropout})
                val_roc, val_ap = get_roc_score(val_edges, val_edges_false, emb)
                print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))

    # Flag to compute Graph AE/VAE training time
    t_model = time.time()


    # Compute embedding

    # Get embedding from model
    emb = sess.run(model.z_mean, feed_dict = feed_dict)
    
    # Compute mean total running time
    mean_time.append(time.time() - t_start)
    
    # Test model
    if verbose:
        print("Testing model...")
    # Link Prediction: classification edges/non-edges
    if task == 'link_prediction':
        # Get ROC and AP scores
        roc_score, ap_score = get_roc_score(test_edges, test_edges_false, emb)
        # Report scores
        mean_roc.append(roc_score)
        mean_ap.append(ap_score)

mean_time_=np.array(mean_time)
write_to_csv(mean_time_.reshape(1,len(mean_time_)),"/home1/07913/xwu236/GAE/1Cora_AE/log/"+p_model_name+"_time_"+str(features_used)+".csv")

mean_roc_=np.array(mean_roc)
write_to_csv(mean_roc_.reshape(1,len(mean_roc_)),"/home1/07913/xwu236/GAE/1Cora_AE/log/"+p_model_name+"_roc_"+str(features_used)+".csv")

mean_ap_=np.array(mean_ap)
write_to_csv(mean_ap_.reshape(1,len(mean_ap_)),"/home1/07913/xwu236/GAE/1Cora_AE/log/"+p_model_name+"_ap_"+str(features_used)+".csv")
