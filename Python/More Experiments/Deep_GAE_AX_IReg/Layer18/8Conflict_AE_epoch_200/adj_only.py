#--------------------------------------------------------------------------------------------------------------------
# Reproducible

import numpy as np
import random as rn
import os
from keras import backend as K
#import  tensorflow.compat.v1  as tf
#tf.disable_v2_behavior() 
import tensorflow as tf

seed=0
os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)
rn.seed(seed)
session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.compat.v1.set_random_seed(seed)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

#--------------------------------------------------------------------------------------------------------------------
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

#--------------------------------------------------------------------------------------------------------------------
import scipy.sparse as sp
import time
import pandas as pd

# Import defined methods
import sys

layers_no=18
data_path="/home/07913/xwu236/Deep_GAE_AX_IReg/Datasets/"
path_now="/home/07913/xwu236/Deep_GAE_AX_IReg/Layer18/8Conflict_AE_epoch_200/log/"

dataset='conflict'
model_name='deep_gcn_ae_resadj_coordinate_AE'

sys.path.append(r"/home/07913/xwu236/Deep_GAE_AX_IReg")

from linear_gae_mat.evaluation import get_roc_score
from linear_gae_mat.input_data import load_data, load_label,load_mat_data
from linear_gae_mat.model import *
from linear_gae_mat.optimizer import OptimizerAE, OptimizerVAE,OptimizerAE_FeatureReconstrution,OptimizerAE_AdjReconstrution
from linear_gae_mat.preprocessing import *

def write_to_csv(p_data,p_path):
    dataframe = pd.DataFrame(p_data)
    dataframe.to_csv(p_path, mode='a',header=False,index=False,sep=',')
    del dataframe
    
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)
		print("Successful!")
	else:
		print("Failure") 

dropout=0.0
epochs=200
features_used=False
learning_rate=0.01
nb_run=10
prop_val=5 # Proportion of edges in validation set
prop_test=10 # Proportion of edges in test set
validation=True # Whether to report validation results at each epoch (for Link Prediction task)
verbose=True
task='link_prediction'

p_model_name=model_name

# Load data
adj_, features_ = load_mat_data(dataset,data_path)

features_init = sp.csr_matrix(features_)

adj_init=sp.csr_matrix(adj_)

num_adj=adj_init.shape[1]

for loss_coeff_adj_i in np.array([0.000001,0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.5,1,1.5,2]):
    for loss_coeff_AE_i in np.array([0.001,0.01,0.000001,0.00001,0.05,0.1,0.5,1,1.5,2,0.005,0.0001]):
        loss_coeff_adj=np.around(loss_coeff_adj_i,8)
        loss_coeff_AE=np.around(loss_coeff_AE_i,8)
        file = path_now+p_model_name+"/adj"+str(loss_coeff_adj)+"_fea"+str(loss_coeff_AE)+"/"
        mkdir(file)

        for seed_i in np.arange(nb_run):
    
            seed=seed_i
            lost_list=[]
            roc_list=[]
            ap_list=[]
            mean_time=[]
    
            if task == 'link_prediction' :
                adj, val_edges, val_edges_false, test_edges, test_edges_false = \
                mask_test_edges(adj_init, seed,prop_test, prop_val)
                
            t_start = time.time()

            if features_used:
                features = features_init
    
            num_nodes = adj.shape[0]

            if not features_used:
                features = sp.identity(adj.shape[0])

            features = sparse_to_tuple(features)
            num_features = features[2][1]
            features_nonzero = features[1].shape[0]
        
            placeholders = {
                'features': tf.sparse_placeholder(tf.float32),
                'adj': tf.sparse_placeholder(tf.float32),
                'adj_orig': tf.sparse_placeholder(tf.float32),
                'dropout': tf.placeholder_with_default(0., shape = ())
            }

            # Create model
            if model_name == 'gcn_ae':
                model = GCNModelAE(placeholders, num_features, features_nonzero)
            elif model_name == 'gcn_vae':
                model = GCNModelVAE(placeholders, num_features, num_nodes,features_nonzero)
            elif model_name == 'linear_ae':
                model = LinearModelAE(placeholders, num_features, features_nonzero)
            elif model_name == 'linear_vae':
                model = LinearModelVAE(placeholders, num_features, num_nodes,features_nonzero)
            elif model_name == 'deep_gcn_ae':
                model = DeepGCNModelAE(placeholders, num_features, features_nonzero)
            elif model_name == 'deep_gcn_vae':
                model = DeepGCNModelVAE(placeholders, num_features, num_nodes,features_nonzero)
            elif model_name=='deep_gcn_ae_resfeature_coordinate_AE':
                model = DeepGCNModelAE_ResFeature_Coordinate_AE(placeholders, layers_no,num_features, features_nonzero)
            elif model_name=='deep_gcn_ae_resadj_coordinate_AE':
                model=DeepGCNModelAE_ResAdj_Coordinate_AE(placeholders, layers_no,num_features, num_adj,features_nonzero)
            else:
                raise ValueError('Undefined model!')

            pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0]- adj.sum()) * 2)
            with tf.name_scope('optimizer'):
                if model_name in ('gcn_ae', 'linear_ae', 'deep_gcn_ae'):
                    opt = OptimizerAE(preds = model.reconstructions,
                                      labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                    validate_indices = False), [-1]),
                                      pos_weight = pos_weight,
                                      norm = norm)

                elif model_name in ('gcn_vae', 'linear_vae', 'deep_gcn_vae'):
                    opt = OptimizerVAE(preds = model.reconstructions,
                                       labels = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                     validate_indices = False), [-1]),
                                       model = model,
                                       num_nodes = num_nodes,
                                       pos_weight = pos_weight,
                                       norm = norm)
    
                elif model_name in ('deep_gcn_ae_resfeature_coordinate_AE'):
                    opt =OptimizerAE_FeatureReconstrution(preds_adj = model.reconstructions,\
                                                          preds_Features=model.feature_reconstruction,\
                                                          labels_adj = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],\
                                                                                                            validate_indices = False), [-1]),\
                                                          labels_Features=placeholders['features'],\
                                                          labels_adj_rec=placeholders['adj'],\
                                                          loss_coeff_adj=loss_coeff_adj,\
                                                          loss_coeff_AE=loss_coeff_AE,\
                                                          pos_weight = pos_weight,\
                                                          norm = norm)
                
                elif model_name in ('deep_gcn_ae_resadj_coordinate_AE'):
                    opt =OptimizerAE_AdjReconstrution(preds_adj = model.reconstructions,\
                                              preds_adj_rec=model.adj_reconstruction,\
                                              labels_adj = tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                                validate_indices = False), [-1]),\
                                              labels_Features=placeholders['features'],\
                                              labels_adj_rec=placeholders['adj'],\
                                              loss_coeff_adj=loss_coeff_adj,\
                                              loss_coeff_AE=loss_coeff_AE,\
                                              pos_weight = pos_weight,\
                                              norm = norm)

            adj_norm = preprocess_graph(adj)
            adj_label = sparse_to_tuple(adj + sp.eye(adj.shape[0]))

            sess = tf.Session()
            sess.run(tf.global_variables_initializer())
    
            for epoch in range(epochs):
                t = time.time()
                feed_dict = construct_feed_dict(adj_norm, adj_label, features,placeholders)
                feed_dict.update({placeholders['dropout']: dropout})
                outs = sess.run([opt.opt_op, opt.cost, opt.accuracy],feed_dict = feed_dict)
                avg_cost = outs[1]
                if verbose:
                    lost_list.append(avg_cost)
                    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),"time=", "{:.5f}".format(time.time() - t))
                    if validation and task == 'link_prediction':
                        feed_dict.update({placeholders['dropout']: 0})
                        emb = sess.run(model.z_mean, feed_dict = feed_dict)
                        feed_dict.update({placeholders['dropout']: dropout})
                        val_roc, val_ap = get_roc_score(val_edges, val_edges_false, emb)
                        roc_list.append(val_roc)
                        ap_list.append(val_ap)
                        print("val_roc=", "{:.5f}".format(val_roc), "val_ap=", "{:.5f}".format(val_ap))

            emb = sess.run(model.z_mean, feed_dict = feed_dict)   
            mean_time.append(time.time() - t_start)
    
            if task == 'link_prediction':
                roc_score, ap_score = get_roc_score(test_edges, test_edges_false, emb)
                roc_list.append(roc_score)
                ap_list.append(ap_score)
        
            mean_time_=np.array(mean_time)
            write_to_csv(mean_time_.reshape(1,len(mean_time_)),file+"time.csv")

            roc_list_=np.array(roc_list)
            write_to_csv(roc_list_.reshape(1,len(roc_list_)),file+"roc.csv")

            ap_list_=np.array(ap_list)
            write_to_csv(ap_list_.reshape(1,len(ap_list_)),file+"ap.csv")