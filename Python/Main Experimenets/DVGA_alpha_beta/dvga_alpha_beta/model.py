import numpy as np
import random as rn
import os
from keras import backend as K
import  tensorflow.compat.v1  as tf
tf.disable_v2_behavior() 

#--------------------------------------------------------------------------------------------------------
# Reproducible
seed=0
os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)
rn.seed(seed)
session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.compat.v1.set_random_seed(seed)    
#--------------------------------------------------------------------------------------------------------------------

from dvgae_alpha_beta.layers import *

dvgae_alpha_beta_feature_hidden=32
dvgae_alpha_beta_feature_latent=16

#------------------------------------------------------------------------------------------------
class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass

#------------------------------------------------------------------------------------------------        
class DVGAE_alpha_beta_feature(Model):

    def __init__(self, placeholders, layers_no,num_features, num_nodes, features_nonzero, **kwargs):
        super(DVGAE_alpha_beta_feature, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.layers_no=layers_no
        self.build()

    def _build(self):
        self.hidden1_ = GraphConvolutionSparse(input_dim = self.input_dim,
                                               output_dim = dvgae_alpha_beta_feature_hidden,
                                               adj = self.adj,
                                               features_nonzero = self.features_nonzero,
                                               act = tf.nn.relu,
                                               dropout = self.dropout,
                                               logging = self.logging)(self.inputs)
        self.hidden1_feature = TraditionalLayer_SparseInput(input_dim=self.input_dim,
                                                            output_dim=dvgae_alpha_beta_feature_hidden,
                                                            features_nonzero=self.features_nonzero,
                                                            act=tf.nn.relu,
                                                            adj = self.adj,
                                                            dropout=self.dropout,
                                                            logging=self.logging)(self.inputs)
        self.feature_reconstruction = TraditionalLayer(input_dim=dvgae_alpha_beta_feature_hidden,
                                                       output_dim=self.input_dim,
                                                       act=tf.nn.relu,
                                                       dropout=self.dropout,
                                                       logging=self.logging)(self.hidden1_feature)            
        
        self.hidden1 = CoordinateOutput(act = lambda x: x,logging = self.logging)([self.hidden1_,self.hidden1_feature])
        self.hidden_cal=self.hidden1
        
        for i in np.arange(self.layers_no-1):
            self.hidden_cal_ = GraphConvolution_Reg(input_dim = dvgae_alpha_beta_feature_hidden,
                                                    output_dim = dvgae_alpha_beta_feature_hidden,
                                                    adj = self.adj,
                                                    act = tf.nn.relu,
                                                    dropout = self.dropout,
                                                    logging = self.logging)(self.hidden_cal)
            self.hidden_cal = CoordinateOutput(act = lambda x: x,logging = self.logging)([self.hidden_cal_,self.hidden1_feature])
        
        self.z_mean = GraphConvolution_Reg(input_dim = dvgae_alpha_beta_feature_hidden,
                                           output_dim = dvgae_alpha_beta_feature_latent,
                                           adj = self.adj,
                                           act = lambda x: x,
                                           dropout = self.dropout,
                                           logging = self.logging)(self.hidden_cal)
        self.z_log_std = GraphConvolution_Reg(input_dim = dvgae_alpha_beta_feature_hidden,
                                              output_dim = dvgae_alpha_beta_feature_latent,
                                              adj = self.adj,
                                              act = lambda x: x,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.hidden_cal)
        
        self.z = self.z_mean + tf.random_normal([self.n_samples, dvgae_alpha_beta_feature_latent]) * tf.exp(self.z_log_std)
        self.reconstructions = InnerProductDecoder(act = lambda x: x,logging = self.logging)(self.z)
        
#------------------------------------------------------------------------------------------------
class DVGAE_alpha_beta_no_feature(Model):
    def __init__(self, placeholders, layers_no,num_features, num_adj,num_nodes,features_nonzero, **kwargs):
        super(DVGAE_alpha_beta_no_feature, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.input_adj_dim =num_adj
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.layers_no=layers_no
        self.build()

    def _build(self):
        self.hidden1_ = GraphConvolutionSparse(input_dim = self.input_dim,
                                               output_dim = dvgae_alpha_beta_feature_hidden,
                                               adj = self.adj,
                                               features_nonzero = self.features_nonzero,
                                               act = tf.nn.relu,
                                               dropout = self.dropout,
                                               logging = self.logging)(self.inputs)
        
        self.hidden1_adj = TraditionalLayer_SparseInput(input_dim=self.input_adj_dim,
                                                        output_dim=dvgae_alpha_beta_feature_hidden,
                                                        features_nonzero=self.features_nonzero,
                                                        act=tf.nn.relu,
                                                        adj = self.adj,
                                                        dropout=self.dropout,
                                                        logging=self.logging)(self.adj)
        
        self.adj_reconstruction = TraditionalLayer(input_dim=dvgae_alpha_beta_feature_hidden,
                                                   output_dim=self.input_adj_dim,
                                                   act=tf.nn.relu,
                                                   dropout=self.dropout,
                                                   logging=self.logging)(self.hidden1_adj)     
        
        self.hidden1 = CoordinateOutput(act = lambda x: x,logging = self.logging)([self.hidden1_,self.hidden1_adj])
        self.hidden_cal=self.hidden1
        
        for i in np.arange(self.layers_no-1):
            self.hidden_cal_ = GraphConvolution_Reg(input_dim = dvgae_alpha_beta_feature_hidden,
                                                    output_dim = dvgae_alpha_beta_feature_hidden,
                                                    adj = self.adj,
                                                    act = tf.nn.relu,
                                                    dropout = self.dropout,
                                                    logging = self.logging)(self.hidden_cal)
        
            self.hidden_cal = CoordinateOutput(act = lambda x: x,logging = self.logging)([self.hidden_cal_,self.hidden1_adj])
        
        self.z_mean = GraphConvolution_Reg(input_dim = dvgae_alpha_beta_feature_hidden,
                                           output_dim = dvgae_alpha_beta_feature_latent,
                                           adj = self.adj,
                                           act = lambda x: x,
                                           dropout = self.dropout,
                                           logging = self.logging)(self.hidden_cal)
        
        self.z_log_std = GraphConvolution_Reg(input_dim = dvgae_alpha_beta_feature_hidden,
                                              output_dim = dvgae_alpha_beta_feature_latent,
                                              adj = self.adj,
                                              act = lambda x: x,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.hidden_cal)

        self.z = self.z_mean + tf.random_normal([self.n_samples, dvgae_alpha_beta_feature_latent]) * tf.exp(self.z_log_std)
        self.reconstructions = InnerProductDecoder(act = lambda x: x,logging = self.logging)(self.z)