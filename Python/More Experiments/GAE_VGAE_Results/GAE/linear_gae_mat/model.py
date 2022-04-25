#--------------------------------------------------------------------------------------------------------
# Reproducible
import numpy as np
import random as rn
import os
#import  tensorflow.compat.v1  as tf
#tf.disable_v2_behavior() 

import tensorflow as tf

from keras import backend as K

seed=0
os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)
rn.seed(seed)
session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

tf.compat.v1.set_random_seed(seed)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)
    
#--------------------------------------------------------------------------------------------------------------------

from linear_gae.layers import *

# LinearModelAE
LinearModelAE_dimension=16

# GAEModelAE
GAEModelAE_hidden=32
GAEModelAE_dimension=16

# Deep3ModelAE
Deep3ModelAE_hidden=32
Deep3ModelAE_dimension=16

# Deep3ModelAE_ResFeature_Coordinate_AE
Deep3ModelAE_ResFeature_Coordinate_AE_hidden=32
Deep3ModelAE_ResFeature_Coordinate_AE_dimension=16

# DeepGCNModelAE_ResAdj_Coordinate_AE
DeepGCNModelAE_ResAdj_Coordinate_AE_hidden=32
DeepGCNModelAE_ResAdj_Coordinate_AE_dimension=16

layers6_no=6

layers12_no=12

layers18_no=18

layers36_no=36

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
    
#-------------------------------------------------------------------------------------------------------- 
class LinearModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(LinearModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.z_mean = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = LinearModelAE_dimension,
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = lambda x: x,
                                             dropout = self.dropout,
                                             logging = self.logging)(self.inputs)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)

#------------------------------------------------------------------------------------------------
class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)
        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = GAEModelAE_hidden,
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout,
                                             logging = self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim = GAEModelAE_hidden,
                                       output_dim = GAEModelAE_dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)
        

#------------------------------------------------------------------------------------------------
class DeepGCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(DeepGCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = Deep3ModelAE_hidden,
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.inputs)

        self.hidden2 = GraphConvolution(input_dim = Deep3ModelAE_hidden,
                                        output_dim = Deep3ModelAE_hidden,
                                        adj = self.adj,
                                        act = tf.nn.relu,
                                        dropout = self.dropout,
                                        logging = self.logging)(self.hidden1)

        self.z_mean = GraphConvolution(input_dim = Deep3ModelAE_hidden,
                                       output_dim = Deep3ModelAE_dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden2)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)
        
#------------------------------------------------------------------------------------------------
class Deep6GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(Deep6GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = Deep3ModelAE_hidden,
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.inputs)
        self.hidden_cal=self.hidden1
        
        for i in np.arange(layers6_no-1):
            self.hidden_cal_ = GraphConvolution(input_dim = Deep3ModelAE_hidden,
                                                output_dim = Deep3ModelAE_hidden,
                                                adj = self.adj,
                                                act = tf.nn.relu,
                                                dropout = self.dropout,
                                                logging = self.logging)(self.hidden_cal)
            self.hidden_cal=self.hidden_cal_

        self.z_mean = GraphConvolution(input_dim = Deep3ModelAE_hidden,
                                       output_dim = Deep3ModelAE_dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden_cal)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)

#------------------------------------------------------------------------------------------------
class Deep12GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(Deep12GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = Deep3ModelAE_hidden,
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.inputs)
        self.hidden_cal=self.hidden1
        
        for i in np.arange(layers12_no-1):
            self.hidden_cal_ = GraphConvolution(input_dim = Deep3ModelAE_hidden,
                                                output_dim = Deep3ModelAE_hidden,
                                                adj = self.adj,
                                                act = tf.nn.relu,
                                                dropout = self.dropout,
                                                logging = self.logging)(self.hidden_cal)
            self.hidden_cal=self.hidden_cal_

        self.z_mean = GraphConvolution(input_dim = Deep3ModelAE_hidden,
                                       output_dim = Deep3ModelAE_dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden_cal)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)
        
        
#------------------------------------------------------------------------------------------------
class Deep18GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(Deep18GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = Deep3ModelAE_hidden,
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.inputs)
        self.hidden_cal=self.hidden1
        
        for i in np.arange(layers18_no-1):
            self.hidden_cal_ = GraphConvolution(input_dim = Deep3ModelAE_hidden,
                                                output_dim = Deep3ModelAE_hidden,
                                                adj = self.adj,
                                                act = tf.nn.relu,
                                                dropout = self.dropout,
                                                logging = self.logging)(self.hidden_cal)
            self.hidden_cal=self.hidden_cal_

        self.z_mean = GraphConvolution(input_dim = Deep3ModelAE_hidden,
                                       output_dim = Deep3ModelAE_dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden_cal)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)
        
#------------------------------------------------------------------------------------------------
class Deep36GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(Deep36GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = Deep3ModelAE_hidden,
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.inputs)
        self.hidden_cal=self.hidden1
        
        for i in np.arange(layers36_no-1):
            self.hidden_cal_ = GraphConvolution(input_dim = Deep3ModelAE_hidden,
                                                output_dim = Deep3ModelAE_hidden,
                                                adj = self.adj,
                                                act = tf.nn.relu,
                                                dropout = self.dropout,
                                                logging = self.logging)(self.hidden_cal)
            self.hidden_cal=self.hidden_cal_

        self.z_mean = GraphConvolution(input_dim = Deep3ModelAE_hidden,
                                       output_dim = Deep3ModelAE_dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden_cal)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)
        
#------------------------------------------------------------------------------------------------        
class DeepGCNModelAE_ResFeature_Coordinate_AE(Model):

    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(DeepGCNModelAE_ResFeature_Coordinate_AE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1_ = GraphConvolutionSparse(input_dim = self.input_dim,
                                               output_dim = Deep3ModelAE_ResFeature_Coordinate_AE_hidden,
                                               adj = self.adj,
                                               features_nonzero = self.features_nonzero,
                                               act = tf.nn.relu,
                                               dropout = self.dropout,
                                               logging = self.logging)(self.inputs)
        
        self.hidden1_feature = TraditionalLayer_SparseInput(input_dim=self.input_dim,
                                                            output_dim=Deep3ModelAE_ResFeature_Coordinate_AE_hidden,
                                                            features_nonzero=self.features_nonzero,
                                                            act=tf.nn.relu,
                                                            dropout=self.dropout,
                                                            logging=self.logging)(self.inputs)
        
        self.feature_reconstruction = TraditionalLayer(input_dim=Deep3ModelAE_ResFeature_Coordinate_AE_hidden,
                                                       output_dim=self.input_dim,
                                                       act=tf.nn.relu,
                                                       dropout=self.dropout,
                                                       logging=self.logging)(self.hidden1_feature)            
        
        self.hidden1 = CoordinateOutput(act = lambda x: x,logging = self.logging)([self.hidden1_,self.hidden1_feature])

        
        self.hidden2_ = GraphConvolution(input_dim = Deep3ModelAE_ResFeature_Coordinate_AE_hidden,
                                         output_dim = Deep3ModelAE_ResFeature_Coordinate_AE_hidden,
                                         adj = self.adj,
                                         act = tf.nn.relu,
                                         dropout = self.dropout,
                                         logging = self.logging)(self.hidden1)
        
        self.hidden2 = CoordinateOutput(act = lambda x: x,logging = self.logging)([self.hidden2_,self.hidden1_feature])

        self.z_mean = GraphConvolution(input_dim = Deep3ModelAE_ResFeature_Coordinate_AE_hidden,
                                       output_dim = Deep3ModelAE_ResFeature_Coordinate_AE_dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden2)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)

#------------------------------------------------------------------------------------------------
class DeepGCNModelAE_ResAdj_Coordinate_AE(Model):
    def __init__(self, placeholders, num_features, num_adj,features_nonzero, **kwargs):
        super(DeepGCNModelAE_ResAdj_Coordinate_AE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.input_adj_dim =num_adj
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1_ = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = DeepGCNModelAE_ResAdj_Coordinate_AE_hidden,
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.inputs)
        
        self.hidden1_adj = TraditionalLayer_SparseInput(input_dim=self.input_adj_dim,
                                                        output_dim=DeepGCNModelAE_ResAdj_Coordinate_AE_hidden,
                                                        features_nonzero=self.features_nonzero,
                                                        act=tf.nn.relu,
                                                        dropout=self.dropout,
                                                        logging=self.logging)(self.adj)
        
        self.adj_reconstruction = TraditionalLayer(input_dim=Deep3ModelAE_ResFeature_Coordinate_AE_hidden,
                                                   output_dim=self.input_adj_dim,
                                                   act=tf.nn.relu,
                                                   dropout=self.dropout,
                                                   logging=self.logging)(self.hidden1_adj)     
        
        self.hidden1 = CoordinateOutput(act = lambda x: x,logging = self.logging)([self.hidden1_,self.hidden1_adj])

        self.hidden2_ = GraphConvolution(input_dim = DeepGCNModelAE_ResAdj_Coordinate_AE_hidden,
                                        output_dim = DeepGCNModelAE_ResAdj_Coordinate_AE_hidden,
                                        adj = self.adj,
                                        act = tf.nn.relu,
                                        dropout = self.dropout,
                                        logging = self.logging)(self.hidden1)
        
        self.hidden2 = CoordinateOutput(act = lambda x: x,logging = self.logging)([self.hidden2_,self.hidden1_adj])

        self.z_mean = GraphConvolution(input_dim = DeepGCNModelAE_ResAdj_Coordinate_AE_hidden,
                                       output_dim = DeepGCNModelAE_ResAdj_Coordinate_AE_dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden2)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z_mean)

#------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------
class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = hidden,
                                             adj = self.adj,
                                             features_nonzero = self.features_nonzero,
                                             act = tf.nn.relu,
                                             dropout = self.dropout,
                                             logging = self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim = hidden,
                                       output_dim = dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden)

        self.z_log_std = GraphConvolution(input_dim = hidden,
                                          output_dim = dimension,
                                          adj = self.adj,
                                          act = lambda x: x,
                                          dropout = self.dropout,
                                          logging = self.logging)(self.hidden)

        self.z = self.z_mean + tf.random_normal([self.n_samples, dimension]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z)

#--------------------------------------------------------------------------------------------------------
class LinearModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(LinearModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):

        self.z_mean = GraphConvolutionSparse(input_dim = self.input_dim,
                                             output_dim = dimension,
                                             adj = self.adj,
                                             features_nonzero=self.features_nonzero,
                                             act = lambda x: x,
                                             dropout = self.dropout,
                                             logging = self.logging)(self.inputs)

        self.z_log_std = GraphConvolutionSparse(input_dim = self.input_dim,
                                                output_dim = dimension,
                                                adj = self.adj,
                                                features_nonzero = self.features_nonzero,
                                                act = lambda x: x,
                                                dropout = self.dropout,
                                                logging = self.logging)(self.inputs)

        self.z = self.z_mean + tf.random_normal([self.n_samples, dimension]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z)

#--------------------------------------------------------------------------------------------------------
class DeepGCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(DeepGCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim = self.input_dim,
                                              output_dim = hidden,
                                              adj = self.adj,
                                              features_nonzero = self.features_nonzero,
                                              act = tf.nn.relu,
                                              dropout = self.dropout,
                                              logging = self.logging)(self.inputs)

        self.hidden2 = GraphConvolution(input_dim = hidden,
                                        output_dim =hidden,
                                        adj = self.adj,
                                        act = tf.nn.relu,
                                        dropout = self.dropout,
                                        logging = self.logging)(self.hidden1)

        self.z_mean = GraphConvolution(input_dim = hidden,
                                       output_dim = dimension,
                                       adj = self.adj,
                                       act = lambda x: x,
                                       dropout = self.dropout,
                                       logging = self.logging)(self.hidden2)

        self.z_log_std = GraphConvolution(input_dim = hidden,
                                          output_dim = dimension,
                                          adj = self.adj,
                                          act = lambda x: x,
                                          dropout = self.dropout,
                                          logging = self.logging)(self.hidden2)

        self.z = self.z_mean + tf.random_normal([self.n_samples, dimension]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(act = lambda x: x,
                                                   logging = self.logging)(self.z)
