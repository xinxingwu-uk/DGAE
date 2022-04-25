#--------------------------------------------------------------------------------------------------------
# Reproducible
import numpy as np
import random as rn
import os
from keras import backend as K
#import tensorflow as tf
#--------------------------------------------------------------------------------------------------------
import tensorflow.compat.v1  as tf
tf.disable_v2_behavior() 

#--------------------------------------------------------------------------------------------------------
def weight_variable_glorot(input_dim, output_dim, name = ""):
    seed=0
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    rn.seed(seed)
    session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    tf.compat.v1.set_random_seed(seed)
    #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #K.set_session(sess)   
    
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval = -init_range,maxval = init_range, dtype = tf.float32, seed=seed)
    return tf.Variable(initial, name = name)

#--------------------------------------------------------------------------------------------------------
def coordinate_parameter_initialize(name = ""):
    seed=0
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    rn.seed(seed)
    session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    tf.compat.v1.set_random_seed(seed)
    #sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    #K.set_session(sess)    

    initial = tf.random_uniform((), minval = 0,maxval = 1, dtype = tf.float32, seed=seed) 
    return tf.Variable(initial, name = name)