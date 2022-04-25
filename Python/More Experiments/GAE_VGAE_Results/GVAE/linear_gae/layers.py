from linear_gae.initializations import weight_variable_glorot, coordinate_parameter_initialize
import  tensorflow.compat.v1  as tf
tf.disable_v2_behavior() 
#import tensorflow as tf

_LAYER_UIDS = {}

#--------------------------------------------------------------------------------------------------------
def get_layer_uid(layer_name = ''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

#--------------------------------------------------------------------------------------------------------
def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

#--------------------------------------------------------------------------------------------------------
class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs

#--------------------------------------------------------------------------------------------------------
class GraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, adj, dropout = 0., act = tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1 - self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

#--------------------------------------------------------------------------------------------------------
class GraphConvolutionSparse(Layer):
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout = 0., act = tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name = "weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs

#--------------------------------------------------------------------------------------------------------
class InnerProductDecoder(Layer):
    def __init__(self, dropout = 0., act = tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1 - self.dropout)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs
    
#---------------------------------------------------------------------------------------------------------
class CoordinateOutput(Layer):
    def __init__(self, dropout = 0., act = tf.nn.relu, **kwargs):
        super(CoordinateOutput, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['coordinateparameter'] = coordinate_parameter_initialize(name="coordinateparameter")
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        inputs_0 = tf.nn.dropout(inputs[0], 1 - self.dropout)
        inputs_1 = tf.nn.dropout(inputs[1], 1 - self.dropout)
        x=self.vars['coordinateparameter'] *inputs_0+(1-self.vars['coordinateparameter'] )*inputs_1
        outputs = self.act(x)
        return outputs
    
#-------------------------------------------------------------------------------------------------------- 
class TraditionalLayer(Layer):
    def __init__(self, input_dim, output_dim,dropout=0., act=tf.nn.relu, **kwargs):
        super(TraditionalLayer, self).__init__(**kwargs)
        
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act  

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        outputs = self.act(x)
        return outputs    

#--------------------------------------------------------------------------------------------------------
class TraditionalLayer_SparseInput(Layer):
    def __init__(self, input_dim, output_dim,features_nonzero,dropout=0., act=tf.nn.relu, **kwargs):
        super(TraditionalLayer_SparseInput, self).__init__(**kwargs)
        
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.act = act  
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])#+self.vars['bias']
        outputs = self.act(x)
        return outputs
