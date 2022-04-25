import  tensorflow.compat.v1  as tf
tf.disable_v2_behavior() 
#import tensorflow as tf

learning_rate=0.01

#--------------------------------------------------------------------------------------------------------
class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                     targets = labels_sub,
                                                     pos_weight = pos_weight))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = \
            tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                     tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
#--------------------------------------------------------------------------------------------------------------------
class OptimizerAE_FeatureReconstrution(object):
    def __init__(self, preds_adj,preds_Features,labels_adj,labels_Features,loss_coeff_adj,loss_coeff_AE,pos_weight,norm):
        preds_sub_adj = preds_adj
        labels_sub_adj = labels_adj

        preds_sub_Features=preds_Features
        labels_sub_Features=labels_Features
        
        rec_features_prediction =tf.square(tf.subtract(preds_sub_Features, tf.sparse.to_dense(labels_sub_Features)))     
        self.cost_features = tf.reduce_mean(rec_features_prediction)
        
        self.cost_adj = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub_adj, targets=labels_sub_adj,pos_weight=pos_weight))
                       
        self.cost=loss_coeff_AE*self.cost_features+loss_coeff_adj*self.cost_adj
        
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub_adj), 0.5), tf.int32),tf.cast(labels_sub_adj, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
#--------------------------------------------------------------------------------------------------------------------
class OptimizerAE_AdjReconstrution(object):
    def __init__(self, preds_adj,preds_adj_rec,labels_adj,labels_adj_rec,loss_coeff_adj,loss_coeff_AE,pos_weight,norm):
        preds_sub_adj = preds_adj
        labels_sub_adj = labels_adj

        preds_sub_adj_rec=preds_adj_rec
        labels_sub_adj_rec=labels_adj_rec
            
        rec_adj_prediction = tf.square(tf.subtract(preds_sub_adj_rec, tf.sparse.to_dense(labels_sub_adj_rec)))     
        self.cost_adj_rec = tf.reduce_mean(rec_adj_prediction)
        
        #self.cost_adj_rec=norm *tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub_adj_rec, targets=tf.sparse.to_dense(labels_sub_adj_rec),pos_weight=pos_weight))
        
        self.cost_adj = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub_adj, targets=labels_sub_adj,pos_weight=pos_weight))
                       
        self.cost=loss_coeff_AE*self.cost_adj_rec+loss_coeff_adj*self.cost_adj
        
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub_adj), 0.5), tf.int32),tf.cast(labels_sub_adj, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
            
#--------------------------------------------------------------------------------------------------------            
class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits = preds_sub,
                                                     targets = labels_sub,
                                                     pos_weight = pos_weight))
        # Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * \
                  tf.reduce_mean(tf.reduce_sum(1 \
                                               + 2 * model.z_log_std \
                                               - tf.square(model.z_mean) \
                                               - tf.square(tf.exp(model.z_log_std)), 1))
        self.cost -= self.kl
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)
        self.correct_prediction = \
            tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                              tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
