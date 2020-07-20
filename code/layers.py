from inits import *
import tensorflow as tf

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


class GraphConvolution():
    """Basic graph convolution layer for undirected graph without edge labels."""

    def __init__(self, input_dim, output_dim, adj, name, dropout=0., act=tf.nn.relu, norm=False, is_train=False):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.norm = norm
        self.is_train = is_train

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = tf.nn.dropout(x, 1 - self.dropout)
            x = tf.matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
            if self.norm:
                outputs = tf.layers.batch_normalization(outputs, training=self.is_train)

        return outputs


class GraphConvolutionSparse():
    """Graph convolution layer for sparse inputs."""

    def __init__(self, input_dim, output_dim, adj, features_nonzero, name, dropout=0., act=tf.nn.relu, norm=False, is_train=False):
        self.name = name
        self.vars = {}
        self.issparse = False
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name='weights')
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero
        self.norm = norm
        self.is_train = is_train

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            x = inputs
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
            x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
            x = tf.sparse_tensor_dense_matmul(self.adj, x)
            outputs = self.act(x)
            if self.norm:
                outputs = tf.layers.batch_normalization(outputs, training=self.is_train)

        return outputs


class AttentionRec():
    """Attention merge layer for each support view"""

    def __init__(self, output_dim, num_support, name, dropout=0., act=tf.nn.sigmoid):
        self.num_nodes = output_dim
        self.num_support = num_support
        self.name = name
        self.dropout = dropout
        self.act = act
        self.attADJ = []
        with tf.variable_scope(self.name + '_attW'):
            self.attweights = tf.get_variable("attWeights", [self.num_support, self.num_nodes, self.num_nodes],
                                             initializer=tf.random_uniform_initializer(minval=0.9, maxval=1.1))
            self.attention = tf.nn.softmax(self.attweights, 0)


    def __call__(self, recs):
        with tf.name_scope(self.name):
            for i in range(self.num_support):
                self.attADJ.append(tf.multiply(self.attention[i], recs[i]))
            confiWeights = tf.add_n(self.attADJ)

            return confiWeights


class InnerProductDecoder():
    """Decoder model layer for link prediction."""

    def __init__(self, output_dim, name, dropout=0., act=tf.nn.sigmoid):
        self.name = name
        self.issparse = False
        self.vars = {}
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(output_dim, output_dim, name='weights')
        self.dropout = dropout
        self.act = act

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            inputs = tf.nn.dropout(inputs, 1 - self.dropout)
            x = tf.matmul(inputs, self.vars['weights'])
            x = tf.matmul(x, tf.transpose(inputs))
            outputs = self.act(x)
        return outputs

