from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class SLMGAE():
    def __init__(self, placeholders, num_features, features_nonzero, num_nodes, num_supView, name):
        self.name = name
        self.num_nodes = num_nodes
        self.num_supView = num_supView
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adjs = placeholders['support']
        self.dropout = placeholders['dropout']
        self.inputs = placeholders['features']
        self.support_recs = []

        with tf.variable_scope(self.name):
            self.attentionLayer = AttentionRec(
                name='Attention_Layer',
                output_dim=self.num_nodes,
                num_support=self.num_supView,
                act=lambda x: x)

            self.build()

    def build(self):
        self.hidden1 = GraphConvolutionSparse(
            name='gcn_sparse_layer1',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adjs[0],
            features_nonzero=self.features_nonzero,
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.inputs)

        self.hidden2 = GraphConvolutionSparse(
            name='gcn_sparse_layer2',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adjs[1],
            features_nonzero=self.features_nonzero,
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.inputs)

        self.hidden3 = GraphConvolutionSparse(
            name='gcn_sparse_layer3',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adjs[2],
            features_nonzero=self.features_nonzero,
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.inputs)

        self.hidden4 = GraphConvolutionSparse(
            name='gcn_sparse_layer3',
            input_dim=self.input_dim,
            output_dim=FLAGS.hidden1,
            adj=self.adjs[3],
            features_nonzero=self.features_nonzero,
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.inputs)

        self.hidden5 = GraphConvolution(
            name='gcn_dense_layer1',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adjs[0],
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.hidden1)

        self.hidden6 = GraphConvolution(
            name='gcn_dense_layer2',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adjs[1],
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.hidden2)

        self.hidden7 = GraphConvolution(
            name='gcn_dense_layer3',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adjs[2],
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.hidden3)

        self.hidden8 = GraphConvolution(
            name='gcn_dense_layer3',
            input_dim=FLAGS.hidden1,
            output_dim=FLAGS.hidden2,
            adj=self.adjs[3],
            act=tf.nn.leaky_relu,
            dropout=self.dropout)(self.hidden4)

        self.support_recs.append(InnerProductDecoder(
                name='gcn_decoder1',
                output_dim=FLAGS.hidden2,
                act=lambda x: x)(self.hidden5))

        self.support_recs.append(InnerProductDecoder(
                name='gcn_decoder2',
                output_dim=FLAGS.hidden2,
                act=lambda x: x)(self.hidden6))

        self.support_recs.append(InnerProductDecoder(
                name='gcn_decoder3',
                output_dim=FLAGS.hidden2,
                act=lambda x: x)(self.hidden7))

        # self.att = tf.reduce_mean(self.support_recs)
        self.att = self.attentionLayer(self.support_recs)

        self.main_rec = InnerProductDecoder(
                name='gcn_decoder_main',
                output_dim=FLAGS.hidden2,
                act=lambda x: x)(self.hidden8)

        self.reconstructions = tf.add(self.main_rec, tf.multiply(FLAGS.Coe, self.att))

    def predict(self):
        return self.reconstructions

