from __future__ import division
from __future__ import print_function

import time
import os

from sklearn.model_selection import KFold

from objective import *
from metrics import *
from models import *
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# tensorflow config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('log_file', "log/SLMGAE_BC.txt", 'log file name.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('eva_epochs', 100, 'Number of epochs to evaluate')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('nn_size', 45, 'Number of K for the KNN')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('Alpha', 0.5, 'Coefficient of support view loss.')
flags.DEFINE_float('Coe', 1.0, 'Coefficient of support view loss.')
flags.DEFINE_float('Beta', 2.0, 'Coefficient of final loss.')
flags.DEFINE_integer('early_stopping', 20, 'Tolerance for early stopping (# of epochs).')

pos_edge, neg_edge, adjs_orig = load_BC_data()

np.random.shuffle(pos_edge)
np.random.shuffle(neg_edge)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = sp.csr_matrix((np.ones(len(pos_edge)), (pos_edge[:, 0], pos_edge[:, 1])), shape=(332, 332))
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

# K-Fold Test
k_round = 0
kf = KFold(n_splits=5, shuffle=True, random_state=123)
pos_edge_kf = kf.split(pos_edge)
neg_edge_kf = kf.split(neg_edge)

auc_pair, aupr_pair, f1_pair, train_time = [], [], [], []
training_loss, testing_loss = [], []
for train_pos, test_pos in pos_edge_kf:
    _, test_neg = next(neg_edge_kf)

    tf.reset_default_graph()
    k_round += 1
    print("Training in the %02d fold..." % k_round)

    row = pos_edge[train_pos, 0]
    col = pos_edge[train_pos, 1]
    val = np.ones(len(train_pos))
    adj = sp.csr_matrix((val, (row, col)), shape=(332, 332))
    adj = adj + adj.T
    adjs = adjs_orig[0: 5]
    adjs.append(adj)

    num_nodes = adj.shape[0]
    num_edges = adj.sum()

    # build test set
    test_edges = pos_edge[test_pos]
    neg_test_edges = neg_edge[test_neg[:len(test_pos)]]
    test_set = np.vstack((test_edges, neg_test_edges))

    x, y = np.triu_indices(num_nodes, k=1)
    test_x, test_y = test_set[:, 0], test_set[:, 1]
    train_index = set(zip(x, y)) - set(zip(test_x, test_y))
    train_neg_index = train_index - set(zip(row, col))

    train_index = np.array(list(train_index))
    train_neg_index = np.array(list(train_neg_index))

    # build features
    features = sparse_to_tuple(adj)
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]

    supports = []
    for a in adjs:
        supports.append(preprocess_graph(a))
    num_supports = len(supports)

    placeholders = {
            'support': [tf.sparse_placeholder(tf.float32, name='adj_{}'.format(_)) for _ in range(num_supports)],
            'features': tf.sparse_placeholder(tf.float32, name='features'),
            'adj_orig': tf.sparse_placeholder(tf.float32, name='adj_orig'),
            'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        }

    # Create model
    model = SLMGAE_BC(placeholders, num_features, features_nonzero, num_nodes, num_supports - 1,
                   name=f'SLMGAE_{k_round}')

    # Create optimizer
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            supp=model.support_recs,
            main=model.main_rec,
            preds=model.reconstructions,
            labels=tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False),
            num_nodes=num_nodes,
            num_edges=num_edges,
            index=train_index
        )

    # Initialize session
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    adj_label = sparse_to_tuple(adj_orig)

    # Construct feed dictionary
    feed_dict = construct_feed_dict(supports, features, adj_label, placeholders)

    # Train model
    eva_score, cost_val, epoch = [], [], 0

    tt = time.time()
    for epoch in range(FLAGS.epochs):
        t = time.time()

        feed_dict = construct_feed_dict(supports, features, adj_label, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # One update of parameter matrices
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)

        cost_val.append(avg_cost)

        print("Epoch: " + '%04d' % (epoch + 1) +
              " train_loss=" + "{:.5f}".format(avg_cost) +
              " time= " + "{:.5f}".format(time.time() - t))

    train_time.append(time.time() - tt)
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    adj_rec = sess.run(model.predict(), feed_dict=feed_dict)

    # roc, aupr, f1 = evalution_cmf(adj_rec, test_set)
    roc, aupr, f1 = evalution_bal(adj_rec, test_edges, neg_test_edges)
    eva_score.append([(epoch + 1), roc, aupr, f1, train_time[-1]])

    print("Test by Evalution:")
    for e in eva_score:
        print('Train_Epoch = %04d,'
              ' Test ROC score: %5f,'
              ' Test AUPR score:%5f,'
              ' Test F1 score: %5f' % (e[0], e[1], e[2], e[3]))

    auc_pair.append(eva_score[-1][1])
    aupr_pair.append(eva_score[-1][2])
    f1_pair.append(eva_score[-1][3])

m1, sdv1 = mean_confidence_interval(auc_pair)
m2, sdv2 = mean_confidence_interval(aupr_pair)
m3, sdv3 = mean_confidence_interval(f1_pair)
m4, sdv4 = mean_confidence_interval(train_time)

log("Average metrics over pairs:\n" +
    " time_mean:%.5f, time_sdv:%.5f\n" % (m4, sdv4) +
    " auc_mean:%.5f, auc_sdv:%.5f\n" % (m1, sdv1) +
    " aupr_mean:%.5f, aupr_sdv:%.5f\n" % (m2, sdv2) +
    " f1_mean: %.5f, f1_sdv: %.5f\n" % (m3, sdv3))

# print(adj_rec[test_set[:, 0], test_set[:, 1]])