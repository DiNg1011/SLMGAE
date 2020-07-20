from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Optimizer():
    def __init__(self, supp, main, preds, labels, num_nodes, num_edges, index):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            labels_sub = tf.gather_nd(labels, index)
            main_sub = tf.gather_nd(main, index)
            preds_sub = tf.gather_nd(preds, index)

            self.loss_supp = 0
            for viewRec in supp:
                viewRec_sub = tf.gather_nd(viewRec, index)
                self.loss_supp += tf.compat.v1.keras.losses.MSE(labels_sub, viewRec_sub)

            self.loss_main = tf.compat.v1.keras.losses.MSE(labels_sub, main_sub)

            self.loss_preds = tf.compat.v1.keras.losses.MSE(labels_sub, preds_sub)

            self.cost = FLAGS.Alpha * self.loss_supp + \
                        FLAGS.Beta * self.loss_preds + \
                        1 * self.loss_main

            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

            self.opt_op = self.optimizer.minimize(self.cost)
            self.grads_vars = self.optimizer.compute_gradients(self.cost)
