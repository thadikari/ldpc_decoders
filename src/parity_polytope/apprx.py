import tensorflow as tf
import numpy as np


def bi(size):
    with tf.variable_scope(tf.get_default_graph().get_name_scope()):  # , reuse=tf.AUTO_REUSE):
        return tf.get_variable('var_b', size, initializer=tf.zeros_initializer)


def wi(shape):
    with tf.variable_scope(tf.get_default_graph().get_name_scope()):  # , reuse=tf.AUTO_REUSE):
        return tf.get_variable('var_W', shape)


def create_layer(acts_p, dim_l, actvn_fn):
    dim_lp = acts_p.get_shape().as_list()[1]
    W, b = wi((dim_lp, dim_l)), bi(dim_l)
    logits = tf.matmul(acts_p, W) + b
    return actvn_fn(logits), logits, [W, b]


class ex_name_scope(object):
    def __init__(self, name):
        cur = tf.get_default_graph().get_name_scope()
        self.ns = tf.name_scope(cur + ('' if len(cur) == 0 else '/') + name + '/')

    def __enter__(self):
        self.ns.__enter__()
        return self.ns

    def __exit__(self, type, value, traceback):
        self.ns.__exit__(type, value, traceback)


def create_fcnet(acts_p, layers, inner_actvn_fn, last_actvn_fn):
    theta_lst = []
    for l in range(len(layers)):
        with ex_name_scope('layer_%d' % (l + 1)):
            acts, logits, theta = create_layer(acts_p, layers[l], inner_actvn_fn)
            theta_lst.extend(theta)
            acts_p = acts
    return last_actvn_fn(logits), logits, theta_lst


class Model:
    def __init__(self, dim, layers):
        self.X = tf.placeholder(tf.float32, shape=[None, dim], name='X')
        self.Y = tf.placeholder(tf.float32, shape=[None, dim], name='Y')

        # TODO: replace tf.nn.sigmoid and loss func w/ ones that suit better
        self.Y_hat, Y_hat_logits, self.weights = create_fcnet(self.X, layers + [dim], tf.nn.relu, tf.nn.sigmoid)
        # Y_hat, Y_hat_logits, self.weights = create_fcnet(self.X, layers + [dim_vec], tf.identity, tf.identity)
        self.dim = dim

    def setup_train(self):
        self.loss = tf.reduce_mean(tf.square(self.Y_hat - self.Y))
        start_rate = .001
        self.opt = tf.train.AdamOptimizer(start_rate).minimize(loss=self.loss)
        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

    def step(self, X, Y):
        self.sess.run(self.opt, feed_dict={self.X: X, self.Y: Y})

    def eval_rows(self, X):
        return self.sess.run(self.Y_hat, feed_dict={self.X: X})

    def eval_vec(self, X):
        return self.eval_rows(X[np.newaxis, :])[0, :]

    def eval_loss(self, X, Y):
        return self.sess.run(self.loss, feed_dict={self.X: X, self.Y: Y})

    def path(self):
        return '../../cache/model_%d.ckpt' % self.dim

    def save(self, saver):
        saver.save(self.sess, self.path())

    def restore(self, saver):
        self.sess = tf.Session()
        saver.restore(self.sess, self.path())


def make_model(dim):
    return Model(dim, [20, 20])


def load_model(dim):
    model = make_model(dim)
    model.restore(tf.train.Saver())
    return model
