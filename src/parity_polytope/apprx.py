import tensorflow as tf
import numpy as np
import logging
import os


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
        self.name = '-'.join((str(i) for i in [dim] + layers + [dim]))
        self.sess = tf.Session()

    def eval_rows(self, X):
        return self.sess.run(self.Y_hat, feed_dict={self.X: X})

    def eval_vec(self, X):
        return self.eval_rows(X[np.newaxis, :])[0, :]

    def path(self):
        dir_name = os.path.dirname(__file__)
        return os.path.join(dir_name, '..', '..', 'cache', 'model_%s.ckpt' % self.name)

    def save(self, saver):
        saver.save(self.sess, self.path())

    def restore(self, saver):
        saver.restore(self.sess, self.path())


class Trainer:
    def __init__(self, model, save_freq):
        self.log = logging.getLogger('Trainer')
        self.model = model
        self.loss = tf.reduce_mean(tf.square(model.Y_hat - model.Y))
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.loss)

        init_op = tf.global_variables_initializer()
        self.sess = model.sess
        self.sess.run(init_op)
        self.saver = tf.train.Saver()

        self.save_freq = save_freq
        self.step_count = 0

    def save(self):
        self.model.save(self.saver)

    def step(self, X, Y):
        rate = 0.001  # / 10 ** (int(self.step_count / 10000))
        self.sess.run(self.opt, feed_dict={self.model.X: X, self.model.Y: Y, self.learning_rate: rate})
        self.step_count += 1
        if self.step_count % self.save_freq == 0:
            # print(rate)
            loss = self.sess.run(self.loss, feed_dict={self.model.X: X, self.model.Y: Y})
            self.log.info('Saving at step=%d, rate=%g, loss=%g' % (self.step_count, rate, loss))
            self.save()

    def eval_loss(self, X, Y):
        return self.sess.run(self.loss, feed_dict={self.model.X: X, self.model.Y: Y})


def make_model(dim, layers):
    return Model(dim, layers)


def load_model(dim, layers):
    tf.reset_default_graph()
    model = make_model(dim, layers)
    model.restore(tf.train.Saver())
    return model
