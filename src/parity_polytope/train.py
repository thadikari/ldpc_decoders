import tensorflow as tf
import numpy as np

import exact
import apprx


def reset_all(seed=0):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)


def gen_data(count, dim):
    X = np.random.rand(count, dim)
    Y = exact.proj_rows(X)
    return X, Y


def test(dim):
    model = apprx.load_model(dim)
    if dim == 6:
        data = [.5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 1, 1]
        arr = np.array(data).reshape(-1, 6)
        print(model.eval_rows(arr).ravel())
    elif dim == 2:
        print(model.eval_vec(np.array((1., .4))))
    else:
        test_data = gen_data(100, dim)
        print(model.eval_loss(*test_data))


class Trainer:
    def __init__(self, model):
        self.model = model
        self.loss = tf.reduce_mean(tf.square(model.Y_hat - model.Y))
        start_rate = .001
        self.opt = tf.train.AdamOptimizer(start_rate).minimize(loss=self.loss)
        init_op = tf.global_variables_initializer()
        self.sess = model.sess
        self.sess.run(init_op)

    def step(self, X, Y):
        self.sess.run(self.opt, feed_dict={self.model.X: X, self.model.Y: Y})

    def eval_loss(self, X, Y):
        return self.sess.run(self.loss, feed_dict={self.model.X: X, self.model.Y: Y})


def train(dim):
    model = apprx.make_model(dim)
    trainer = Trainer(model)
    saver = tf.train.Saver()
    test_data = gen_data(100, dim)
    for step in range(0, 10000):
        trainer.step(*gen_data(1000, dim))
        if step % 100 == 0:
            print(['step', step, 'loss', trainer.eval_loss(*test_data)])

    model.save(saver)


if __name__ == '__main__':
    reset_all()
    # train(6)
    test(6)
