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
    test_data = gen_data(100, dim)
    # print(model.eval_loss(*test_data))
    print(model.eval_vec(np.array((1., .4))))


def train(dim):
    model = apprx.make_model(dim)
    saver = tf.train.Saver()
    model.setup_train()
    test_data = gen_data(100, dim)
    for step in range(0, 10000):
        model.step(*gen_data(1000, dim))
        if step % 100 == 0:
            print(['step', step, 'loss', model.eval_loss(*test_data)])

    model.save(saver)


if __name__ == '__main__':
    reset_all()
    # train(3)
    test(2)
