import tensorflow as tf
import numpy as np
import logging

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


def test(dim, layers):
    model = apprx.load_model(dim, layers)
    trainer = apprx.Trainer(model)
    if dim == 6:
        data = [.5, .5, .5, .5, .5, .5, 0, 0, 0, 0, 1, 1]
        arr = np.array(data).reshape(-1, 6)
        print(model.eval_rows(arr).ravel())
    elif dim == 2:
        print(model.eval_vec(np.array((1., .4))))
    else:
        test_data = gen_data(100, dim)
        print(trainer.eval_loss(*test_data))


def train(dim, layers):
    model = apprx.make_model(dim, layers)
    trainer = apprx.Trainer(model, 500)
    test_data = gen_data(100, dim)
    for step in range(0, 10000):
        trainer.step(*gen_data(1000, dim))
        if step % 100 == 0:
            print(['step', step, 'loss', trainer.eval_loss(*test_data)])

    trainer.save()


if __name__ == '__main__':
    reset_all()
    logging.basicConfig(format='%(name)s|%(message)s', level=logging.DEBUG)
    train(3, [200, 200, 200])
    # test(4, [100, 100])
