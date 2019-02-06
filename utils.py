import argparse
import json
import logging
import os
import unittest

import numpy as np

import codes


def setup_parser(code_names, channel_names, decoder_names):
    parser = argparse.ArgumentParser()

    parser.add_argument('code', help='code', choices=code_names)
    parser.add_argument('channel', help='channel', choices=channel_names)
    parser.add_argument('decoder', help='decoder', choices=decoder_names)

    parser.add_argument('--params', nargs='+', type=float, default=[.1, .01, .001, .0001])

    parser.add_argument('--codeword', help='-1:random from cb, 0:all-zero, 1:all-ones', default=0, type=int,
                        choices=[-1, 0, 1])
    parser.add_argument('--min-wec', help='min word errors to accumulate', default=200, type=int)
    parser.add_argument('--max-iter', help='max iterations in bp', default=100, type=int)
    return bind_parser_common(parser)


def bind_parser_common(parser):
    parser.add_argument('--data-dir', help='data directory', default=os.path.join('..', 'data'))
    parser.add_argument('--debug', help='logs debug info', action='store_true')
    parser.add_argument('--console', help='prints log onto console', action='store_true')
    return parser


def setup_console_logger(level=logging.DEBUG):
    logging.basicConfig(format='%(name)s|%(message)s', level=level)


def setup_file_logger(path, name, level=logging.DEBUG):
    logging.basicConfig(filename=os.path.join(path, '%s.log' % name),
                        filemode='a',
                        format='%(asctime)s,%(msecs)03d|%(name)s|%(levelname)s|%(message)s',
                        datefmt='%H:%M:%S',
                        level=level)
    logging.info('Logger init to file. %s' % ('%' * 80))


class TestCase(unittest.TestCase):
    def sample(self, code, param, decoders, x, y):
        x_, y_ = np.array(x), np.array(y)
        for decoder in decoders:
            dec = decoder(param, codes.get_code(code))
            # print(dec, x_, dec.decode(y_))
            self.assertTrue((dec.decode(y_) == x_).all())
            # spa = SPA(param, codes.get_code(code))
            # self.assertTrue((spa.decode(y_) == x_).all())


def log_sum_exp_rows(arr):
    arr_max = arr.max(axis=1)
    return arr_max + np.log(np.exp(arr - arr_max[:, None]).sum(axis=1))
    # sum_terms_1 = np.array([[1, 2, 3], [6, -1, -6]])
    # print(sum_terms_1)
    # print(log_sum_exp_rows(sum_terms_1))


def arg_max_rand(values):
    max_ind = np.argwhere(values == np.max(values))
    return np.random.choice(max_ind.flatten(), 1)[0]


class Saver:
    def __init__(self, dir_path, file_name):
        self.file_path = os.path.join(dir_path, '%s.json' % file_name)

    def load(self, default):
        try:
            ff = open(self.file_path, 'r')
            data = json.load(ff)
            ff.close()
        except:
            data = default
        finally:
            return data

    def add(self, run_id, val):
        data = self.load({})
        temp = data
        for key in run_id[0:-1]:
            if key not in temp.keys(): temp[key] = {}
            temp = temp[key]
        temp[run_id[-1]] = val

        with open(self.file_path, 'w') as ff:
            json.dump(data, ff)
