import argparse
import json
import logging
import os
import unittest

import numpy as np

import codes
import collections
import time
import collections


def setup_parser(code_names, channel_names, decoder_names):
    parser = argparse.ArgumentParser()

    parser.add_argument('channel', help='channel', choices=channel_names)
    parser.add_argument('code', help='code', choices=code_names)
    parser.add_argument('decoder', help='decoder', choices=decoder_names)

    parser.add_argument('--params', nargs='+', type=float, default=[.1, .01])

    parser.add_argument('--codeword', help='-1:random from cb, 0:all-zero, 1:all-ones', default=0, type=int,
                        choices=[-1, 0, 1])
    parser.add_argument('--min-wec', help='min word errors to accumulate', default=100, type=int)
    parser.add_argument('--max-iter', help='max iterations in bp', default=10, type=int)
    parser.add_argument('--log-freq', help='log frequency in seconds', default=2., type=float)
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


def load_json(file_path):
    try:
        ff = open(file_path, 'r')
        data = json.load(ff, object_pairs_hook=collections.OrderedDict)
        ff.close()
    except:
        data = None
        print('Error loading: %s' % file_path)
    finally:
        return data


class Saver:
    def __init__(self, data_dir, run_ids):
        self.dict = collections.OrderedDict(run_ids)
        dir_path, file_name = data_dir, '-'.join(self.dict.values())
        self.file_path = os.path.join(dir_path, '%s.json' % file_name)

    def add(self, param, wer, ber):
        data = load_json(self.file_path)
        if data is None:
            data = collections.OrderedDict()
            for key in self.dict: data[key] = self.dict[key]
            data['wer'], data['ber'] = {}, {}

        data['wer'][str(param)] = wer
        data['ber'][str(param)] = ber
        with open(self.file_path, 'w') as fp:
            json.dump(data, fp, indent=4)

    def add_deprecated(self, run_id, val):
        data = self.load({})
        temp = data
        for key in run_id[0:-1]:
            if key not in temp.keys(): temp[key] = {}
            temp = temp[key]
        temp[run_id[-1]] = val

        with open(self.file_path, 'w') as ff:
            json.dump(data, ff)


class LoopProfiler:
    class Tag:
        def __init__(self, name, line, prof):
            self.name, self.line, self.prof = name, line, prof

        def elapsed(self):
            return (time.time() - self.updated) * 1000

        def __enter__(self):
            self.updated = time.time()
            extra = '' if self.line is None else ': ' + self.line
            self.prof.log.debug("(( '" + self.name + "'" + extra)
            return self

        def __exit__(self, type, value, traceback):
            elapsed = self.elapsed()
            self.prof.log.debug('    elapsed[%s] ))' % str(int(elapsed)))
            self.prof.tags[self.name] = self.prof.tags.get(self.name, 0) + elapsed

    def __init__(self, log, dump_freq):
        self.log = log
        self.updated = time.time()
        self.dump_freq = dump_freq
        self.tags = collections.OrderedDict()
        self.step_count = 0

    def __enter__(self):
        return self

    def start(self, line=None):
        self.step_count += 1
        if line is not None: self.log.debug(line)
        return self

    def tag(self, name, line=None):
        return LoopProfiler.Tag(name, line, self)

    def __exit__(self, typ, value, traceback):
        if self.dump_freq > 0 and self.step_count % self.dump_freq == 0:
            summary = ', '.join(["'%s':%d" % (key, int(val)) for key, val in self.tags.items()])
            self.log.info('Summary at[%d] for[%d]: [' % (self.step_count, self.dump_freq) + summary + ']')
            for key in self.tags: self.tags[key] = 0
