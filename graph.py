import matplotlib.pyplot as plt
import argparse
import logging
import os
import re

from models import models
import utils
import codes

x_labels = {'bsc': 'Crossover probability',
            'bec': 'Erasure probability',
            'biawgn': 'E_b/N in dB for E_b=1'}
lines = {'ML': 'b-+', 'SPA': 'g--', 'LP': 'r-.'}


def is_valid_(data, args):
    if data is None: return 0
    if data.get('channel', '') != args.channel: return 0
    if data.get('decoder', '') not in args.decoder: return 0
    return 1


def is_valid(data, args, chk_lam):
    if not chk_lam(data, args): return 0
    return is_valid_(data, args)


def is_valid_avg(data, args):
    if 'sources' not in data.keys(): return 0
    if args.prefix:
        if args.code != data['prefix']: return 0
    else:
        if args.code not in data['sources']: return 0
    return is_valid_(data, args)


def plot_(pairs, decoder):
    pairs_ = list(zip(map(float, pairs.keys()),
                      pairs.values()))
    pairs_.sort(key=lambda x: x[0])
    plt.plot(*list(zip(*pairs_)),
             'k-' if decoder is None else lines[decoder],
             linewidth=3,
             label='Average' if decoder is None else decoder)


def main(args):
    log = logging.getLogger()
    file_list = utils.get_data_file_list(args.data_dir)
    if args.prefix:
        pattern = re.compile('^' + args.code + '_[0-9]+$')
        log.info('regex to match: ' + pattern.pattern)
        chk_lam = lambda data_, args_: pattern.match(data.get('code', ''))
    else:
        chk_lam = lambda data_, args_: data_.get('code', '') == args_.code

    for file_name in file_list:
        data = utils.load_json(os.path.join(args.data_dir, file_name))
        if is_valid(data, args, chk_lam):
            log.info('match: %s' % file_name)
            plot_(data[args.error], data['decoder'])
        if args.avg and is_valid_avg(data, args):
            log.info('AVG match: %s' % file_name)
            plot_(data[args.error], None)

    plt.xlabel(x_labels[args.channel])
    plt.yscale('log')
    plt.ylabel(args.error.upper())
    plt.legend(loc='lower right')
    plt.grid(True, which='both')
    title = ', '.join((args.channel.upper(), args.code))
    plt.title(title)
    if args.save:
        img_name = '-'.join((args.channel, args.code)) + '.png'
        img_path = os.path.join(args.data_dir, img_name)
        plt.savefig(img_path, bbox_inches='tight')
    plt.margins(0)  # autoscale(tight=True)
    plt.show()


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('channel', help='channel', choices=models.keys())
    parser.add_argument('code', help='code name')
    parser.add_argument('decoder', help='decoder', nargs='+', choices=utils.decoder_names)
    parser.add_argument('--prefix', help='is code a prefix? if yes will plot all matching codes', action='store_true')
    parser.add_argument('--error', help='which error rate', default='ber', choices=['wer', 'ber'])
    parser.add_argument('--avg', help='look for and plot average', action='store_true')
    parser.add_argument('--save', help='save as png', action='store_true')

    return utils.bind_parser_common(parser)


if __name__ == "__main__":
    utils.setup_console_logger()
    main(setup_parser().parse_args())
