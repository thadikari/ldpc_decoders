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
lines = {'ML': 'b-+', 'SPA': 'g--', 'MSA': 'r-.'}
line_styles2 = ['b--', 'r-']
line_styles = list(it1 + it2
                   for it1 in ['-', '--', '-.', ':']
                   for it2 in ['b', 'g', 'r'])


def plot_(pairs, style, label):
    pairs_ = list(zip(map(float, pairs.keys()), pairs.values()))
    pairs_.sort(key=lambda x: x[0])
    plt.plot(*list(zip(*pairs_)), style, linewidth=3, label=label)


def_title = lambda args: ', '.join((args.channel.upper(), args.code))


def main(args):
    log = logging.getLogger()
    data_list = []
    for file_name in utils.get_data_file_list(args.data_dir):
        data = utils.load_json(os.path.join(args.data_dir, file_name))
        if data['channel'] == args.channel: data_list.append((file_name, data))

    def filter_data(expr, comp=None):
        ll = []
        for name, item in data_list:
            # print('filter:', name)
            if expr(item):
                log.info('Match: %s' % name)
                ll.append(item)
        if comp is not None: ll.sort(key=comp)
        return ll

    if args.type == 'single':
        chk = lambda it: it.get('code', '') == args.code and \
                         it.get('decoder', '') == args.decoder[0] and \
                         'max_iter' not in it.keys()
        data = filter_data(chk)[0]
        plot_(data[args.error], 'k-', data['decoder'])
        title = def_title(args)

    elif args.type == 'compare':
        chk = lambda it: (it.get('code', '') == args.code or \
                          it.get('prefix', '') == args.code or \
                          it.get('code', '') == args.extra or \
                          it.get('prefix', '') == args.extra) and \
                         it.get('decoder', '') == args.decoder[0] and \
                         'max_iter' not in it.keys()
        for data, style in zip(filter_data(chk), line_styles2):
            plot_(data[args.error], style,
                  data.get('prefix', '') + data.get('code', ''))
        title = args.channel.upper() + ', %s decoder' % args.decoder[0]

    elif args.type == 'comp_dec':
        chk = lambda it: it.get('code', '') == args.code and \
                         it.get('decoder', '') in args.decoder and \
                         'max_iter' not in it.keys()
        for data in filter_data(chk):
            decoder = data['decoder']
            plot_(data[args.error], lines[decoder], decoder)
        title = def_title(args)

    elif args.type == 'ensemble':
        chk = lambda it: it.get('decoder', '') == args.decoder[0] and \
                         re.compile('^' + args.code + '_[0-9]+$'). \
                             match(it.get('code', ''))
        log.info('Matching ensemble codes')
        for data in filter_data(chk): plot_(data[args.error], 'r--', None)

        chk_avg = lambda it: 'sources' in it.keys() and \
                             it.get('prefix', '') == args.code
        log.info('Searching for average')
        plot_(filter_data(chk_avg)[0][args.error], 'b-', 'Average')
        title = def_title(args) + ' code ensemble' + ', %s decoder' % args.decoder[0]

    elif args.type == 'max_iter':
        chk = lambda it: it.get('code', '') == args.code and \
                         it.get('decoder', '') == args.decoder[0] and \
                         'max_iter' in it.keys()
        for data, style in zip(filter_data(chk, lambda it: int(it['max_iter'])), line_styles):
            decoder = data['decoder']
            plot_(data[args.error], style, data['max_iter'])
        title = def_title(args) + ', %s decoder' % args.decoder[0] + ', Error as a function of iterations cap'

    else:
        return

    plt.xlabel(x_labels[args.channel])
    plt.yscale('log')
    plt.ylabel(args.error.upper())
    plt.legend(loc='best')
    plt.grid(True, which='both')
    if args.xlim is not None: plt.xlim(args.xlim)
    if args.ylim is not None: plt.ylim(args.ylim)
    plt.title(title)
    if args.save:
        img_name = '-'.join((args.channel, args.code)) + '.png'
        img_path = os.path.join(args.data_dir, img_name)
        plt.savefig(img_path, bbox_inches='tight')
    plt.margins(0)  # autoscale(tight=True)
    plt.show()


def setup_parser():
    # https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    parser = argparse.ArgumentParser()
    parser.add_argument('channel', help='channel', choices=models.keys())
    parser.add_argument('code', help='code name OR prefix')
    parser.add_argument('decoder', help='decoders list', nargs='+', choices=utils.decoder_names)
    parser.add_argument('type', help='plot type',
                        choices=['single',  # plot of a code, channel, decoder
                                 'comp_dec',  # compare multiple decoders
                                 'ensemble',  # average and ensemble
                                 'max_iter',  # compare iterations cap
                                 'compare'  # compare with another
                                 ])
    parser.add_argument('--extra', help='code names to compare with')
    parser.add_argument('--xlim', help='x-axis range', nargs=2, type=float)
    parser.add_argument('--ylim', help='y-axis range', nargs=2, type=float)
    parser.add_argument('--error', help='which error rate', default='ber', choices=['wer', 'ber'])
    parser.add_argument('--save', help='save as png', action='store_true')

    return utils.bind_parser_common(parser)


if __name__ == "__main__":
    utils.setup_console_logger()
    main(setup_parser().parse_args())
