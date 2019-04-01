import argparse
import logging
import os
import re
import matplotlib
from models import models
import utils

x_labels = {'bsc': 'Crossover probability',
            'bec': 'Erasure probability',
            'biawgn': 'E_b/N in dB for E_b=1'}
lines = {'ML': 'b:', 'SPA': 'g--', 'MSA': 'r-.', 'LP': 'm-+', 'ADMM': 'k--'}
line_styles4 = ['b--', 'r-', 'g-.', 'm:']
line_styles = list(it1 + it2
                   for it1 in ['-', '--', '-.', ':']
                   for it2 in ['b', 'g', 'r'])


def plot_(pairs, style, label):
    pairs_ = list(zip(map(float, pairs.keys()), pairs.values()))
    pairs_.sort(key=lambda x: x[0])
    plt.plot(*list(zip(*pairs_)), style, linewidth=3, label=label)


def_title = lambda args: ', '.join((args.channel.upper(), args.code))


def graph_(args):
    log = logging.getLogger()
    data_list = []
    for file_name in utils.get_data_file_list(args.data_dir):
        data = utils.load_json(os.path.join(args.data_dir, file_name))
        if data['type'] != 'simulation': continue
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

    def extra_filter(it):
        if 'max_iter' in it.keys() and args.max_iter is not None:
            return int(it['max_iter']) == args.max_iter
        # elif 'eps' in it.keys() and args.eps is not None:
        #     return float(it['eps']) == args.eps
        # elif 'mu' in it.keys() and args.mu is not None:
        #     return float(it['mu']) == args.mu
        else:
            return True

    def get_first(ll, rsn):
        if len(ll) == 0:
            log.error('No matching data found for: %s.' % rsn)
            exit()
        else:
            return ll[0]

    prefix_code = lambda ar_: ar_.get('prefix', '') + ar_.get('code', '')
    prefix_or_code = lambda it, ar_: (it.get('code', '') == args.code or
                                      it.get('prefix', '') == args.code or
                                      it.get('code', '') == args.extra or
                                      it.get('prefix', '') == args.extra)

    if args.type == 'single':
        chk = lambda it: it.get('code', '') == args.code and \
                         it.get('decoder', '') == args.decoder[0] and \
                         extra_filter(it)
        data = get_first(filter_data(chk), 'single')
        plot_(data[args.error], 'k-', data['decoder'])
        title = def_title(args)

    elif args.type == 'compare':
        chk = lambda it: prefix_or_code(it, args) and \
                         it.get('decoder', '') == args.decoder[0] and \
                         extra_filter(it)
        for data, style in zip(filter_data(chk), line_styles4):
            plot_(data[args.error], style, prefix_code(data))
        title = args.channel.upper() + ', %s decoder' % args.decoder[0]

    elif args.type == 'comp_dec':
        chk = lambda it: prefix_or_code(it, args) and \
                         it.get('decoder', '') in args.decoder and \
                         extra_filter(it)
        filtered = filter_data(chk)
        same_code = len(set([prefix_code(data) for data in filtered])) <= 1  # check if all are for same code
        for data, style in zip(filtered, line_styles4):
            decoder = data['decoder']
            leg = decoder if same_code else '%s-%s' % (decoder, prefix_code(data))
            plot_(data[args.error], style, leg)
        title = def_title(args) if same_code else 'Comparison of decoders'

    elif args.type == 'ensemble':
        chk = lambda it: it.get('decoder', '') == args.decoder[0] and \
                         re.compile('^' + args.code + '_[0-9]+$'). \
                             match(it.get('code', ''))
        log.info('Matching ensemble codes')
        for data in filter_data(chk): plot_(data[args.error], 'r--', None)

        chk_avg = lambda it: 'sources' in it.keys() and \
                             it.get('prefix', '') == args.code and \
                             it.get('decoder', '') == args.decoder[0]
        log.info('Searching for average')
        plot_(get_first(filter_data(chk_avg), 'average')[args.error], 'b-', 'Average')
        title = def_title(args) + ' code ensemble' + ', %s decoder' % args.decoder[0]

    elif args.type == 'max_iter':
        chk = lambda it: it.get('code', '') == args.code and \
                         it.get('decoder', '') == args.decoder[0] and \
                         'max_iter' in it.keys()
        for data, style in zip(filter_data(chk, lambda it: int(it['max_iter'])), line_styles):
            decoder = data['decoder']
            plot_(data[args.error], style, data['max_iter'])
        title = def_title(args) + ', %s decoder' % args.decoder[0] + ', Effect of iterations cap'

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
    plt.margins(0)  # autoscale(tight=True)
    if args.save is not None:
        utils.make_dir_if_not_exists(args.plots_dir)
        img_path = os.path.join(args.plots_dir, args.save)
        plt.savefig(img_path, bbox_inches='tight')
    if not args.silent: plt.show()


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

    parser.add_argument('--max-iter', help='filter out multiple matches', type=int)
    parser.add_argument('--mu', help='mu', type=float)
    parser.add_argument('--eps', help='epsilon', type=float)
    parser.add_argument('--extra', help='code names to compare with in [type=compare] plots')

    parser.add_argument('--xlim', help='x-axis range', nargs=2, type=float)
    parser.add_argument('--ylim', help='y-axis range', nargs=2, type=float)
    parser.add_argument('--error', help='which error rate', default='ber', choices=['wer', 'ber'])
    parser.add_argument('--save', help='save as file name', type=str)
    parser.add_argument('--silent', help='do not show plot output', action='store_true')
    parser.add_argument('--agg', help='set matplotlib backend to Agg', action='store_true')

    return utils.bind_parser_common(parser)


def main(args):
    if args.agg: matplotlib.use('Agg')
    import matplotlib.pyplot

    global plt
    plt = matplotlib.pyplot
    graph_(args)


if __name__ == "__main__":
    utils.setup_console_logger()
    main(setup_parser().parse_args())
