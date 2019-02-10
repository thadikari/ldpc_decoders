import matplotlib.pyplot as plt
import argparse
import logging
import os

from models import models
import utils
import codes

x_labels = {'bsc': 'crossover probability',
            'bec': 'erasure probability',
            'biawgn': 'E_b/N in dB for E_b=1'}
lines = {'ML': 'b-', 'SPA': 'g--', 'LP': 'r-.'}


def is_valid(data, args):
    if data is None: return 0
    if data.get('channel', '') != args.channel: return 0
    if data.get('code', '') != args.code: return 0
    if data.get('decoder', '') not in args.decoder: return 0
    return 1


def main(args):
    log = logging.getLogger()
    file_list = tuple(it for it in next(os.walk(args.data_dir))[2]
                      if os.path.splitext(it)[1] == '.json')

    for file_name in file_list:
        data = utils.load_json(os.path.join(args.data_dir, file_name))
        if is_valid(data, args):
            log.info('file: %s' % file_name)
            decoder = data['decoder']
            # if args.wer:
            pairs = data['wer']
            pairs_ = list(zip(map(float, pairs.keys()),
                              pairs.values()))
            pairs_.sort(key=lambda x: x[0])
            plt.plot(*list(zip(*pairs_)), lines[decoder],
                     linewidth=3, label=decoder)

    plt.xlabel(x_labels[args.channel])
    plt.yscale('log')
    plt.ylabel('WER')
    plt.legend(loc='lower right')
    plt.grid(True, which='both')
    plt.show()


def setup_parser(code_names, channel_names, decoder_names):
    parser = argparse.ArgumentParser()
    parser.add_argument('channel', help='channel', choices=channel_names)
    parser.add_argument('code', help='code', choices=code_names)
    # parser.add_argument('--xlog', help='x-axis in log', action='store_true')
    parser.add_argument('decoder', help='decoder', nargs='+', default=['SPA'], choices=decoder_names)

    parser.add_argument('--wer', help='plot wer', action='store_true')
    parser.add_argument('--ber', help='plot ber', action='store_true')

    return utils.bind_parser_common(parser)


if __name__ == "__main__":
    utils.setup_console_logger()
    main(setup_parser(codes.get_code_names(), models.keys(), ['ML', 'SPA']).parse_args())
