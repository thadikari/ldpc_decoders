import matplotlib.pyplot as plt
import numpy as np
import argparse

from models import models
import utils
import codes

x_labels = {'bsc': 'crossover probability',
            'bec': 'erasure probability',
            'biawgn': 'E_b/N in dB for E_b=1'}
lines = {'ML': 'b-', 'SPA': 'g--', 'LP': 'r-.'}


def main(args):
    saver = utils.Saver(args.data_dir, args.channel)
    data = saver.load(None)
    if data is None:
        print('No data available for:', args.channel)
        return

    for code in sorted(data.keys()):
        code_data = data[code]
        for decoder in sorted(code_data.keys()):
            pairs = code_data[decoder]
            pairs_ = list(zip(map(float, pairs.keys()),
                              pairs.values()))
            pairs_.sort(key=lambda x: x[0])
            plt.plot(*list(zip(*pairs_)), lines[decoder],
                     linewidth=3, label=decoder)

    if args.xlog: plt.xscale('log')
    plt.xlabel(x_labels[args.channel])
    plt.yscale('log')
    plt.ylabel('WER')
    plt.legend(loc='lower right')
    plt.grid(True, which='both')
    plt.show()


def setup_parser(code_names, channel_names, decoder_names):
    parser = argparse.ArgumentParser()
    # parser.add_argument('code', help='code', choices=code_names)
    parser.add_argument('channel', help='channel', choices=channel_names)
    parser.add_argument('--xlog', help='x-axis in log', action='store_true')
    # parser.add_argument('decoder', help='decoder', choices=decoder_names)
    return utils.bind_parser_common(parser)


if __name__ == "__main__":
    main(setup_parser(codes.get_code_names(), models.keys(), ['ML', 'SPA']).parse_args())
