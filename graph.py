import matplotlib.pyplot as plt
import numpy as np
import argparse

from models import models
from utils import Saver
import codes

x_labels = {'bsc': 'crossover probability',
            'bec': 'erasure probability',
            'biawgn': 'noise variance'}


def main(args):
    saver = Saver('./', args.channel)
    data = saver.load(None)
    if data is None:
        print('No data available for:', args.channel)
        return

    func = np.log if args.xlog else (lambda val: val)
    for code in data.keys():
        code_data = data[code]
        for decoder in code_data.keys():
            pairs = code_data[decoder]
            pairs_ = list(zip(map(lambda x: func(float(x)), pairs.keys())
                              , map(np.log, pairs.values())))
            pairs_.sort(key=lambda x: x[0])
            plt.plot(*list(zip(*pairs_)))

    plt.xlabel(('log(%s)' if args.xlog else '%s')
               % x_labels[args.channel])
    plt.ylabel('log(WER)')
    plt.show()


def setup_parser(code_names, channel_names, decoder_names):
    parser = argparse.ArgumentParser()
    # parser.add_argument('code', help='code', choices=code_names)
    parser.add_argument('channel', help='channel', choices=channel_names)
    parser.add_argument('--xlog', help='x-axis in log', action='store_true')
    # parser.add_argument('decoder', help='decoder', choices=decoder_names)
    return parser


if __name__ == "__main__":
    main(setup_parser(codes.get_code_names(), models.keys(), ['ML', 'SPA']).parse_args())
