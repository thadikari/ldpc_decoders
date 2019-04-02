import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

import utils


def plot_file(args):
    for file_name in utils.get_data_file_list(args.data_dir):
        data = utils.load_json(os.path.join(args.data_dir, file_name))
        if 'type' not in data.keys() or data['type'] != 'luby': continue
        if float(data['c']) in args.c:
            plot_hist(data['arr'], int(data['k']),
                      int(data['n']), float(data['c']))


def plot_hist(arr, k, n, c):
    title = 'c=%g, mean=%g, std_dev=%g, var=%g' \
            % (c, np.mean(arr), np.std(arr), np.var(arr))
    plt.hist(arr, bins=50)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title(title)
    plt.xlim(k, n)
    plt.show()


def plot_avg_deg(ll_c, avg_deg):
    plt.plot(ll_c, avg_deg)
    plt.show()


def plot_soliton(rho, tau, mu, c, cut):
    bar_width = 0.32
    average = lambda dst: dst @ np.arange(1, len(dst) + 1)
    bar_plt = lambda ind, dst, name, clr: \
        plt.bar(np.arange(1, cut + 1) + bar_width * ind,
                dst[:cut], bar_width, linewidth=0, color=clr,
                label='%s, avg_deg=%g' % (name, average(dst)))

    bar_plt(0, rho, 'rho', 'r')
    bar_plt(1, tau, 'tau', 'b')
    bar_plt(2, mu, 'mu', 'y')

    plt.autoscale(enable=True, axis='x', tight=True)
    plt.title('c=%g' % c)
    plt.legend()
    plt.show()


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('c', nargs='+', type=float)
    return utils.bind_parser_common(parser)


if __name__ == "__main__":
    plot_file((setup_parser().parse_args()))
    # Usage: python src/luby_graph.py .01 .03 .1
