import numpy as np
import matplotlib
import argparse
import os
import re

import utilities as ut
import utilities.mpl
import utils


ut.mpl.init(font_size=12, legend_font_size=12, tick_size=12)

legend_reg = ut.Registry()
r_ = legend_reg.put
r_('decoder', lambda d_: d_['decoder'])
r_('channel_decoder', lambda d_: d_['channel'].upper() + ', %s decoder' % d_['decoder'])
r_('channel_code', lambda d_: d_['channel'].upper() + ', %s code' % d_['code'])

x_labels = {'bsc': 'Crossover probability',
            'bec': 'Erasure probability',
            'biawgn': 'E_b/N in dB for E_b=1'}


class DataRoot:
    def __init__(self, file_name, label):
        self.label = label
        self.file_name = file_name
        self.data = utils.load_json(os.path.join(args.data_dir, file_name))
        if self.data is None: print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>', self.file_name)
    def get_label(self):
        if args.legend_format is None:
            return self.label
        else:
            return legend_reg.get(args.legend_format)(self.data)


def plot_(pairs, label, style=None):
    pairs_ = list(zip(map(float, pairs.keys()), pairs.values()))
    pairs_.sort(key=lambda x: x[0])
    argsl = list(zip(*pairs_))
    kwargs = {'linewidth':args.linewidth, 'label':label}
    if style is None: plt.plot(*argsl, **kwargs)
    else: plt.plot(*argsl, style, **kwargs)


plot_reg = ut.Registry()
def reg_plot(help_str):
    def inner(func):
        func.help_str = help_str
        plot_reg.put(func.__name__, func)
        return func
    return inner


@reg_plot('plots of all available data')
def plot_all(dl):
    for r in dl: plot_(r.data[args.error], r.get_label())
    fmt_err()
    plot_common()


def comp_average(dl):
    pot = {}
    for r in dl:
        for point,val in r.data[args.error].items():
            if point not in pot.keys(): pot[point] = []
            pot[point].append(val)
    for point in pot:
        vals = pot[point]
        pot[point] = sum(vals) / float(len(vals))
    return pot

@reg_plot('ensemble of codes and their average')
def ensemble(dl):
    for r in dl: plot_(r.data[args.error], None, 'r--')
    plot_(comp_average(dl), 'Average', 'b-')
    fmt_err()
    plot_common('Performance of code ensemble')


@reg_plot('compute average of regex matching files')
def regex_average(dl):
    used_l = []
    for rg in args.group_regex:
        new_dl = []
        for r in dl:
            if re.search(rg[0], r.file_name):
                new_dl.append(r)
                used_l.append(r)
        print('Regex group: %s'%rg, *[r.file_name for r in new_dl], sep='\n')
        plot_(comp_average(new_dl), rg[1])
    rest_dl = list(set(dl) - set(used_l))
    for r in rest_dl: plot_(r.data[args.error], r.get_label())
    fmt_err()
    plot_common()


@reg_plot('histogram of iteration count for e.g. ADMM decoder')
def hist_iter(dl):
    ax = plt.gca()
    if args.param is None: raise Exception('Parameter is None!')
    xmin, xmax = 1e10, 0
    for r in dl:
        series = np.array(r.data['dec'][str(args.param)]['iter'])
        xvals = range(len(series))
        avg = r.data['dec'][str(args.param)]['average']
        ax.bar(xvals, series, label='Average=%g'%avg)
        nzero = series.nonzero()[0]
        xmin = min(xmin, xvals[nzero[0]])
        xmax = max(xmax, xvals[nzero[-1]])
    ax.set_yticks([])
    diff = max(3, int((xmax-xmin)*0.01))
    ax.set_xlim(max(0, xmin-diff), xmax+diff)
    ut.mpl.fmt_ax(ax, 'Number of iterations', 'Frequency', leg=1, grid=1)
    plot_common('Iteration count histogram')


@reg_plot('average iteration count for e.g. ADMM decoder')
def avg_iter(dl):
    for r in dl:
        dec = r.data['dec']
        pot = {point:dec[point]['average'] for point in dec}
        plot_(pot, r.get_label())
    xlab, ylab = x_labels[args.channel], 'Average number of iterations'
    ut.mpl.fmt_ax(plt.gca(), xlab, ylab, leg=1, grid=1)
    plot_common('Average iteration count')


def plot_common(title=None):
    plt.legend(loc='best')
    if args.xlim is not None: plt.xlim(args.xlim)
    if args.ylim is not None: plt.ylim(args.ylim)
    if not args.title is None: title = args.title
    if title: plt.title(title)
    plt.margins(0)  # autoscale(tight=True)
    utils.make_dir_if_not_exists(args.plots_dir)
    img_path = os.path.join(args.plots_dir, args.file_name)
    ut.mpl.save_show_fig(args, plt, img_path)

def fmt_err():
    xlab, ylab = x_labels[args.channel], args.error.upper()
    ut.mpl.fmt_ax(plt.gca(), xlab, ylab, leg=1, grid=1, grid_kwargs={'which':'both'})
    plt.yscale('log')


def main(args):
    global matplotlib
    if args.agg: matplotlib.use('Agg')
    import matplotlib.pyplot

    global plt
    plt = matplotlib.pyplot

    file_names = ut.file.filter_strings(args, utils.get_data_file_list(args.data_dir))
    if not file_names: exit()
    labels = ut.file.gen_unique_labels(file_names, tokens=['_', '__', '-', '.json'])
    data_list = [DataRoot(fn, lb) for fn,lb in zip(file_names, labels)]
    data_list.sort(key=lambda it: ut.file.naturalkey(it.get_label()))
    args.channel = data_list[0].data['channel']
    plot_reg.get(args.type)(data_list)


def setup_parser():
    # https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', help='type of the graph', choices=plot_reg.keys(), default='plot_all')
    parser.add_argument('--param', help='parameter used in hist_iter graph', type=float)
    parser.add_argument('--error', help='error rate metric', default='ber', choices=['wer', 'ber'])
    parser.add_argument('--group_regex', nargs=2, action='append', help='if --type regex_average, matches a regex to file names and only plot their average. format: [<regex>, <legend name>]')

    parser.add_argument('--linewidth', type=float, default=2)
    parser.add_argument('--xlim', help='x-axis range', nargs=2, type=float)
    parser.add_argument('--ylim', help='y-axis range', nargs=2, type=float)

    parser.add_argument('--legend_format', help='format of legend entries', choices=legend_reg.keys())
    parser.add_argument('--title', help='graph title', type=str)
    parser.add_argument('--file_name', help='save name', type=str, default='graph')
    parser.add_argument('--agg', help='set matplotlib backend to Agg', action='store_true')

    ut.mpl.bind_fig_save_args(parser)
    ut.file.bind_filter_args(parser)
    return utils.bind_parser_common(parser)


if __name__ == '__main__':
    args = setup_parser().parse_args()
    print(vars(args))
    main(args)
