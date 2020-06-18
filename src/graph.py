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
r_('chl_dec', lambda d_: d_['channel'].upper() + ', %s decoder' % d_['decoder'])
r_('chl_code', lambda d_: d_['channel'].upper() + ', %s code' % d_['code'])

x_labels = {'bsc': 'Crossover probability',
            'bec': 'Erasure probability',
            'biawgn': 'E_b/N in dB for E_b=1'}


class DataRoot:
    def __init__(self, file_name, label):
        self.label = label
        self.data = utils.load_json(os.path.join(args.data_dir, file_name))
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
reg_plot = plot_reg.reg

@reg_plot
def plot_all(dl): # plot all files
    for r in dl: plot_(r.data[args.error], r.get_label())
    fmt_err()
    plot_common(args.title)

@reg_plot
def ensemble(dl): # average and ensemble
    pot = {}
    for r in dl:
        for point,val in r.data[args.error].items():
            if point not in pot.keys(): pot[point] = []
            pot[point].append(val)
    for point in pot:
        vals = pot[point]
        pot[point] = sum(vals) / float(len(vals))

    for r in dl: plot_(r.data[args.error], None, 'r--')
    plot_(pot, 'Average', 'b-')
    fmt_err()
    title = 'Code ensemble' # + ', %s decoder' % args.decoder[0]
    plot_common(title)


@reg_plot # depricated, discontinued after commit e9f545908d27fee157f8896fcdf40939022708d5
def hist_iter(dl): # histogram and stats of iteration count
    chk = lambda it: it.get('code', '') == args.code and \
                     it.get('decoder', '') == args.decoder[0]
    data = get_first(filter_data(dl, chk), 'single')
    # plot_(data[args.error], 'k-', data['decoder'])
    series = data['dec'][str(args.param)]['iter']
    xvals = range(len(series))
    avg = sum([a1_ * a2_ for a1_, a2_ in zip(xvals, series)]) / sum(series)
    plt.bar(xvals, series, label='Average=%g' % avg)
    plt.xlabel('Number of iterations')
    plt.gca().set_yticks([])
    plot_common()

@reg_plot # depricated, discontinued after commit e9f545908d27fee157f8896fcdf40939022708d5
def avg_iter(dl): # plot on average number of iterations
    chk = lambda it: it.get('code', '') == args.code and \
                     it.get('decoder', '') in args.decoder
    for data in filter_data(dl, chk):
        # plot_(data[args.error], 'k-', data['decoder'])
        params = sorted([param for param in data['dec'].keys()])
        avgs = [data['dec'][param]['average'] for param in params]
        plt.plot(params, avgs, label=data['decoder'])
        plt.xlabel(x_labels[args.channel])
        plt.ylabel('Average number of iterations')
        plt.grid(True, which='both')
    plot_common()


def plot_common(title=None):
    plt.legend(loc='best')
    if args.xlim is not None: plt.xlim(args.xlim)
    if args.ylim is not None: plt.ylim(args.ylim)
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

    args.and_kw.append(args.channel)
    file_names = ut.file.filter_strings(args, utils.get_data_file_list(args.data_dir))
    if not file_names: exit()
    labels = ut.file.gen_unique_labels(file_names, tokens=['_', '__', '-', '.json'])
    data_list = [DataRoot(fn, lb) for fn,lb in zip(file_names, labels)]
    plot_reg.get(args.type)(data_list)


def setup_parser():
    # https://stackoverflow.com/questions/17073688/how-to-use-argparse-subparsers-correctly
    parser = argparse.ArgumentParser()
    parser.add_argument('channel', help='channel', choices=['bec', 'bsc', 'biawgn'])
    parser.add_argument('--type', help='plot type', choices=plot_reg.keys(), default='plot_all')
    parser.add_argument('--param', help='param', type=float)
    parser.add_argument('--error', help='which error rate', default='ber', choices=['wer', 'ber'])

    parser.add_argument('--linewidth', type=float, default=2)
    parser.add_argument('--xlim', help='x-axis range', nargs=2, type=float)
    parser.add_argument('--ylim', help='y-axis range', nargs=2, type=float)

    parser.add_argument('--legend_format', choices=legend_reg.keys())
    parser.add_argument('--title', type=str)
    parser.add_argument('--file_name', type=str)
    parser.add_argument('--agg', help='set matplotlib backend to Agg', action='store_true')

    ut.mpl.bind_fig_save_args(parser)
    ut.file.bind_filter_args(parser)
    return utils.bind_parser_common(parser)


if __name__ == "__main__":
    args = setup_parser().parse_args()
    print(vars(args))
    main(args)
