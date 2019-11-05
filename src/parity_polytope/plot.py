from mpl_toolkits.mplot3d import axes3d, Axes3D
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import argparse

import exact
import apprx


def plot_pp(dim):
    fig = plt.figure()
    if dim == 2:
        pts = np.array([[0, 0], [1, 1], [1, 1.000001]])
        ax = fig.add_subplot(111)
        for sx in ConvexHull(pts).simplices:
            plt.plot(pts[sx, 0], pts[sx, 1], 'k-', zorder=1)
    else:
        # https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud
        pts = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
        hull = ConvexHull(pts)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(pts.T[0], pts.T[1], pts.T[2], 'ko')
        for s in hull.simplices:
            s = np.append(s, s[0])  # Here we cycle back to the first coordinate
            ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")
            # ax.set_zticklabels([])

    return fig, ax


def main(args):
    demo, dim = args.type, args.dim
    fig, ax = plot_pp(dim)
    margin = .1

    config = {'markersize': 15, 'fillstyle': 'full', 'markeredgewidth': 0.0}
    ax_ = lambda d_, m_, l_: ax.plot(*d_, m_, label=l_, **config)[0]
    lines_ = lambda d_, e_, a_: (ax_(d_, 'g^', 'Data'), ax_(e_, 'bs', 'Exact'), ax_(a_, 'ro', 'Apprx'))

    model = apprx.load_model(dim, [20, 20])

    if demo == 'static':
        config['markersize'] = 10
        dat = np.random.rand(5, dim)
        prj = exact.proj_rows(dat)
        apx = model.eval_rows(dat)
        lines_(dat.T, prj.T, apx.T)
    else:
        current = np.zeros(dim, dtype=float)
        conf = current[:, np.newaxis]
        # text = ax.text(0, 1, '', fontsize=15)
        l_dat, l_prj, l_apx = lines_(conf, conf, conf)

        # https://matplotlib.org/examples/animation/simple_3danim.html
        def set_data(line, dat):
            line.set_data(dat[0:2])
            if dim == 3: line.set_3d_properties(dat[2])

        def update(dat):
            prj, apx = exact.proj_vec(dat), model.eval_vec(dat)
            set_data(l_dat, dat)
            set_data(l_prj, prj)
            set_data(l_apx, apx)
            err = 'Error: %.5f' % np.linalg.norm(apx - prj)
            # text.set_text(err)
            fig.canvas.draw_idle()
            # y=1 to avid title from drifting in 3d plot
            plt.title(err, y=1)

        if demo == 'hover':
            # https://stackoverflow.com/questions/16527930/matplotlib-update-position-of-patches-or-set-xy-for-circles
            def hover(event):
                if event.inaxes is not None:
                    update(np.array((event.xdata, event.ydata)))

            if dim == 2: fig.canvas.mpl_connect("motion_notify_event", hover)

        else:
            speed = .001
            interval = 10
            bbox = [-margin, 1 + margin]
            bbox = [bbox, bbox] + ([bbox] if dim == 3 else [])

            def rand_unit_vec():
                theta = np.random.rand() * 2 * np.pi
                zed = np.random.rand() * 2 - 1
                zed1 = np.sqrt(1 - zed ** 2)
                ct, st = np.cos(theta), np.sin(theta)
                # print(np.array([ct, st]))
                return np.array([ct, st]) if dim == 2 else \
                    np.array([zed1 * ct, zed1 * st, zed])

            vec = rand_unit_vec()

            def edge(dat, ind):
                if dat[ind] < bbox[ind][0] or dat[ind] > bbox[ind][1]:
                    dat[ind] = np.clip(dat[ind], *bbox[ind])
                    vec[ind] *= -1  # flip direction at edge
                    # also sample new direction, to randomize movement
                    vec[:] = np.sign(vec) * np.abs(rand_unit_vec())

            def on_timer():
                current[:] = current + speed * interval * vec
                for i in range(len(current)): edge(current, i)
                update(current)

            def on_click(event):
                if event.inaxes is not None:
                    current[:] = np.array((event.xdata, event.ydata))
                    update(current)

            if dim == 2: fig.canvas.mpl_connect('button_press_event', on_click)
            timer = fig.canvas.new_timer(interval=interval, callbacks=[(on_timer, [], {})])
            timer.start()

    ax.set_aspect('equal')
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    plt.margins(margin)
    plt.tight_layout()
    plt.legend(numpoints=1, loc='best')
    plt.show()


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='plot-type', choices=['static', 'hover', 'walk'])
    parser.add_argument('dim', help='dimension', choices=[2, 3], type=int)
    return parser


if __name__ == "__main__":
    main(setup_parser().parse_args())
