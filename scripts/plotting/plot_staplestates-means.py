#!/usr/bin/python

"""Plot occupancy curves of each staple type."""

import argparse

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import numpy as np

from matplotlibstyles import styles
from origamipy import plot
from origamipy import utility


def main():
    args = parse_args()
    f = setup_figure()
    gs = gridspec.GridSpec(
        1, 2, f, width_ratios=[10, 1], height_ratios=[1])
    ax = f.add_subplot(gs[0])
    mappable = plot_figure(f, ax, vars(args))
    setup_axis(ax)
    set_labels(f, ax, mappable)
    plot_filebase = f'{args.plot_dir}/{args.filebase}_staplestates-means'
    save_figure(f, plot_filebase)


def setup_figure():
    styles.set_thin_style()
    figsize = (styles.cm_to_inches(14), styles.cm_to_inches(10))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, ax, args):
    input_dir = args['input_dir']
    filebase = args['filebase']
    stapletypes = args['stapletypes']
    rtag = args['rtag']
    rvalue = args['rvalue']
    xtag = args['xtag']
    contin = args['continuous']

    inp_filebase = f'{input_dir}/{filebase}'
    tagbase = 'staplestates'
    tags = [f'{tagbase}{i}' for i in range(1, stapletypes + 1)]
    aves, stds = plot.read_expectations(inp_filebase)
    temps = aves['temp']
    melting_points = utility.estimate_staple_melting_points(stapletypes, aves, temps)
    min_t = np.min(melting_points)
    max_t = np.max(melting_points)
    cmap = cm.get_cmap('viridis')
    mappable = styles.create_linear_mappable(cmap, min_t, max_t)
    if rtag:
        aves = aves[aves[args.rtag] == rvalue]
        stds = stds[stds[args.rtag] == rvalue]

    for i, tag in enumerate(tags):
        xvars = aves[xtag]
        color = mappable.to_rgba(melting_points[i])
        darkcolor = styles.darken_color(color[:3], 0.4)
        if contin:
            ax.fill_between(
                xvars, aves[tag] + stds[tag], aves[tag] - stds[tag],
                color='0.8')
        else:
            ax.errorbar(xvars, aves[tag], yerr=stds[tag], color=darkcolor,
                        linestyle='None', marker='o')

        ax.plot(xvars, aves[tag], color=color, marker='None')

    return mappable


def setup_axis(ax):
    ax.set_xlabel('$T$ / K')
    ax.set_ylabel('Staple state')


def set_labels(f, ax, mappable):
    f.colorbar(mappable, orientation='vertical')


def save_figure(f, plot_filebase):
    # f.savefig(plot_filebase + '.pgf', transparent=True)
    f.savefig(plot_filebase + '.pdf', transparent=True)
    f.savefig(plot_filebase + '.png', transparent=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory of inputs')
    parser.add_argument(
        'plot_dir',
        type=str,
        help='Plot directory')
    parser.add_argument(
        'filebase',
        type=str,
        help='Filebase')
    parser.add_argument(
        'stapletypes',
        type=int,
        help='Number of staple types')
    parser.add_argument(
        '--xtag',
        default='temp',
        type=str,
        help='Dependent variable tag')
    parser.add_argument(
        '--rtag',
        type=str,
        help='Tag to slice on')
    parser.add_argument(
        '--rvalue',
        type=float,
        help='Slice value')
    parser.add_argument(
        '--continuous',
        default=False,
        type=bool,
        help='Plot curves as continuous')

    return parser.parse_args()


if __name__ == '__main__':
    main()
