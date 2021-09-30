#!/usr/bin/python

"""Plot melting temperartures as heatmap."""

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
    gs = gridspec.GridSpec(1, 1, f)
    ax = f.add_subplot(gs[0])
    plot_figure(
        f, ax, args.input_dir, args.filebase, args.stapletypes,
        args.mapfile, args.rtag, args.rvalue)
    setup_axis(ax)
    plot_filebase = f'{args.plot_dir}/{args.filebase}_staplestates-melting'
    save_figure(f, plot_filebase)


def setup_figure():
    styles.set_thin_style()
    figsize = (styles.cm_to_inches(14), styles.cm_to_inches(10))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(
        f, ax, input_dir, filebase, stapletypes, mapfile, rtag, rvalue):

    inp_filebase = f'{input_dir}/{filebase}'
    index_to_stapletype = np.loadtxt(mapfile, dtype=int)
    aves, stds = plot.read_expectations(inp_filebase)
    temps = aves['temp']
    melting_points = utility.estimate_staple_melting_points(
        stapletypes, aves, temps)
    min_t = np.min(melting_points)
    max_t = np.max(melting_points)
    cmap = cm.get_cmap('viridis')
    mappable = styles.create_linear_mappable(cmap, min_t, max_t)
    if rtag:
        aves = aves[aves[rtag] == rvalue]
        stds = stds[stds[rtag] == rvalue]

    rows = index_to_stapletype.shape[0]
    cols = index_to_stapletype.shape[1]
    assembled_array = np.zeros([rows, cols])
    for row, staple_types in enumerate(index_to_stapletype):
        for col, staple_type in enumerate(staple_types):
            assembled_array[row, col] = melting_points[staple_type - 1]

    # Plot simulation melting points
    im = ax.imshow(assembled_array, vmin=min_t, vmax=max_t, cmap=cmap)
    plt.colorbar(im, orientation='horizontal')


def setup_axis(ax):
    ax.axis('off')


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
        'mapfile',
        type=str,
        help='Index-to-staple type map filename')
    parser.add_argument(
        '--rtag',
        type=str,
        help='Tag to slice on')
    parser.add_argument(
        '--rvalue',
        type=float,
        help='Slice value')

    return parser.parse_args()


if __name__ == '__main__':
    main()
