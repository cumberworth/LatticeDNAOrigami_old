#!/usr/bin/python

"""Plot LFEs for give order parameter across temperature range."""

import argparse
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from matplotlibstyles import styles


def main():
    args = parse_args()
    f = setup_figure()
    gs = gridspec.GridSpec(2, 1, f)
    ax = f.add_subplot(gs[0, 0])
    mappable = plot_figure(f, ax, vars(args))
    setup_axis(ax)
    set_labels(f, ax, mappable)
    plot_filebase = f'{args.output_dir}/{args.filebase}'
    save_figure(f, plot_filebase)


def setup_figure():
    styles.set_default_style()
    figsize = (styles.cm_to_inches(15), styles.cm_to_inches(10))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, ax, args):
    filebase = args['filebase']
    input_dir = args['input_dir']

    inp_filebase = f'{input_dir}/{filebase}'
    aves = pd.read_csv(f'{inp_filebase}.aves', sep=' ')
    temps = np.array(aves.columns[1:], dtype=float)
    norm = mpl.colors.Normalize(vmin=temps[0], vmax=temps[-1])
    cmap = mpl.cm.viridis
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    stds = pd.read_csv(f'{inp_filebase}.stds', sep=' ')
    for i, temp in enumerate(temps):
        temp_key = f'{temp:.3f}'
        ax.errorbar(
            aves.index, aves[temp_key], yerr=stds[temp_key], marker='o',
            color=mappable.to_rgba(temp))

    return mappable


def setup_axis(ax):
    ax.set_ylabel('$k_\mathrm{b}T$')
    ax.set_ylim([-0.5, 20])


def set_labels(f, ax, mappable):
    cbar = f.colorbar(mappable, orientation='horizontal')
    cbar.ax.set_xlabel('Temperature / K')


def save_figure(f, plot_filebase):
    #f.savefig(plot_filebase + '.pgf', transparent=True)
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
        'output_dir',
        type=str,
        help='Output directory')
    parser.add_argument(
        'filebase',
        type=str,
        help='Filebase')

    return parser.parse_args()


if __name__ == '__main__':
    main()
