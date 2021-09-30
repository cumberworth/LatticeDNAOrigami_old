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
    gs = gridspec.GridSpec(
        2, 1, f, width_ratios=[1], height_ratios=[20, 1])
    ax = f.add_subplot(gs[0, 0])
    ax_c = f.add_subplot(gs[1, 0])
    plot_figure(f, ax, ax_c, args.filebase, args.input_dir)
    setup_axis(ax, ax_c)
    plot_filebase = f'{args.output_dir}/{args.filebase}'
    save_figure(f, plot_filebase)


def setup_figure():
    styles.set_default_style()
    figsize = (styles.cm_to_inches(15), styles.cm_to_inches(10))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, ax, ax_c, filebase, input_dir):
    inp_filebase = f'{input_dir}/{filebase}'
    aves = pd.read_csv(f'{inp_filebase}.aves', sep=' ')
    temps = np.array(aves.columns[1:], dtype=float)
    norm = mpl.colors.Normalize(vmin=temps[0], vmax=temps[-1])
    cmap = mpl.cm.viridis
    scalarmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    stds = pd.read_csv(f'{inp_filebase}.stds', sep=' ')
    for i, temp in enumerate(temps):
        temp_key = f'{temp:.3f}'
        ax.errorbar(
            aves.index, aves[temp_key], yerr=stds[temp_key], marker='o',
            color=scalarmap.to_rgba(temp))

    # Colorbar
    cbar = mpl.colorbar.ColorbarBase(
        ax_c, orientation='horizontal', cmap=cmap, norm=norm)


def setup_axis(ax, ax_c):
    ax.set_ylabel('$k_\mathrm{b}T$')
    ax.set_ylim([-0.5, 20])

    ax_c.set_xlabel('Temperature / K')


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
