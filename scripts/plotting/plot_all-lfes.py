#!/usr/bin/python

"""Plot LFEs for give order parameter across temperature range."""

import argparse
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from origamipy import plot


def main():
    args = parse_args()

    filebase = '{}/{}-{}'.format(args.input_dir, args.system, args.vari)
    out_filebase = '{}/{}-{}_{}-{}'.format(args.output_dir, args.system,
            args.vari, args.tag, args.lfebase)
    figsize = (plot.cm_to_inches(14), plot.cm_to_inches(10))

    plot.set_default_appearance()
    f = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[20, 1])
    gs_main = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, :],
                                               wspace=0, hspace=0)
    gs_cb = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, :])
    ax = f.add_subplot(gs[0])
    ax.set_ylabel('$k_\mathrm{b}T$')
    ax.set_ylim([-0.5, 20])

    aves_filename = '{}_{}-{}.aves'.format(filebase, args.tag, args.lfebase)
    aves = pd.read_csv(aves_filename, sep=' ')
    temps = np.array(aves.columns[1:], dtype=float)
    norm = mpl.colors.Normalize(vmin=temps[0],
            vmax=temps[-1])
    cmap = mpl.cm.viridis
    scalarmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    stds_filename = '{}_{}-{}.stds'.format(filebase, args.tag, args.lfebase)
    stds = pd.read_csv(stds_filename, sep=' ')
    for i, temp in enumerate(temps):
        temp_key = '{:.3f}'.format(temp)
        ax.errorbar(aves.ops, aves[temp_key], yerr=stds[temp_key], marker='o',
                color=scalarmap.to_rgba(temp))

    # Colorbar
    ax = f.add_subplot(gs_cb[0])
    cbar = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap,
            norm=norm)
    ax.set_xlabel('Temperature / K')

    plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
    f.savefig('{}.png'.format(out_filebase), transparent=True)
    f.savefig('{}.pdf'.format(out_filebase), transparent=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'input_dir',
            type=str,
            help='Directory of inputs')
    parser.add_argument(
            'output_dir',
            type=str,
            help='Output directory')
    parser.add_argument(
            'system',
            type=str,
            help='System')
    parser.add_argument(
            'vari',
            type=str,
            help='Simulation variant')
    parser.add_argument(
            'tag',
            type=str,
            help='OP tag')
    parser.add_argument(
            'lfebase',
            type=str,
            help='LFE filename base')

    return parser.parse_args()


if __name__ == '__main__':
    main()
