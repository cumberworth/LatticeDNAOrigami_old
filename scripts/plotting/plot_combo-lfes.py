#!/usr/bin/python

"""Plot LFE difference of a binary OP against another OP."""

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
    tag2 = args.tag2

    figsize = (plot.cm_to_inches(8.5), plot.cm_to_inches(7))

    tag1s = ['staplestates{}'.format(i) for i in range(1, args.stapletypes + 1)]
    for tag1 in tag1s:
        inp_filebase = '{}/{}-{}_{}-{}-lfes'.format(args.input_dir, args.system,
                args.vari, tag1, tag2)
        out_filebase = '{}/{}-{}_{}-{}-lfes-all'.format(args.output_dir,
                args.system, args.vari, tag1, tag2)

        plot.set_default_appearance()
        f = plt.figure(figsize=figsize, dpi=300)
        gs = gridspec.GridSpec(2, 1, width_ratios=[1], height_ratios=[20, 1])
        gs_main = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0, :],
                                                   wspace=0, hspace=0)
        gs_cb = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[1, :])
        ax = f.add_subplot(gs[0])
        ax.set_ylabel('$k_\mathrm{b}T$')

        norm = mpl.colors.Normalize(vmin=args.temps[0], vmax=args.temps[-1])
        cmap = mpl.cm.viridis
        scalarmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        aves_filename = '{}.aves'.format(inp_filebase, tag1, tag2)
        aves = pd.read_csv(aves_filename, sep=' ')
        aves_one = aves[aves[tag1] == 1].sort_values(tag2).set_index(tag2)
        aves_zero = aves[aves[tag1] == 0].sort_values(tag2).set_index(tag2)
        aves_diff = aves_one - aves_zero

        stds_filename = '{}.stds'.format(inp_filebase, tag1, tag2)
        stds = pd.read_csv(stds_filename, sep=' ')
        stds_one = stds[stds[tag1] == 1].sort_values(tag2).set_index(tag2)
        stds_zero = stds[stds[tag1] == 0].sort_values(tag2).set_index(tag2)
        stds_prop = np.sqrt(stds_one**2 + stds_zero**2)

        for i, temp in enumerate(args.temps):
            temp_key = '{:.3f}'.format(temp)
            ax.errorbar(aves_diff.index, aves_diff[temp_key],
                    yerr=stds_prop[temp_key], marker='o',
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
            'stapletypes',
            type=int,
            help='Number of staple types')
    parser.add_argument(
            'tag2',
            type=str,
            help='Cumaltive OP tag to plot against')
    parser.add_argument(
            '--temps',
            nargs='+',
            type=float,
            help='Temperatures')

    return parser.parse_args()


if __name__ == '__main__':
    main()
