#!/usr/bin/python

"""Plot melting LFEs given order parameter."""

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

    out_filebase = '{}/{}_{}-lfes-melting'.format(args.filebase, args.tag)
    figsize = (plot.cm_to_inches(14), plot.cm_to_inches(10))
    plot.set_default_appearance()
    f = plt.figure(figsize=figsize, dpi=300)
    ax = f.add_subplot(1, 1, 1)
    ax.set_ylabel('$k_\mathrm{b}T$')
    ax.set_ylim([-0.5, 20])

    for system, vari in zip(args.systems, args.varis):
        inp_filebase = '{}/{}_{}-lfes-melting'.format(args.filebase, args.tag)
        lfes = pd.read_csv('{}.aves'.format(inp_filebase), sep=' ', index_col=0)
        lfe_stds = pd.read_csv('{}.stds'.format(inp_filebase), sep=' ', index_col=0)
        temp = lfes.columns[0]
        lfes = lfes[temp]
        lfe_stds = lfe_stds[temp]

        ax.errorbar(lfes.index, lfes, yerr=lfe_stds, marker='o',
                label='{}-{}, {} K'.format(system, vari, temp))

    plt.legend()
    plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
    f.savefig('{}.png'.format(out_filebase), transparent=True)
    f.savefig('{}.pdf'.format(out_filebase), transparent=True)


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
            help='Directory of inputs')
    parser.add_argument(
            'filebase',
            type=str,
            help='Filebase')
    parser.add_argument(
            'tag',
            type=str,
            help='OP tag')
    parser.add_argument(
            '--systems',
            nargs='+',
            type=str,
            help='Systems')
    parser.add_argument(
            '--varis',
            nargs='+',
            type=str,
            help='Simulation variants')

    return parser.parse_args()


if __name__ == '__main__':
    main()
