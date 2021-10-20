#!/usr/bin/python

"""Plot LFEs of given order parameter."""

import argparse
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from matplotlibstyles import styles


def main():
    args = parse_args()
    f = setup_figure()
    gs = gridspec.GridSpec(1, 1, f)
    ax = f.add_subplot(gs[0, 0])
    if args.post_lfes == None:
        args.post_lfes = ['' for i in range(len(args.systems))]

    plot_figure(f, ax, vars(args))
    setup_axis(ax, args.tag)
    #set_labels(ax)
    save_figure(f, args.plot_filebase)


def setup_figure():
    styles.set_default_style()
    figsize = (styles.cm_to_inches(10), styles.cm_to_inches(7))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, ax, args):
    systems = args['systems']
    varis = args['varis']
    input_dir = args['input_dir']
    tag = args['tag']
    post_lfes = args['post_lfes']

    for system, vari, post_lfe in zip(systems, varis, post_lfes):
        if post_lfe != '':
            post_lfe = '-' + post_lfe

        inp_filebase = f'{input_dir}/{system}-{vari}_lfes{post_lfe}-{tag}'
        lfes = pd.read_csv(f'{inp_filebase}.aves', sep=' ', index_col=0)
        lfe_stds = pd.read_csv(f'{inp_filebase}.stds', sep=' ', index_col=0)
        temp = lfes.columns[0]
        lfes = lfes[temp]
        lfe_stds = lfe_stds[temp]

        label = f'{system}-{vari}'
        ax.errorbar(lfes.index, lfes, yerr=lfe_stds, marker='o', label=label)


def setup_axis(ax, tag):
    ax.set_ylabel('$k_\mathrm{b}T$')
    ax.set_xlabel(tag)
#    ax.set_ylim([-0.5, 20])


def set_labels(ax):
    plt.legend()


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
        help='Input directory')
    parser.add_argument(
        'plot_filebase',
        type=str,
        help='Plots directory')
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
    parser.add_argument(
        '--post_lfes',
        nargs='+',
        type=str,
        help='Filename additions after lfes, if any')

    return parser.parse_args()


if __name__ == '__main__':
    main()
