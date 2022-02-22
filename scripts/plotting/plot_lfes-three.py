#!/usr/bin/python

"""Plot numstaples, numfulldomains, and numfullybound staples LFEs."""

import argparse
import sys

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from matplotlibstyles import styles
from matplotlibstyles import plotutils


def main():
    args = parse_args()
    f = setup_figure()
    gs = gridspec.GridSpec(1, 1, f)
    ax = f.add_subplot(gs[0, 0])
    axes = [ax, ax.twiny()]
    if args.post_lfes == None:
        args.post_lfes = ['']*3

    plot_figure(f, axes, vars(args))
    setup_axis(axes)
    #set_labels(axes)
    save_figure(f, args.plot_filebase)


def setup_figure():
    styles.set_default_style()
    figsize = (plotutils.cm_to_inches(10), plotutils.cm_to_inches(7))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, axes, args):
    system = args['system']
    varis = args['varis']
    input_dir = args['input_dir']
    post_lfes = args['post_lfes']
    tags = ['numstaples', 'numfullyboundstaples', 'numfulldomains']
    labels = ['Staples', 'Fully bound staples', 'Bound domains']
    cmap = cm.get_cmap('tab10')
    for i, tag in enumerate(tags):
        if tag == 'numfulldomains':
            ax = axes[1]
        else:
            ax = axes[0]

        vari = varis[i]
        post_lfe = post_lfes[i]
        if post_lfe != '':
            post_lfe = '-' + post_lfe

        inp_filebase = f'{input_dir}/{system}-{vari}_lfes{post_lfe}-{tag}'
        lfes = pd.read_csv(f'{inp_filebase}.aves', sep=' ', index_col=0)
        lfe_stds = pd.read_csv(f'{inp_filebase}.stds', sep=' ', index_col=0)
        temp = lfes.columns[0]
        lfes = lfes[temp]
        lfes = lfes - lfes[0]
        lfe_stds = lfe_stds[temp]

        ax.errorbar(
            lfes.index,
            lfes,
            yerr=lfe_stds,
            marker='o',
            label=labels[i],
            color=cmap(i))


def setup_axis(axes, ylabel=None, xlabel_bottom=None, xlabel_top=None):
    ax = axes[0]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel_bottom)

    ax = axes[1]
    ax.spines.top.set_visible(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.set_xlabel(xlabel_top)


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
        'system',
        type=str,
        help='System')
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
