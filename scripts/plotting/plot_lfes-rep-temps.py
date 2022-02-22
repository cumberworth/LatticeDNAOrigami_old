#!/usr/bin/python

"""Plot LFEs for give order parameter across temperature range."""

import argparse

import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
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
    plot_figure(f, axes, vars(args))
    setup_axis(axes)
#    set_labels(f, ax, mappable)
    plot_filebase = f'{args.plots_dir}/{args.filebase}'
    save_figure(f, plot_filebase)


def setup_figure():
    styles.set_default_style()
    figsize = (plotutils.cm_to_inches(10), plotutils.cm_to_inches(7))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, axes, args):
    filebase = args['filebase']
    input_dir = args['input_dir']
    cmap = cm.get_cmap('tab10')
    markers = ['^', 's', 'o']
    tags = ['numfullyboundstaples', 'numfulldomains']
    labels = ['Fully bound staples', 'Bound domains']

    for j, tag in enumerate(tags):
        if tag == 'numfulldomains':
            ax = axes[1]
        else:
            ax = axes[0]

        replica_filebase = f'{input_dir}/{filebase}_lfes-{tag}'
        replica_aves = pd.read_csv(f'{replica_filebase}.aves', sep=' ',
                                   index_col=0)
        replica_stds = pd.read_csv(f'{replica_filebase}.stds', sep=' ',
                                  index_col=0)
        melting_filebase = f'{input_dir}/{filebase}_lfes-melting-{tag}'
        melting_aves = pd.read_csv(f'{melting_filebase}.aves', sep=' ',
                                  index_col=0)
        melting_stds = pd.read_csv(f'{melting_filebase}.stds', sep=' ',
                                  index_col=0)

        temps = np.array(replica_aves.columns, dtype=float)
        #norm = mpl.colors.Normalize(vmin=temps[0], vmax=temps[-1])
        #cmap = mpl.cm.viridis
        #mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

        for i, k in enumerate([0, -1]):
            ax.errorbar(
                    replica_aves.index,
                    replica_aves.iloc[:, k],
                    yerr=replica_stds.iloc[:, k],
                    marker=markers[i],
#                    alpha=0.5,
                    color=plotutils.darken_color(cmap(j)[:-1], 1.3))
                    #color=mappable.to_rgba(temps[i]))

        ax.errorbar(
                melting_aves.index,
                melting_aves.iloc[:, 0],
                yerr=melting_stds.iloc[:, 0],
                marker=markers[2],
                color=cmap(j),
                label=labels[j])
                #color=mappable.to_rgba(temps[i]))

    #return mappable


def setup_axis(axes, ylabel=None, xlabel_bottom=None, xlabel_top=None, ylim_top=None):
    ax = axes[0]
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel_bottom)

    ax = axes[1]
    ax.spines.top.set_visible(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.set_xlabel(xlabel_top)

    ax.set_ylim()
    ax.set_ylim([-1.0, ylim_top])


def set_labels(f, ax, mappable):
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
        help='Directory of inputs')
    parser.add_argument(
        'plots_dir',
        type=str,
        help='Plots directory')
    parser.add_argument(
        'filebase',
        type=str,
        help='Filebase')

    return parser.parse_args()


if __name__ == '__main__':
    main()
