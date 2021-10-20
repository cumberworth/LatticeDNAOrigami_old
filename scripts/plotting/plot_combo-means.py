#!/usr/bin/python

"""Plot four order parameters from multiple simulations.

Plots number of bound staples, number of bound domains, number of misbound
domains, and number of stacked pairs.
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib import gridspec

from origamipy import plot
from matplotlibstyles import styles


def main():
    args = parse_args()
    f = setup_figure()
    gs_main = gridspec.GridSpec(2, 2, figure=f)
#    gs_main = gridspec.GridSpecFromSubplotSpec(
#        2, 2, subplot_spec=gs[:2, :], wspace=0.3, hspace=0.3)
    axes = [f.add_subplot(gs_main[i]) for i in range(4)]
    plot_figure(f, axes, vars(args))
    setup_axes(axes)
#    gs_lgd = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[2, :])
#    ax = f.add_subplot(gs_lgd[0])
#    set_labels(ax, axes)
    save_figure(f, args.plot_filebase)


def setup_figure():
    styles.set_thin_style()
    figsize = (styles.cm_to_inches(14), styles.cm_to_inches(11))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, axes, args):
    systems = args['systems']
    varis = args['varis']
    input_dir = args['input_dir']
    raw_assembled_values = args['all_assembled_values']
    xtag = args['xtag']
    contin = args['contin']

    tags = ['numstaples', 'numfulldomains', 'nummisdomains', 'numstackedpairs']

    # Parse assembled values
    all_assembled_values = []
    for assembled_values in raw_assembled_values:
        parsed_values = []
        for assembled_value in assembled_values.split(','):
            parsed_values.append(int(assembled_value))

        all_assembled_values.append(parsed_values)

    for assembled_values in all_assembled_values:
        for i, assembled_value in enumerate(assembled_values):
            ax = axes[i]
            ax.axhline(assembled_value, linestyle='--')

    for system, vari in zip(systems, varis):
        inp_filebase = f'{input_dir}/{system}-{vari}{post}'
        all_aves, all_stds = plot.read_expectations(inp_filebase)
        xvars = all_aves[xtag]
        for i, tag in enumerate(tags):
            means = all_aves[tag]
            stds = all_stds[tag]
            ax = axes[i]
            if contin:
                ax.fill_between(
                    xvars, means + stds, means - stds, color='0.8', label=vari)
                ax.plot(xvars, means, marker='None', label=vari)
            else:
                ax.errorbar(xvars, means, yerr=stds, marker='o', label=vari)


def setup_axes(axes):
    yaxis_labels = [
        '(Mis)bound staples',
        'Bound domain pairs',
        'Misbound domain pairs',
        'Stacked pairs']
    for ax, label in zip(axes, yaxis_labels):
        ax.set_xlabel(r'$T / K$')
        ax.set_ylabel(label)


def set_labels(ax, axes):
    ax.set_axis_off()
    handles, labels = axes[0].get_legend_handles_labels()
    ax.legend(handles, labels, loc='center', frameon=False, ncol=1)


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
        'plot_filebase',
        type=str,
        help='Plots directory')
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
        '--all_assembled_values',
        nargs='+',
        type=str,
        help='Bound staples,bound domains,misbound domains,'
        'fully stacked pairs')
    parser.add_argument(
        '--xtag',
        default='temp',
        type=str,
        help='Dependent variable tag')
    parser.add_argument(
        '--post',
        default='',
        type=str,
        help='Extra part of mean name (e.g. _temps for MWUS extrapolation')
    parser.add_argument(
        '--continuous',
        default=False,
        type=bool,
        help='Plot curves as continuous')

    return parser.parse_args()


if __name__ == '__main__':
    main()
