#!/usr/bin/python

"""Plot four order parameters from multiple simulations.

Plots number of bound staples, number of bound domains, number of misbound
domains, and number of stacked pairs.
"""

import argparse

import matplotlib.pyplot as plt
from matplotlib import gridspec

from origamipy import plot


def main():
    args = parse_args()

    tags = ['numstaples', 'numfulldomains', 'nummisdomains', 'numstackedpairs']
    yaxis_labels = [
        '(Mis)bound staples',
        'Bound domain pairs',
        'Misbound domain pairs',
        'Stacked pairs']
    figsize = (plot.cm_to_inches(14), plot.cm_to_inches(10))

    plot.set_default_appearance()
    f = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    gs_main = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[:, 0:2],
                                               wspace=0.3, hspace=0.3)
    axes = create_axes(f, gs_main, yaxis_labels)

    all_assembled_values = []
    for assembled_values in args.all_assembled_values:
        parsed_values = []
        for assembled_value in assembled_values:
            parsed_values.append(int(assembled_value))

        all_assembled_values.append(parsed_values)
            
    for assembled_values in args.all_assembled_values:
        ax = axes[i]
        ax.axhline(values, linestyle='--', color='0.8')

    for system, vari in zip(args.systems, args.varis):
        sim_filebases = '{}/{}-{}'.format(args.input_dir, system, vari)
        all_aves, all_stds = plot.read_expectations(sim_filebases)
        reduced_aves = all_aves[all_aves.bias == 0]
        reduced_stds = all_stds[all_stds.bias == 0]
        temps = reduced_aves.temp
        for i, tag in enumerate(tags):
            means = reduced_aves[tag]
            stds = reduced_stds[tag]
            ax = axes[i]
            ax.errorbar(temps, means, yerr=stds, marker='o', label=vari)

    # Plot legend
#    gs_lgd = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, 2])
    #ax = f.add_subplot(gs_lgd[0])
    #ax.set_axis_off()
    #handles, labels = axes[0].get_legend_handles_labels()
    #ax.legend(handles, labels, loc='center', frameon=False, ncol=1)

    gs.tight_layout(f, pad=1.0, h_pad=0, w_pad=0)
    f.savefig(args.output_filebase + '.png', transparent=True)
    f.savefig(args.output_filebase + '.pdf', transparent=True)


def create_axes(f, gs, yaxis_labels):
    axes = []
    for i, label in enumerate(yaxis_labels):
        ax = f.add_subplot(gs[i])
        ax.set_xlabel('$T$ / K')
        ax.set_ylabel(yaxis_labels[i])
        axes.append(ax)

    return axes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'input_dir',
            type=str,
            help='Directory of inputs')
    parser.add_argument(
            'output_filebase',
            type=str,
            help='Output filebase')
    parser.add_argument(
            '--systems',
            nargs='+',
            type=str,
            help='System file')
    parser.add_argument(
            '--varis',
            nargs='+',
            type=str,
            help='Simulation variant names')
    parser.add_argument(
            '--all_assembled_values',
            nargs='+',
            type=str,
            help='Bound staples,bound domains,misbound domains,'
                    'fully stacked pairs')

    return parser.parse_args()


if __name__ == '__main__':
    main()
