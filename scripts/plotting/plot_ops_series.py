#!/usr/bin/python

"""Plot order parameter series for all runs and reps.

Plots number of bound staples, number of bound domains, number of misbound
domains, and number of stacked pairs.
"""

import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import origamipy.plot as plot


def main():
    args = parse_args()
    skip = 1
    out_filebase = '{}/{}-{}_timeseries-{}'.format(args.output_dir,
            args.system, args.vari, args.temp_i)
    tags = ['numstaples', 'numfulldomains', 'nummisdomains', 'numstackedpairs']
    labels = [
        'Bound staples',
        'Bound domains',
        'Misbound domains',
        'Stacked pairs']
    figsize = (plot.cm_to_inches(18), plot.cm_to_inches(36))
    plot.set_default_appearance()
    f, axes = plt.subplots(3, 1, figsize=figsize, dpi=300)

    for rep in range(args.reps):
        ax = axes[rep]
        ax.set_xlabel('Walltime / s')
        timeseries = {}
        times = []
        for tag in tags:
            timeseries[tag] = []

        for run in range(args.runs):
            filebase = '{}/{}-{}_run-{}_rep-{}-{}'.format(args.input_dir,
                    args.system, args.vari, run, rep, args.temp)
            ops_filename = '{}.ops'.format(filebase)
            ops = read_ops_from_file(ops_filename, tags, skip)
            times_filename = '{}.times'.format(filebase)
            new_times = np.loadtxt(times_filename, skiprows=1)[::skip, 1]
            if run != 0:
                new_times += times[-1]

            times.extend(new_times.tolist())

            for tag in tags:
                timeseries[tag].extend(ops[tag])

        # Plot timeseries
        for i, tag in enumerate(tags):
            ax.plot(times, timeseries[tag], marker=None, label=labels[i],
                    color='C{}'.format(i), zorder=4-i)

        # Plot expected value
        for i, tag in enumerate(tags):
            if args.assembled_values[i] != 0:
                ax.axhline(args.assembled_values[i], linestyle='--', color='C{}'.format(i))

    # Plot legend
    ax = axes[0]
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, frameon=False, loc='center',
            bbox_to_anchor=(0.7, 0.25))

    plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
    f.savefig('{}.pdf'.format(out_filebase), transparent=True)
    f.savefig('{}.png'.format(out_filebase), transparent=True)


def read_ops_from_file(filename, tags, skip):
    """Read specified order parameters from file

    Returns a dictionary of tags to values.
    """
    with open(filename) as inp:
        header = inp.readline().split(', ')

    all_ops = np.loadtxt(filename, skiprows=1, dtype=int)[::skip]
    ops = {}
    for i, tag in enumerate(header):
        if tag in tags:
            ops[tag] = all_ops[:, i]

    return ops


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
            'temp',
            type=str,
            help='Temperature to plot')
    parser.add_argument(
            'temp_i',
            type=str,
            help='Index of temperature')
    parser.add_argument(
            'runs',
            type=int,
            help='Number of runs')
    parser.add_argument(
            'reps',
            type=int,
            help='Number of reps')
    parser.add_argument(
            '--assembled_values',
            nargs='+',
            type=int,
            help='Bound staples bound domains misbound domains '
                    'fully stacked pairs')

    return parser.parse_args()


if __name__ == '__main__':
    main()
