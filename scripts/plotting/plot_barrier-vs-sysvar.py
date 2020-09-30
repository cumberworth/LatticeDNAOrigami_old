#!/usr/bin/python

"""Plot barrier height vs temperature."""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from origamipy import plot


def main():
    args = parse_args()

    out_filebase = '{}_{}-barrier-vs-temp'.format(args.output_filebase,
            args.tag)
    figsize = (plot.cm_to_inches(18), plot.cm_to_inches(12))
    plot.set_default_appearance()
    f = plt.figure(figsize=figsize, dpi=300)
    ax = f.add_subplot()

    # Calculate barriers
    barriers = []
    barrier_stds = []
    for system, vari in zip(args.systems, args.varis):
        filebase = '{}/{}-{}_{}-lfes-melting'.format(args.input_dir, system, vari,
                args.tag)
        lfes = pd.read_csv('{}.aves'.format(filebase), sep=' ', index_col=0)
        lfes = lfes[lfes.index <= args.assembled_op]
        lfe_stds = pd.read_csv('{}.stds'.format(filebase), sep=' ', index_col=0)
        temp = lfes.columns[0]
        lfes = lfes[temp]
        lfe_stds = lfe_stds[temp]

        peaks, stds = calc_barrier_heights_and_errors(lfes,
                lfe_stds, temp)
        barriers.append(peaks[0])
        barrier_stds.append(stds[0])

    ax.errorbar(args.x_range, barriers, yerr=barrier_stds, marker='o')

#    plt.legend()
    plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
    f.savefig(out_filebase + '.png', transparent=True)
    f.savefig(out_filebase + '.pdf', transparent=True)


def calc_barrier_heights_and_errors(lfes, lfe_stds, temp):
    barrier_i = find_barrier(lfes, temp)
    if barrier_i == -1:
        return (np.nan, np.nan), (np.nan, np.nan)

    minima, minima_i = find_minima(lfes, barrier_i)
    lower_peak = lfes[barrier_i] - minima[0]
    lower_std = np.sqrt(lfe_stds[barrier_i]**2 + lfe_stds[minima_i[0]]**2)
    upper_peak = lfes[barrier_i] - minima[1]
    upper_std = np.sqrt(lfe_stds[barrier_i]**2 + lfe_stds[minima_i[1]]**2)

    return (lower_peak, upper_peak), (lower_std, upper_std)


def find_minima(lfes, maximum_i):
    lower_lfes = lfes[:maximum_i]
    upper_lfes = lfes[maximum_i:]
    minima = (lower_lfes.min(), upper_lfes.min())
    minima_i = (lower_lfes.idxmin(), upper_lfes.idxmin())

    return minima, minima_i


def find_barrier(lfes, temp):
    # Find largest maximum
    maxima_i = argrelextrema(np.array(lfes), np.greater)[0]
    if len(maxima_i) == 0:
        print('No barrier detected at {}'.format(temp))
        return -1

    maxima = lfes[maxima_i]
    maximum_i = maxima.idxmax()

    return maximum_i


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
            'input_dir',
            type=str,
            help='Directory of inputs')
    parser.add_argument(
            'output_filebase',
            type=str,
            help='Output filebase')
    parser.add_argument(
            'tag',
            type=str,
            help='OP tag')
    parser.add_argument('assembled_op',
            type=int,
            help='Value of order parameter in assembled state')
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
            '--x_range',
            nargs='+',
            type=int,
            help='Control variable values')

    return parser.parse_args()


if __name__ == '__main__':
    main()
