#!/usr/bin/python

"""Plot an order parameter from multiple simulations."""

import argparse

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import numpy as np
from scipy import interpolate

from matplotlibstyles import styles
import origamipy.nearest_neighbour as nn
from origamipy import plot


def main():
    args = parse_args()
    f = setup_figure()
    gs = gridspec.GridSpec(1, 1, f)
    ax = f.add_subplot(gs[0])
    plot_figure(f, ax, vars(args))
    setup_axis(ax, args.tag)
#    set_labels(ax, ax)
    save_figure(f, args.plot_filebase)


def setup_figure():
    styles.set_thin_style()
    figsize = (styles.cm_to_inches(14), styles.cm_to_inches(11))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, ax, args):
    systems = args['systems']
    varis = args['varis']
    input_dir = args['input_dir']
    tag = args['tag']
    assembled_values = args['assembled_values']
    posts = args['posts']
    nncurves = args['nncurves']
    staple_M = args['staple_M']
    binds = args['binds']
    bindh = args['bindh']
    stackene = args['stackene']
    contins = args['continuous']

    cmap = cm.get_cmap('tab10')

    for i in range(len(systems)):
        system = systems[i]
        vari = varis[i]
        assembled_value = assembled_values[i]
        if posts is not None:
            post = posts[i]
        else:
            post = ''

        if nncurves is not None:
            nncurve = nncurves[i]
        else:
            nncurve = False

        if contins is not None:
            contin = contins[i]
        else:
            contin = False

        ax.axhline(assembled_value, linestyle='--', color=cmap(i))

        inp_filebase = f'{input_dir}/{system}-{vari}{post}'
        all_aves, all_stds = plot.read_expectations(inp_filebase)
        temps = all_aves['temp']
        means = all_aves[tag]
        stds = all_stds[tag]
        if nncurve:
            fracs = nn.calc_excess_bound_fractions(bindh, binds, staple_M, 10)
            interpolated_temp = interpolate.interp1d(means, temps, kind='linear')
            halfway_temp = interpolated_temp(assembled_value/2)
            occ_temps = np.linspace(halfway_temp - 10, halfway_temp + 10, 50)
            ax.plot(occ_temps, fracs*assembled_value, color=cmap(i))

        if contin:
            ax.fill_between(temps, means + stds, means - stds, label=vari,
                            color='0.8')
            ax.plot(temps, means, marker='None', label=vari, color=cmap(i))
        else:
            ax.errorbar(temps, means, yerr=stds, marker='o', label=vari,
                        color=cmap(i))


def setup_axis(ax, ylabel):
    ax.set_xlabel(r'$T / K$')
    ax.set_ylabel(ylabel)


def set_labels(ax):
    ax.set_axis_off()
    handles, labels = ax.get_legend_handles_labels()
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
        'tag',
        type=str,
        help='OP tag')
    parser.add_argument(
        '--assembled_values',
        nargs='+',
        type=int,
        help='Values of OP in assembled state')
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
        '--posts',
        nargs='+',
        type=str,
        help='Extra part of mean name (e.g. _temps for MWUS extrapolation')
    parser.add_argument(
        '--nncurves',
        nargs='+',
        type=bool,
        help='Include shifted NN curve')
    parser.add_argument(
        '--staple_M',
        default='',
        type=float,
        help='Staple concentration')
    parser.add_argument(
        '--binds',
        default='',
        type=float,
        help='Domain binding entropy')
    parser.add_argument(
        '--bindh',
        default='',
        type=float,
        help='Domain binding enthalpy')
    parser.add_argument(
        '--stackene',
        default='',
        type=float,
        help='Stacking energy')
    parser.add_argument(
        '--continuous',
        nargs='+',
        type=bool,
        help='Plot curves as continuous')

    return parser.parse_args()


if __name__ == '__main__':
    main()
