#!/usr/bin/python

"""Plot occupancy curves of each staple type."""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import origamipy.plot as plot


def main():
    args = parse_args()

    tags = ['staplestates{}'.format(i) for i in range(1, args.stapletypes + 1)]
    filebase = '{}/{}-{}'.format(args.input_dir, args.system, args.vari)
    out_filebase = '{}/{}-{}_staplestates-means'.format(args.output_dir,
            args.system, args.vari)
    figsize = (plot.cm_to_inches(18), plot.cm_to_inches(12))

    plot.set_default_appearance()
    f = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[10, 1], height_ratios=[1])
    ax = create_axis(f, gs, args.system)
    aves, stds = plot.read_expectations(filebase)
    if args.rtag:
        aves = aves[aves[args.rtag] == args.rvalue]
        stds = stds[stds[args.rtag] == args.rvalue]

    for j, tag in enumerate(tags):
        xvars = aves[args.xtag]
        ax.errorbar(xvars, aves[tag], yerr=stds[tag], marker='o')

        # Write to file
        plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
        f.savefig(out_filebase + '.png', transparent=True)
        f.savefig(out_filebase + '.pdf', transparent=True)


def create_axis(f, gs, system):
    ax = f.add_subplot(gs[0])
    ax.set_xlabel('$T$ / K')
    ax.set_ylabel('Occupancy')
    ax.set_title(system)

    return ax


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
            '--xtag',
            default='temp',
            type=str,
            help='Dependent variable tag')
    parser.add_argument(
            '--rtag',
            type=str,
            help='Tag to slice on')
    parser.add_argument(
            '--rvalue',
            type=float,
            help='Slice value')

    return parser.parse_args()


if __name__ == '__main__':
    main()
