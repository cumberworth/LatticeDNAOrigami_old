#!/usr/bin/python

"""Plot melting temperartures as heatmap."""

import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy import interpolate
import colorcet

from origamipy import plot
from origamipy import nearest_neighbour
from origamipy import files


def main():
    args = parse_args()
    inp_filebase = '{}/{}-{}'.format(args.input_dir, args.system, args.vari)
    out_filebase = '{}/{}-{}_domainstate-melting'.format(args.output_dir,
            args.system, args.vari)
    index_to_domaintype = np.loadtxt(args.mapfile, dtype=int)
    figsize = (plot.cm_to_inches(18), plot.cm_to_inches(12))

    plot.set_default_appearance()
    f = plt.figure(figsize=figsize, dpi=300)
    gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
    gs_main = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, :])
    ax = f.add_subplot(gs_main[0])
    ax.axis('off')

    aves, stds = plot.read_expectations(inp_filebase)
    temps = aves.temp

    melting_points = estimate_melting_points(args.scaffolddomains, aves, temps)
    melting_array = fill_assembled_shape_array(melting_points, index_to_domaintype)

    # Plot simulation melting points
    min_t = np.min(melting_array)
    max_t = np.max(melting_array)
    im = ax.imshow(melting_array, vmin=min_t, vmax=max_t)
    plt.colorbar(im, orientation='horizontal')
    plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
    f.savefig(out_filebase + '.png', transparent=True)
    f.savefig(out_filebase + '.pdf', transparent=True)


def estimate_melting_points(num_domain_types, aves, temps):
    melting_points = []
    for domain_i in range(num_domain_types):
        tag = 'domainstate{}'.format(domain_i)
        melting_points.append(estimate_melting_point(aves[tag], temps))

    return np.array(melting_points)


def estimate_melting_point(ave_states, temps):
    if ave_states.iloc[-1] >= 0.5:
        return np.NaN

    return float(interpolate.interp1d(ave_states, temps)(0.5))


def fill_assembled_shape_array(flat_array, index_to_domaintype):
    rows = index_to_domaintype.shape[0]
    cols = index_to_domaintype.shape[1]
    assembled_array = np.zeros([rows, cols])
    for row, domain_types in enumerate(index_to_domaintype):
        for col, domain_type in enumerate(domain_types):
            assembled_array[row, col] = flat_array[domain_type]

    return assembled_array


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
            'scaffolddomains',
            type=int,
            help='Number of scaffold binding domains')
    parser.add_argument(
            'mapfile',
            type=str,
            help='Index-to-staple type map filename')

    return parser.parse_args()


if __name__ == '__main__':
    main()
