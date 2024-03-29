#!/usr/bin/python

"""Plot frequencies of configurations at given order parameter"""

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

    op_tag = 'numfulldomains'
    tagbase = 'domainstate'
    tags = ['{}{}'.format(tagbase, i) for i in range(args.scaffolddomains)]
    inp_filebase = '{}/{}-{}'.format(args.input_dir, args.filebase, op_tag)
    index_to_domaintype = np.loadtxt(args.mapfile, dtype=int)
    figsize = (plot.cm_to_inches(18), plot.cm_to_inches(12))
    for op_value in range(1, args.scaffolddomains + 1):
        out_filebase = '{}/{}_{}-{}-{}-freqs'.format(
                args.output_dir, args.filebase, op_tag, op_value, tagbase)

        plot.set_default_appearance()
        f = plt.figure(figsize=figsize, dpi=300)
        gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
        gs_main = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, :])
        ax = f.add_subplot(gs_main[0])
        ax.axis('off')

        aves, stds = plot.read_expectations(inp_filebase)
        if not op_value in aves[op_tag].values:
            continue

        reduced_aves = aves[aves[op_tag] == op_value]
        freqs = [reduced_aves[t] for t in tags]
        freq_array = fill_assembled_shape_array(freqs, index_to_domaintype)

        # Plot simulation melting points
        min_t = np.min(0)
        max_t = np.max(1)
        im = ax.imshow(freq_array, vmin=min_t, vmax=max_t)
        plt.colorbar(im, orientation='horizontal')
        plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
        f.savefig(out_filebase + '.png', transparent=True)
        f.savefig(out_filebase + '.pdf', transparent=True)


def fill_assembled_shape_array(flat_array, index_to_domaintype):
    rows = index_to_domaintype.shape[0]
    cols = index_to_domaintype.shape[1]
    assembled_array = np.zeros([rows, cols])
    for row, domain_types in enumerate(index_to_domaintype):
        for col, domain_type in enumerate(domain_types):
            assembled_array[row, col] = flat_array[domain_type]

    return assembled_array


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
            'input_dir',
            type=str,
            help='Directory of inputs')
    parser.add_argument(
            'output_dir',
            type=str,
            help='Output directory')
    parser.add_argument(
            'filebase',
            type=str,
            help='Filebase')
    parser.add_argument(
            'scaffolddomains',
            type=int,
            help='Number of scaffold domains')
    parser.add_argument(
            'mapfile',
            type=str,
            help='Index-to-domain type map filename')

    return parser.parse_args()


if __name__ == '__main__':
    main()
