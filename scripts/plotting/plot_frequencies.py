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

    op_tag = 'numfullyboundstaples'
    tagbase = 'staplestates'
    tags = ['{}{}'.format(tagbase, i + 1) for i in range(args.stapletypes)]
    inp_filebase = '{}/{}-{}-{}'.format(args.input_dir, args.system,
            args.vari, op_tag)
    figsize = (plot.cm_to_inches(18), plot.cm_to_inches(12))
    index_to_stapletype = np.loadtxt(args.mapfile, dtype=int)
    for op_value in range(1, args.stapletypes + 1):
        out_filebase = '{}/{}-{}_{}-{}-{}-freqs'.format(args.output_dir, args.system,
                args.vari, op_tag, op_value, tagbase)

        plot.set_default_appearance()
        f = plt.figure(figsize=figsize, dpi=300)
        gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1])
        gs_main = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[:, :])
        ax = f.add_subplot(gs_main[0])
        ax.axis('off')

        aves, stds = plot.read_expectations(inp_filebase)
        reduced_aves = aves[aves[op_tag] == op_value]
        freqs = [reduced_aves[t] for t in tags]
        freq_array = fill_assembled_shape_array(freqs, index_to_stapletype)

        # Plot simulation melting points
        min_t = np.min(0)
        max_t = np.max(1)
        im = ax.imshow(freq_array, vmin=min_t, vmax=max_t)
        plt.colorbar(im, orientation='horizontal')
        plt.tight_layout(pad=0.5, h_pad=0, w_pad=0)
        f.savefig(out_filebase + '.png', transparent=True)
        f.savefig(out_filebase + '.pdf', transparent=True)


def fill_assembled_shape_array(flat_array, index_to_stapletype):
    rows = index_to_stapletype.shape[0]
    cols = index_to_stapletype.shape[1]
    assembled_array = np.zeros([rows, cols])
    for row, staple_types in enumerate(index_to_stapletype):
        for col, staple_type in enumerate(staple_types):
            assembled_array[row, col] = flat_array[staple_type - 1]

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
            'stapletypes',
            type=int,
            help='Number of staple types')
    parser.add_argument(
            'mapfile',
            type=str,
            help='Index-to-staple type map filename')

    return parser.parse_args()


if __name__ == '__main__':
    main()
