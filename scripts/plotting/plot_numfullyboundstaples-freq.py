#!/usr/bin/python

"""Plot frequencies of configurations at given order parameter"""

import argparse

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import numpy as np

from matplotlibstyles import styles
from origamipy import plot
from origamipy import utility


def main():
    args = parse_args()
    f = setup_figure()
    gs = gridspec.GridSpec(
        args.stapletypes - 1, 1, f, height_ratios=[1]*(args.stapletypes - 1))
    axes = []
    for i in range(args.stapletypes - 1):
        axes.append(f.add_subplot(gs[i, 0]))

    slice_tag = 'numfullyboundstaples'
    tagbase = 'staplestates'
    plot_figure(
        f, axes, args.input_dir, args.filebase, args.stapletypes, slice_tag,
        tagbase, args.mapfile)
    setup_axes(axes)
    plot_filebase = (f'{args.plot_dir}/{args.filebase}_{slice_tag}-{tagbase}_'
        'freqs')
    save_figure(f, plot_filebase)


def setup_figure():
    styles.set_thin_style()
    figsize = (styles.cm_to_inches(5), styles.cm_to_inches(20))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(
        f, axes, input_dir, filebase, stapletypes, slice_tag, tagbase,
        mapfile):

    tags = [f'{tagbase}{i + 1}' for i in range(stapletypes)]
    inp_filebase = f'{input_dir}/{filebase}-{slice_tag}'
    index_to_stapletype = np.loadtxt(mapfile, dtype=int)
    for i in range(stapletypes - 1):
        op_value = i + 1
        aves, stds = plot.read_expectations(inp_filebase)
        if op_value not in aves[slice_tag].values:
            print('Missing value')
            sys.exit()

        ax = axes[i]
        reduced_aves = aves[aves[slice_tag] == op_value]
        freqs = [reduced_aves[t] for t in tags]
        freq_array = utility.fill_assembled_shape_array(freqs, index_to_stapletype)

        # Plot simulation melting points
        min_t = np.min(0)
        max_t = np.max(1)
        cmap = cm.get_cmap('viridis')
        mappable = styles.create_linear_mappable(cmap, min_t, max_t)
        im = ax.imshow(freq_array, vmin=min_t, vmax=max_t, cmap=cmap)

#    plt.colorbar(im, orientation='horizontal')


def setup_axes(axes):
    for ax in axes:
        ax.axis('off')


def save_figure(f, plot_filebase):
    # f.savefig(plot_filebase + '.pgf', transparent=True)
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
        'plot_dir',
        type=str,
        help='Output directory')
    parser.add_argument(
        'filebase',
        type=str,
        help='Filebase')
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
