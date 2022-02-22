#!/usr/bin/python

"""Plot frequencies of configurations at given order parameter"""

import argparse

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import numpy as np

from matplotlibstyles import styles
from matplotlibstyles import plotutils
from origamipy import plot
from origamipy import utility


def main():
    args = parse_args()
    f = setup_figure()
    gs = gridspec.GridSpec(args.stapletypes - 1, 1, f)
    axes = []
    for i in range(args.stapletypes - 1):
        axes.append(f.add_subplot(gs[i, 0]))

    plot_figure(f, axes, vars(args))
    setup_axes(axes)
    set_labels(f, axes)
    plot_filebase = (f'{args.plot_dir}/{args.filebase}_'
                     f'{args.slice_tag}-{args.tagbase}_freqs')
    save_figure(f, plot_filebase)


def setup_figure():
    styles.set_thin_style()
    figsize = (plotutils.cm_to_inches(5), plotutils.cm_to_inches(20))

    return plt.figure(figsize=figsize, dpi=300, constrained_layout=True)


def plot_figure(f, axes, args):
    input_dir = args['input_dir']
    filebase = args['filebase']
    stapletypes = args['stapletypes']
    slice_tag = args['slice_tag']
    tagbase = args['tagbase']
    mapfile = args['mapfile']

    cmap = cm.get_cmap('viridis')

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
        freq_array = utility.fill_assembled_shape_array(
            freqs, index_to_stapletype)

        # Plot simulation melting points
        ax.imshow(freq_array, vmin=0, vmax=1, cmap=cmap)


def setup_axes(axes, titles=None):
    for i, ax in enumerate(axes):
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[i])


def set_labels(f, axes):
    cmap = cm.get_cmap('viridis')
    mappable = plotutils.create_linear_mappable(cmap, 0, 1)
    cbar = f.colorbar(mappable, ax=axes, orientation='horizontal')
    cbar.set_label('Fraction with fully bound staple')


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
    parser.add_argument(
        '--slice_tag',
        default='numfullyboundstaples',
        type=str,
        help='OP tag to slice along')
    parser.add_argument(
        '--tagbase',
        default='staplestates',
        type=str,
        help='OP tag base')

    return parser.parse_args()


if __name__ == '__main__':
    main()
