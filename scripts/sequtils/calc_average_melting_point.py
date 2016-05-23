#!/usr/bin/env python

"""Calculate average melting temperature for a given system and strand []."""

import argparse
import sys

sys.path.insert(0, '../../lib/lattice_origami_domains')

from lattice_dna_origami.lattice_origami_domains import *
from lattice_dna_origami.nearest_neighbour import *

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Configuration file name.')
parser.add_argument('strand_M', type=float, help='Staple concentration (M).')
parser.add_argument('cation_M', type=float, help='Cation concentration (M).')

args = parser.parse_args()
config_filename = args.filename
strand_M = args.strand_M
cation_M = args.cation_M

input_file = JSONInputFile(config_filename)

# Calculate all melting points
melting_points = []
for seq in input_file.sequences:
    melting_point = calc_melting_point(seq, strand_M, cation_M)
    melting_points.append(melting_point)

# Averages
mean_T = np.mean(melting_points)
min_T = min(melting_points)
max_T = max(melting_points)

print('Average melting temperature: {:.1} K'.format(mean_T))
print('Maximum melting temperature: {:.1} K'.format(max_T))
print('Minimum melting temperature: {:.1} K'.format(min_T))
