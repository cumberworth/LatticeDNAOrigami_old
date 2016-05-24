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

print('Unbound melting temperatures:')
print('Average: {:.1f} K'.format(mean_T))
print('Maximum: {:.1f} K'.format(max_T))
print('Minimum: {:.1f} K'.format(min_T))
print()

# Calculate internal melting temperatures
internal_melting_points = []
for seq in input_file.sequences:
    internal_melting_point = calc_internal_melting_point(seq, cation_M)
    internal_melting_points.append(internal_melting_point)

# Averages
mean_iT = np.mean(internal_melting_points)
min_iT = min(internal_melting_points)
max_iT = max(internal_melting_points)

print('Bound melting temperatures:')
print('Average: {:.1f} K'.format(mean_iT))
print('Maximum: {:.1f} K'.format(max_iT))
print('Minimum: {:.1f} K'.format(min_iT))
