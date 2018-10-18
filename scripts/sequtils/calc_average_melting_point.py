#!/usr/bin/env python

"""Calculate average melting temperature for a given system and strand []."""

import argparse
import sys

from origamipy.origami_io import *
from origamipy.nearest_neighbour import *

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Configuration file name.')
parser.add_argument('strand_M', type=float, help='Staple concentration (M).')
parser.add_argument('cation_M', type=float, help='Cation concentration (M).')

args = parser.parse_args()
config_filename = args.filename
strand_M = args.strand_M
cation_M = args.cation_M

input_file = JSONInputFile(config_filename)

# Calculate melting points of individual fully bound domains
melting_points = []
for staple in input_file.sequences[1:]:
    for seq in staple:
        melting_point = calc_melting_point(seq, strand_M, cation_M)
        melting_points.append(melting_point)
        print('{:.1f} K'.format(melting_point), end=" ")
    print()
print()

# Averages
mean_T = np.mean(melting_points)
min_T = min(melting_points)
max_T = max(melting_points)

print('Isolated domain melting temperatures:')
print('Average: {:.1f} K'.format(mean_T))
print('Maximum: {:.1f} K'.format(max_T))
print('Minimum: {:.1f} K'.format(min_T))
print()

# Calculate internal melting temperatures
internal_melting_points = []
for seq in input_file.sequences[0]:
    internal_melting_point = calc_internal_melting_point(seq, cation_M)
    internal_melting_points.append(internal_melting_point)

# Averages
mean_iT = np.mean(internal_melting_points)
min_iT = min(internal_melting_points)
max_iT = max(internal_melting_points)

print('Bound staple domain melting temperatures:')
print('Average: {:.1f} K'.format(mean_iT))
print('Maximum: {:.1f} K'.format(max_iT))
print('Minimum: {:.1f} K'.format(min_iT))
print()

# Calculate melting points of fully bound staples
staple_melting_points = []
for staple in input_file.sequences[1:]:
    staple_seq = ''.join(staple)
    staple_melting_point = calc_melting_point(staple_seq, strand_M, cation_M)
    staple_melting_points.append(staple_melting_point)

mean_sT = np.mean(staple_melting_points)
min_sT = min(staple_melting_points)
max_sT = max(staple_melting_points)

print('Whole staple melting temperatures:')
print('Average: {:.1f} K'.format(mean_sT))
print('Maximum: {:.1f} K'.format(max_sT))
print('Minimum: {:.1f} K'.format(min_sT))
print()
