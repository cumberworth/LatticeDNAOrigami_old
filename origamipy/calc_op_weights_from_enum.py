#!/usr/bin/env python3

"""Marginalize and output simulated and enumerated weights for plotting"""

import argparse
import pickle
import numpy as np
from operator import itemgetter

from libanalysis import *


STAPLES_I = 0
DOMAINS_I = 1
DISTS_I = 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase')

    args = parser.parse_args()
    filebase = args.filebase

    # Parse data and pickle raw weights
    weights = parse_out_file(filebase + '.weights')
    pickle_file = open(filebase + '.all', 'wb')
    pickle.dump(weights, pickle_file)

    # Marginalize
    staple_weights = {}
    staple_weights = marginalize_single(weights, staple_weights, STAPLES_I)
    domain_weights = {}
    domain_weights = marginalize_single(weights, domain_weights, DOMAINS_I)
    dist_weights = {}
    dist_weights = marginalize_single(weights, dist_weights, DISTS_I)
    sd_weights = {}
    sd_weights = marginalize_2d_single(weights, sd_weights, [STAPLES_I, DOMAINS_I])

    staples, staple_weights = order_weights(staple_weights)
    staple_filename = filebase + '_weights.staples'
    write_pgf_weights(staple_filename, staples, staple_weights)
    staple_pmfs = calc_pmf(staple_weights)
    staple_filename = filebase + '_pmfs.staples'
    write_pgf_weights(staple_filename, staples, staple_pmfs)

    domains, domain_weights = order_weights(domain_weights)
    domain_filename = filebase + '_weights.domains'
    write_pgf_weights(domain_filename, domains, domain_weights)
    domain_pmfs = calc_pmf(domain_weights)
    domain_filename = filebase + '_pmfs.domains'
    write_pgf_weights(domain_filename, domains, domain_pmfs)

    dists, dist_weights = order_weights(dist_weights)
    dist_filename = filebase + '_weights.dists'
    write_pgf_weights(dist_filename, dists, dist_weights)
    dist_pmfs = calc_pmf(dist_weights)
    dist_filename = filebase + '_pmfs.dists'
    write_pgf_weights(dist_filename, dists, dist_pmfs)

    sd_weights = fill_weights(sd_weights, [0, max(staples)], [0, max(domains)])
    sds, sd_weights = order_weights(sd_weights)
    sd_pmfs = calc_pmf(sd_weights)
    sd_filename = filebase + '_pmfs.sds'
    write_2d_pgf_weights(sd_filename, sds, sd_pmfs)


def parse_out_file(filename):
    """Parse enumeration output file."""
    with open(filename) as inp:
        lines = inp.read().splitlines()

    # First line is header
    points_weights = {}
    for line in lines[1:-1]:
        line = line.split()
        point = (int(line[0][1:]), int(line[1]), int(line[2][:-1]))
        weight = float(line[-1])
        points_weights[point] = weight

    return points_weights


if __name__ == '__main__':
    main()
