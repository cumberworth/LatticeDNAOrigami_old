#!/usr/bin/env python3

"""Marginalize and output simulated and enumerated weights for plotting"""

import argparse
import pickle
import numpy as np
from operator import itemgetter

from libanalysis import *


STAPLES_I = 0
DOMAINS_I = 1
outfile_postfix = ['.staples', '.domains']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase')
    parser.add_argument('run', type=int, help='Run number')
    parser.add_argument('reps', type=int, help='Numer of reps')

    args = parser.parse_args()

    filebase = args.filebase
    run = args.run
    reps = args.reps

    filebase_run = '{}_run-{}'.format(filebase, run)

    staple_weights = {}
    domain_weights = {}
    for rep in range(reps):
        filebase_rep = '{}_rep-{}'.format(filebase_run, rep)
        weights = parse_out_file(filebase_rep + '.out')
        weights = normalize(weights)
        pickle_file = open(filebase_rep + '.all', 'wb')
        pickle.dump(weights, pickle_file)

        staple_weights = marginalize_multiple(weights, staple_weights, rep, STAPLES_I)
        domain_weights = marginalize_multiple(weights, domain_weights, rep, DOMAINS_I)

    staples, staple_rep_weights = order_weights(staple_weights)
    staple_means, staple_stds = calc_mean_std(staple_rep_weights)
    staple_filename = filebase_run + outfile_postfix[STAPLES_I]
    write_pgf_weights_errors(staple_filename, staples, staple_means, staple_stds)

    domains, domain_rep_weights = order_weights(domain_weights)
    domain_means, domain_stds = calc_mean_std(domain_rep_weights)
    domain_filename = filebase_run + outfile_postfix[DOMAINS_I]
    write_pgf_weights_errors(domain_filename, domains, domain_means, domain_stds)


def parse_out_file(filename):
    """Parse output from adaptive US simulation"""
    with open(filename) as inp:
        lines = inp.read().splitlines()

    # Get lines with points and data
    info_lines = []
    for line in reversed(lines):
        if len(line) == 0:
            continue
        if line[0] == 'G':
            break
        else:
            info_lines.append(line)

    # Get points and probabilities out of line
    points_weights = {}
    for line in info_lines:
        split_line = line.split()
        point = (int(split_line[0]), int(split_line[1]))
        weight = float(split_line[-2])
        points_weights[point] = weight

    return points_weights


if __name__ == '__main__':
    main()
