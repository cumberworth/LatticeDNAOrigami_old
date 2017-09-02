#!/usr/bin/env python3

"""Marginalize and output simulated and enumerated weights for plotting"""

import argparse
import pickle
import numpy as np
from operator import itemgetter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filebase', type=str, help='Filebase')

    args = parser.parse_args()
    filebase = args.filebase

    # Marginalize
    staple_weights = {}
    staple_weights = marginalize_single(weights, staple_weights, STAPLES_I)
    sd_weights = {}
    sd_weights = marginalize_2d_single(weights, sd_weights, [STAPLES_I, DOMAINS_I])

    staples, staple_weights = order_weights(staple_weights)
    staple_filename = filebase + '_weights.staples'
    write_pgf_weights(staple_filename, staples, staple_weights)
    staple_pmfs = calc_pmf(staple_weights)
    staple_filename = filebase + '_pmfs.staples'
    write_pgf_weights(staple_filename, staples, staple_pmfs)

    sd_weights = fill_weights(sd_weights, [0, max(staples)], [0, max(domains)])
    sds, sd_weights = order_weights(sd_weights)
    sd_pmfs = calc_pmf(sd_weights)
    sd_filename = filebase + '_pmfs.sds'
    write_2d_pgf_weights(sd_filename, sds, sd_pmfs)


if __name__ == '__main__':
    main()
