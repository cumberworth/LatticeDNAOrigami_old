#!/usr/bin/env python

"""Calculate expectation values across a given order parameter"""

import argparse
import sys

import numpy as np
from scipy.signal import argrelextrema
from scipy import interpolate
from scipy.optimize import minimize

from origamipy import biases
from origamipy import conditions
from origamipy import files
from origamipy import outputs
from origamipy import decorrelate
from origamipy import mbar_wrapper
from origamipy import utility


def main():
    args = parse_args()
    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(args, fileformatter)
    inp_filebase = create_input_filepathbase(args)

    mbarws = []
    all_decor_outs = []
    for i in range(1, args.assembled_op + 1):
        sim_collections = outputs.create_sim_collections(inp_filebase,
                all_conditions, args.reps)
        decor_outs = decorrelate.DecorrelatedOutputs(sim_collections, all_conditions)
        decor_outs.read_decors_from_files(data_only=True)
        decor_outs.filter_collections(args.tag, i)
        all_decor_outs.append(decor_outs)
        mbarw = mbar_wrapper.MBARWrapper(decor_outs)
        mbarw.perform_mbar()
        mbarws.append(mbarw)

    out_filebase = create_output_filepathbase(args)

    # Calculate expectations across selected order parameter
    conds = conditions.SimConditions({'temp': args.temp, 'staple_m': args.staple_m,
            'bias': biases.NoBias()}, fileformatter)
    all_tags = []
    for i in range(1, args.staple_types + 1):
        all_tags.append('staples{}'.format(i))
        all_tags.append('staplestates{}'.format(i))

    for i in range(args.scaffold_domains):
        all_tags.append('domainstate{}'.format(i))

    aves, stds = calc_reduced_expectations(conds, mbarws, all_decor_outs, all_tags)

    ops = np.arange(1, args.assembled_op + 1).astype(float)
    aves = np.concatenate([[ops], np.array(aves).T])
    aves_file = files.TagOutFile('{}-{}.aves'.format(out_filebase, args.tag))
    aves_file.write([args.tag] + all_tags, aves.T)

    stds = np.concatenate([[ops], np.array(stds).T])
    stds_file = files.TagOutFile('{}-{}.stds'.format(out_filebase, args.tag))
    stds_file.write([args.tag] + all_tags, stds.T)


def calc_reduced_expectations(conds, mbarws, all_decor_outs, tags):
    all_aves = []
    all_stds = []
    for mbarw, decor_outs in zip(mbarws, all_decor_outs):
        aves = []
        stds = []
        for tag in tags:
            series = decor_outs.get_concatenated_series(tag)
            ave, std = mbarw.calc_expectation(series, conds)
            aves.append(ave)
            stds.append(std)

        all_aves.append(aves)
        all_stds.append(stds)

    return all_aves, all_stds


def parse_tag_pairs(tag_pairs):
    return [tuple(tag_pair.split(',')) for tag_pair in tag_pairs]


def construct_conditions(args, fileformatter):
    conditions_map = {'temp': args.temps,
                      'staple_m': [args.staple_m],
                      'bias': [biases.NoBias()]}

    return conditions.AllSimConditions(conditions_map, fileformatter)


def construct_fileformatter():
    specs = [conditions.ConditionsFileformatSpec('temp', '{}')]
    return conditions.ConditionsFileformatter(specs)


def create_input_filepathbase(args):
    return '{}/{}'.format(args.input_dir, args.filebase)


def create_output_filepathbase(args):
    return '{}/{}'.format(args.output_dir, args.filebase)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'filebase',
            type=str,
            help='Base name for files')
    parser.add_argument(
            'input_dir',
            type=str,
            help='Directory of inputs')
    parser.add_argument(
            'output_dir',
            type=str,
            help='Directory to output to')
    parser.add_argument(
            'temp',
            type=float,
            help='Temperature')
    parser.add_argument(
            'staple_m',
            type=float,
            help='Staple molarity (mol/V)')
    parser.add_argument(
            'stack_ene',
            type=float,
            help='Stacking energy (kb K)')
    parser.add_argument('tag',
            type=str,
            help='Order parameter tag')
    parser.add_argument('assembled_op',
            type=int,
            help='Value of order parameter in assembled state')
    parser.add_argument('staple_types',
            type=int,
            help='Number of staple types')
    parser.add_argument('scaffold_domains',
            type=int,
            help='Number of scaffold domains')
    parser.add_argument(
            '--reps',
            nargs='+',
            type=int,
            help='Reps (leave empty for all available)')
    parser.add_argument(
            '--temps',
            nargs='+',
            type=str,
            help='Temperatures')

    return parser.parse_args()


if __name__ == '__main__':
    main()
