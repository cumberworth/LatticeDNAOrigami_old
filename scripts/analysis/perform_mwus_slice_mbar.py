#!/usr/bin/env python

"""Carry out standard MBAR analysis on MWUS simulation output."""

import argparse
import json

import numpy as np

from origamipy import biases
from origamipy import conditions
from origamipy import decorrelate
from origamipy import files
from origamipy import mbar_wrapper
from origamipy import outputs
from origamipy import us_process
from origamipy import utility


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    staple_lengths = system_file._staple_lengths
    inp_filebase = create_input_filepathbase(args)
    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(
        args, fileformatter, inp_filebase, system_file)
    out_filebase = create_output_filepathbase(args)
    conds = conditions.SimConditions({'temp': args.temp, 'staple_m': args.staple_m,
                                      'bias': biases.NoBias()}, fileformatter, staple_lengths)

    # Expecations along OP slices
    mbarws = []
    all_decor_outs = []
    sampled_ops = []
    for i in range(1, args.assembled_op + 1):
        sim_collections = create_simplesim_collections(args, inp_filebase,
                                                       all_conditions)
        decor_outs = decorrelate.SimpleDecorrelatedOutputs(
            sim_collections, all_conditions)
        decor_outs.read_decors_from_files()
        filtered_count = decor_outs.filter_collections(args.tag, i)
        if filtered_count == 0:
            continue

        sampled_ops.append(i)
        all_decor_outs.append(decor_outs)
        mbarw = mbar_wrapper.MBARWrapper(decor_outs)
        mbarw.perform_mbar()
        mbarws.append(mbarw)

    # Calculate expectations across selected order parameter
    all_tags = []
    for i in range(1, args.staple_types + 1):
        all_tags.append('staples{}'.format(i))
        all_tags.append('staplestates{}'.format(i))

    for i in range(args.scaffold_domains):
        all_tags.append('domainstate{}'.format(i))

    aves, stds = calc_reduced_expectations(
        conds, mbarws, all_decor_outs, all_tags)

    aves = np.concatenate([[sampled_ops], np.array(aves).T])
    aves_file = files.TagOutFile('{}-{}.aves'.format(out_filebase, args.tag))
    aves_file.write([args.tag] + all_tags, aves.T)

    stds = np.concatenate([[sampled_ops], np.array(stds).T])
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


def construct_fileformatter():
    specs = [conditions.ConditionsFileformatSpec('bias', '{}')]
    return conditions.ConditionsFileformatter(specs)


def construct_conditions(args, fileformatter, inp_filebase, system_file):
    bias_tags, windows = us_process.read_windows_file(args.windows_filename)
    bias_functions = json.load(open(args.bias_functions_filename))
    op_tags = us_process.get_op_tags_from_bias_functions(
        bias_functions, bias_tags)

    # Linear square well functions are all the same
    for bias_function in bias_functions['origami']['bias_functions']:
        if bias_function['type'] == 'LinearStepWell':
            slope = bias_function['slope']
            min_outside_bias = bias_function['min_bias']

    grid_biases = []
    for window in windows:
        for rep in range(args.reps):
            filebase = '{}_run-{}_rep-{}'.format(inp_filebase, args.run, rep)
            grid_biases.append(biases.GridBias(op_tags, window,
                                               min_outside_bias, slope, args.temp, filebase))

    conditions_map = {'temp': [args.temp],
                      'staple_m': [args.staple_m],
                      'bias': grid_biases}

    # either get rid of this too or make a list of filebases for creating sim collections
    return conditions.AllSimConditions(conditions_map, fileformatter, system_file)


def create_simplesim_collections(args, inp_filebase, all_conditions):
    sim_collections = []
    rep = 0
    for conditions in all_conditions:
        filebase = '{}_run-{}_rep-{}{}_decor'.format(inp_filebase, args.run, rep,
                                                     conditions.fileformat)
        sim_collection = outputs.SimpleSimCollection(
            filebase, conditions, args.reps)
        sim_collections.append(sim_collection)
        rep += 1
        rep %= args.reps

    return sim_collections


def create_input_filepathbase(args):
    return '{}/{}'.format(args.input_dir, args.filebase)


def create_output_filepathbase(args):
    return '{}/{}'.format(args.output_dir, args.filebase)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'system_filename',
        type=str,
        help='System file')
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
        'windows_filename',
        type=str,
        help='Windows filename')
    parser.add_argument(
        'bias_functions_filename',
        type=str,
        help='Bias functions filename')
    parser.add_argument(
        'temp',
        type=float,
        help='Temperature (K)')
    parser.add_argument(
        'staple_m',
        type=float,
        help='Staple molarity (mol/V)')
    parser.add_argument(
        'stack_ene',
        type=float,
        help='Stacking energy (kb K)')
    parser.add_argument(
        'reps',
        type=int,
        help='Number of reps')
    parser.add_argument(
        'run',
        type=int,
        help='Number of reps')
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
    parser.add_argument('--tags',
                        nargs='+',
                        type=str,
                        help='Order parameter tags')
    parser.add_argument('--tag_pairs',
                        nargs='+',
                        type=str,
                        help='Tags to calculate 2D pmf for (comma delim)')

    return parser.parse_args()


if __name__ == '__main__':
    main()
