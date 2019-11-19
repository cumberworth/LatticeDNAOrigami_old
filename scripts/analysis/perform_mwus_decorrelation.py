#!/usr/bin/env python

"""Perform decorrelation on MWUS simulation output."""

import argparse
import json

from origamipy import biases
from origamipy import conditions
from origamipy import decorrelate
from origamipy import files
from origamipy import outputs
from origamipy import us_process


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    inp_filebase = create_input_filepathbase(args)
    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(
        args, fileformatter, inp_filebase, system_file)
    sim_collections = create_simplesim_collections(args, inp_filebase,
                                                   all_conditions)
    decor_outs = decorrelate.SimpleDecorrelatedOutputs(
        sim_collections, all_conditions)
    decor_outs.perform_decorrelation(args.skip)
    decor_outs.apply_masks()
    decor_outs.write_decors_to_files()


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
        filebase = '{}_run-{}_rep-{}{}'.format(inp_filebase, args.run, rep,
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
        'skip',
        type=int,
        help='Number of steps to skip')
    parser.add_argument(
        'reps',
        type=int,
        help='Number of reps')
    parser.add_argument(
        'run',
        type=int,
        help='Number of reps')

    return parser.parse_args()


if __name__ == '__main__':
    main()
