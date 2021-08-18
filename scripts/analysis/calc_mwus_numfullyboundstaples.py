#!/usr/bin/env python

"""Calculate total number of fully bound staples for a simulation set"""


import argparse
import json
import os.path

import numpy as np

from origamipy import biases
from origamipy import conditions
from origamipy import datatypes
from origamipy import files
from origamipy import outputs
from origamipy import decorrelate
from origamipy import mbar_wrapper
from origamipy import us_process


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    inp_filebase = '{}/{}'.format(args.input_dir, args.filebase)
    fileformatter = construct_fileformatter()
    sim_collections = []
    for rep in range(args.reps):
        all_conditions = construct_conditions(
            args, fileformatter, inp_filebase, rep, system_file)
        rep_sim_collections = create_simplesim_collections(
            args, inp_filebase, rep, all_conditions)
        sim_collections.append(rep_sim_collections)

    tag = 'numfullyboundstaples'
    for rep_sim_collections in sim_collections:
        for sim_collection in rep_sim_collections:
            staple_states = sim_collection.get_data('staplestates')
            back_ops = datatypes.OrderParams.from_file(sim_collection.filebase)
            total_staples = staple_states._data[1:, :].sum(axis=0)
            if tag in back_ops.tags:
                back_ops[tag] = total_staples
            else:
                back_ops.add_column(tag, total_staples)

            back_ops.to_file(sim_collection.filebase)


def construct_fileformatter():
    specs = [conditions.ConditionsFileformatSpec('bias', '{}')]
    return conditions.ConditionsFileformatter(specs)


def construct_conditions(args, fileformatter, inp_filebase, rep, system_file):
    bias_tags, windows = us_process.read_windows_file(args.windows_filename)
    bias_functions = json.load(open(args.bias_functions_filename))
    op_tags = us_process.get_op_tags_from_bias_functions(
        bias_functions, bias_tags)

    # Linear square well functions are all the same
    for bias_function in bias_functions['origami']['bias_functions']:
        if bias_function['type'] == 'LinearStepWell':
            slope = bias_function['slope']
            min_outside_bias = bias_function['min_bias']
            break

    grid_biases = []
    for window in windows:
        filebase = '{}_run-{}_rep-{}'.format(
            inp_filebase, args.run, rep)
        grid_biases.append(
            biases.GridBias(
                op_tags, window, min_outside_bias, slope, args.temp,
                filebase, args.itr))

    conditions_keys = ['temp', 'staple_m', 'bias']
    conditions_values = [[args.temp], [args.staple_m], grid_biases]

    return conditions.AllSimConditions(
            conditions_keys, [conditions_values], fileformatter, system_file)


def create_simplesim_collections(args, inp_filebase, rep, all_conditions):
    sim_collections = []
    for conditions in all_conditions:
        filebase = '{}_run-{}_rep-{}{}'.format(inp_filebase, args.run, rep,
                                               conditions.fileformat)
        sim_collection = outputs.SimpleSimCollection(filebase, conditions)
        sim_collections.append(sim_collection)

    return sim_collections


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
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
        'reps',
        type=int,
        help='Number of reps')
    parser.add_argument(
        'run',
        type=int,
        help='Number of reps')
    parser.add_argument(
        'itr',
        type=int,
        help='US iteration')

    return parser.parse_args()


if __name__ == '__main__':
    main()
