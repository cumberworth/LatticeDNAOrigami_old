#!/usr/bin/env python

"""Calculate total number of fully bound staples for a simulation set"""


import argparse
import os.path

import numpy as np

from origamipy import biases
from origamipy import conditions
from origamipy import datatypes
from origamipy import outputs
from origamipy import decorrelate
from origamipy import mbar_wrapper


def main():
    args = parse_args()
    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(args, fileformatter)
    inp_filebase = create_input_filepathbase(args)
    sim_collections = outputs.create_sim_collections(inp_filebase,
            all_conditions, args.reps)

    tag = 'numfullyboundstaples'
    for sim_collection in sim_collections:
        staple_states = sim_collection.get_reps_data('staplestates', concatenate=False)
        for rep in sim_collection._reps:
            runs = len(staple_states[rep])
            for run in range(runs):
                back_ops_filebase = sim_collection.get_filebase(run, rep)
                back_ops = datatypes.OrderParams.from_file(back_ops_filebase)
                total_staples = staple_states[rep][run]._data[1:, :].sum(axis=0)
                if tag in back_ops.tags:
                    back_ops[tag] = total_staples
                else:
                    back_ops.add_column(tag, total_staples)

                back_ops.to_file(back_ops_filebase)


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
        'staple_m',
        type=float,
        help='Staple molarity (mol/V)')
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
