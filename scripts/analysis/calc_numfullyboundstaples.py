#!/usr/bin/env python

"""Calculate total number of fully bound staples for a simulation set"""


import argparse
import os.path

import numpy as np

from origamipy import biases
from origamipy import conditions
from origamipy import datatypes
from origamipy import files
from origamipy import outputs
from origamipy import decorrelate
from origamipy import mbar_wrapper
from origamipy import utility


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    staple_lengths = utility.calc_staple_lengths(system_file)
    inp_filebase = f'{args.input_dir}/{args.filebase}'
    fileformatter = construct_fileformatter()
    all_conditions = conditions.construct_remc_conditions(
        args.temps, args.staple_m, fileformatter, staple_lengths)
    sim_collections = []
    for rep in range(args.reps):
        rep_sim_collections = outputs.create_sim_collections(
            inp_filebase, all_conditions, rep)
        sim_collections.append(rep_sim_collections)

    tag = 'numfullyboundstaples'
    for rep_sim_collections in sim_collections:
        for sim_collection in rep_sim_collections:
            staple_states = sim_collection.get_data(
                'staplestates', concatenate=False)
            runs = len(staple_states)
            for run in range(runs):
                back_ops_filebase = sim_collection.get_filebase(run)
                back_ops = datatypes.OrderParams.from_file(back_ops_filebase)
                total_staples = staple_states[run]._data[1:, :].sum(axis=0)
                if tag in back_ops.tags:
                    back_ops[tag] = total_staples
                else:
                    back_ops.add_column(tag, total_staples)

                back_ops.to_file(back_ops_filebase)


def construct_fileformatter():
    specs = [conditions.ConditionsFileformatSpec('temp', '{}')]
    return conditions.ConditionsFileformatter(specs)


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
        'staple_m',
        type=float,
        help='Staple molarity (mol/V)')
    parser.add_argument(
        'reps',
        type=int,
        help='Number of reps')
    parser.add_argument(
        '--temps',
        nargs='+',
        type=str,
        help='Temperatures')

    return parser.parse_args()


if __name__ == '__main__':
    main()
