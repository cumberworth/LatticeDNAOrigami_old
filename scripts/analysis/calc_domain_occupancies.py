#!/usr/bin/env python

"""Calculate scaffold domain occupancies for a simulation set"""


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
    inp_filebase = f'{args.outs_dir}/{args.filebase}'
    fileformatter = construct_fileformatter()
    all_conditions = conditions.construct_remc_conditions(
        args.temps, args.staple_m, fileformatter, staple_lengths)
    sim_collections = []
    for rep in range(args.reps):
        rep_sim_collections = outputs.create_sim_collections(
            inp_filebase, all_conditions, rep)
        sim_collections.append(rep_sim_collections)

    for rep_sim_collections in sim_collections:
        for sim_collection in rep_sim_collections:
            ops = sim_collection.get_data('ops', concatenate=False)
            runs = len(ops)
            for run in range(runs):
                run_filebase = sim_collection.get_filebase(run)
                states_filename = f'{run_filebase}.states'
                states = np.loadtxt(states_filename)[:, :args.scaffold_domains]
                states = states == 2
                ops = datatypes.OrderParams.from_file(run_filebase)
                for i in range(args.scaffold_domains):
                    tag = f'domainstate{i}'
                    if tag in ops.tags:
                        ops[tag] = states[:, i]
                    else:
                        ops.add_column(tag, states[:, i])

                out_filebase = run_filebase + '_mod'
                ops.to_file(run_filebase)


def construct_conditions(args, fileformatter, system_file):
    conditions_map = {'temp': args.temps,
                      'staple_m': [args.staple_m],
                      'bias': [biases.NoBias()]}

    return conditions.AllSimConditions(conditions_map, fileformatter, system_file)


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
        'outs_dir',
        type=str,
        help='outs directory')
    parser.add_argument(
        'staple_m',
        type=float,
        help='Staple molarity (mol/V)')
    parser.add_argument(
        'scaffold_domains',
        type=int,
        help='Number of scaffold domains')
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
