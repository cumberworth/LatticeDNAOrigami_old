#!/usr/bin/env python

"""Calculate total number of fully bound staples for a simulation set"""


import argparse
import os.path

import numpy as np

from origamipy import biases
from origamipy import conditions
from origamipy import config_process
from origamipy import datatypes
from origamipy import files
from origamipy import outputs
from origamipy import decorrelate
from origamipy import mbar_wrapper


def main():
    args = parse_args()
    system_file = files.JSONStructInpFile(args.system_filename)
    domain_pairs = parse_domain_pairs(args.domain_pairs)

    fileformatter = construct_fileformatter()
    all_conditions = construct_conditions(args, fileformatter)
    inp_filebase = create_input_filepathbase(args)
    sim_collections = outputs.create_sim_collections(inp_filebase,
            all_conditions, args.reps)
    for sim_collection in sim_collections:
        for rep in sim_collection._reps:
            ops = sim_collection.get_reps_data('ops', concatenate=False)
            runs = len(ops[rep])
            for run in range(runs):
                run_filebase = sim_collection.get_filebase(run, rep)
                trj_filename = '{}.trj'.format(run_filebase)
                trj_file = files.TxtTrajInpFile(trj_filename, system_file)
                ops = datatypes.OrderParams.from_file(run_filebase)
                all_dists = [[] for i in range(len(domain_pairs))]
                for i, step in enumerate(trj_file):
                    config = np.array(step[0]['positions'])
                    for j, domain_pair in enumerate(domain_pairs):
                        pos_i = config[domain_pair[0]]
                        pos_j = config[domain_pair[1]]
                        all_dists[j].append(config_process.calc_dist(pos_i, pos_j))

                for domain_pair, dists in zip(domain_pairs, all_dists):
                    dist_tag = 'dist-d{}-d{}'.format(domain_pair[0], domain_pair[1])
                    if dist_tag in ops.tags:
                        ops[dist_tag] = dists
                    else:
                        ops.add_column(dist_tag, dists)
                    adj_tag = 'adj-d{}-d{}'.format(domain_pair[0], domain_pair[1])
                    adj_sites = np.array(dists) == 1
                    if adj_tag in ops.tags:
                        ops[adj_tag] = adj_sites
                    else:
                        ops.add_column(adj_tag, adj_sites)

                dist_sum = np.sum(all_dists, axis=0)
                tag = 'dist-sum'
                if tag in ops.tags:
                    ops[tag] = dist_sum
                else:
                    ops.add_column(tag, dist_sum)

                ops.to_file(run_filebase)


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


def parse_domain_pairs(domain_pair_strings):
    domain_pairs = []
    for domain_pair in domain_pair_strings:
        domain_pairs.append([int(d) for d in domain_pair.split(',')])

    return domain_pairs


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
    parser.add_argument(
            '--domain_pairs',
            nargs='+',
            type=str,
            help='Scaffold domain pairs to calculate distances between')

    return parser.parse_args()


if __name__ == '__main__':
    main()
