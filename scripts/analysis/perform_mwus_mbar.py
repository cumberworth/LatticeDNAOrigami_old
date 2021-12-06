#!/usr/bin/env python

"""Carry out standard MBAR analysis on MWUS simulation output."""

import argparse
import json

import numpy as np
from scipy.signal import argrelextrema
from scipy import interpolate
from scipy.optimize import minimize

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
    staple_lengths = utility.calc_staple_lengths(system_file)
    inp_filebase = f'{args.outs_dir}/{args.filebase}'
    fileformatter = construct_fileformatter()
    reps_all_conditions = conditions.construct_mwus_conditions(
        args.windows_filename, args.bias_functions_filename, args.reps,
        args.start_run, args.temp, args.itr, args.staple_m, fileformatter,
        inp_filebase, staple_lengths, False)
    sim_collections = []
    for rep in range(args.reps):
        rep_sim_collections = outputs.create_sim_collections(
            inp_filebase,
            reps_all_conditions[rep],
            rep,
            args.start_run,
            args.end_run)
        sim_collections.append(rep_sim_collections)

    all_conditions = conditions.construct_mwus_conditions(
        args.windows_filename, args.bias_functions_filename, args.reps,
        args.start_run, args.temp, args.itr, args.staple_m, fileformatter,
        inp_filebase, staple_lengths, True)
    decor_outs = decorrelate.DecorrelatedOutputs(
        sim_collections, all_conditions=all_conditions,
        rep_conditions_equal=False)
    decor_outs.read_decors_from_files()

    mbarw = mbar_wrapper.MBARWrapper(decor_outs)
    mbarw.perform_mbar()

    # Calc LFEs and expectations with simulation temp
    out_filebase = '{}/{}_run-{}-{}_iter-{}'.format(
        args.analysis_dir,
        args.filebase,
        args.start_run,
        args.end_run,
        args.itr)
    conds = conditions.SimConditions(
        {'temp': args.temp,
         'staple_m': args.staple_m,
         'bias': biases.NoBias()},
        fileformatter, staple_lengths)
    se_tags = decor_outs.all_series_tags
    lfes_filebase = f'{out_filebase}_lfes'
    mbarw.calc_all_expectations(out_filebase, se_tags, [conds])
    mbarw.calc_all_1d_lfes(lfes_filebase, se_tags, [conds])

    # Calc melting temp and use for LFEs and expectations
    bias = biases.NoBias()
    melting_temp = est_melting_temp_and_barrier(
        mbarw, fileformatter, staple_lengths, conds, bias, args)
    conds = conditions.SimConditions(
        {'temp': melting_temp,
         'staple_m': args.staple_m,
         'bias': biases.NoBias()},
        fileformatter, staple_lengths)

    exps_filebase = f'{out_filebase}-melting'
    lfes_filebase = f'{out_filebase}_lfes-melting'
    mbarw.calc_all_1d_lfes(lfes_filebase, se_tags, [conds])
    mbarw.calc_all_expectations(exps_filebase, se_tags, [conds])

    # Calc melting temps for other stacking energies
#    for stack_mult in [0, 0.25, 0.5, 0.75]:
#        print()
#        print(f'Stack mult: {stack_mult}')
#        stack_bias = biases.StackingBias(args.stack_ene, stack_mult)
#        conds = conditions.SimConditions(
#            {'temp': args.temp,
#             'staple_m': args.staple_m,
#             'bias': stack_bias},
#            fileformatter, staple_lengths)
#        melting_temp = est_melting_temp_and_barrier(
#            mbarw, fileformatter, staple_lengths, conds, stack_bias, args)
#        conds = conditions.SimConditions(
#            {'temp': melting_temp,
#             'staple_m': args.staple_m,
#             'bias': stack_bias},
#            fileformatter, staple_lengths)
#
#        lfes_filebase = f'{out_filebase}_lfes-melting-stack-{stack_mult}'
#        mbarw.calc_all_1d_lfes(lfes_filebase, se_tags, [conds])
#        mbarw.calc_all_expectations(out_filebase, se_tags, [conds])

    # Calc expectations for a range of temps around melting
    temp_conds = []
    for temp in np.linspace(melting_temp - 5, melting_temp + 5, 100):
        conds = conditions.SimConditions(
            {'temp': temp,
             'staple_m': args.staple_m,
             'bias': biases.NoBias()},
            fileformatter, staple_lengths)
        temp_conds.append(conds)

    exps_filebase = f'{out_filebase}_temps'
    mbarw.calc_all_expectations(exps_filebase, se_tags, temp_conds)

    # Expecations along OP slices
    mbarws = []
    all_decor_outs = []
    sampled_ops = []
    for i in range(1, args.assembled_op + 1):
        sim_collections = []
        for rep in range(args.reps):
            rep_sim_collections = outputs.create_sim_collections(
                inp_filebase,
                reps_all_conditions[rep],
                rep,
                args.start_run,
                args.end_run)
            sim_collections.append(rep_sim_collections)

        decor_outs = decorrelate.DecorrelatedOutputs(
            sim_collections, all_conditions, False)
        decor_outs.read_decors_from_files()

        filtered_count = decor_outs.filter_collections(args.tag, i)
        if filtered_count == 0:
            continue

        sampled_ops.append(i)
        all_decor_outs.append(decor_outs)
        mbarw = mbar_wrapper.MBARWrapper(decor_outs)
        mbarw.perform_mbar()
        mbarws.append(mbarw)

    # Calculate expectations for staple and domain states
    se_tags = []
    num_staple_types = utility.calc_num_staple_types(system_file)
    for i in range(1, num_staple_types + 1):
        se_tags.append(f'staples{i}')
        se_tags.append(f'staplestates{i}')

    #num_scaffold_domains = utility.calc_num_scaffold_domains(system_file)
    #for i in range(num_scaffold_domains):
    #    se_tags.append(f'domainstate{i}')

    conds = conditions.SimConditions(
        {'temp': melting_temp,
         'staple_m': args.staple_m,
         'bias': biases.NoBias()},
        fileformatter, staple_lengths)
    aves, stds = calc_reduced_expectations(
        conds, mbarws, all_decor_outs, se_tags)

    aves = np.concatenate([[sampled_ops], np.array(aves).T])
    aves_file = files.TagOutFile(f'{out_filebase}-{args.tag}.aves')
    aves_file.write([args.tag] + se_tags, aves.T)

    stds = np.concatenate([[sampled_ops], np.array(stds).T])
    stds_file = files.TagOutFile(f'{out_filebase}-{args.tag}.stds')
    stds_file.write([args.tag] + se_tags, stds.T)


def est_melting_temp_and_barrier(
        mbarw,
        fileformatter,
        staple_lengths,
        conds,
        bias,
        args):

#    try:
#        melting_temp = mbarw.estimate_melting_temp(conds, args.temp)
#    except:
    melting_temp = mbarw.estimate_melting_temp_endpoints(conds, args.temp)

    conds = conditions.SimConditions(
        {'temp': melting_temp,
         'staple_m': args.staple_m,
         'bias': bias},
        fileformatter, staple_lengths)

    melting_temp_f = '{:.3f}'.format(np.around(melting_temp, decimals=3))
    print(f'Estimated melting temperature: {melting_temp_f} K')
    for se_tag in ['numfullyboundstaples', 'numfulldomains']:
        lfes, stds, bins = mbarw.calc_1d_lfes(se_tag, conds)
        try:
            barrier_i = mbar_wrapper.find_barrier(lfes)
            barrier_height = mbar_wrapper.calc_forward_barrier_height(lfes)
            print()
            print(f'Barrier height, {se_tag}: {barrier_height:.3f} kT')
            print(f'Barrier peak, {se_tag}: {bins[barrier_i]:.3f}')
        except:
            pass

    return melting_temp


def calc_reduced_expectations(conds, mbarws, all_decor_outs, se_tags):
    all_aves = []
    all_stds = []
    for mbarw, decor_outs in zip(mbarws, all_decor_outs):
        aves = []
        stds = []
        for se_tag in se_tags:
            ave, std = mbarw.calc_expectation(se_tag, conds)
            aves.append(ave)
            stds.append(std)

        all_aves.append(aves)
        all_stds.append(stds)

    return all_aves, all_stds


def parse_tag_pairs(tag_pairs):
    return [tuple(tag_pair.split(',')) for tag_pair in tag_pairs]


def construct_fileformatter():
    specs = [conditions.ConditionsFileformatSpec('bias', '{}')]
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
        'analysis_dir',
        type=str,
        help='analysis directory')
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
        'start_run',
        type=int,
        help='Run to concatenate from')
    parser.add_argument(
        'end_run',
        type=int,
        help='Run to concatenate to')
    parser.add_argument(
        'itr',
        type=int,
        help='US iteration')
    parser.add_argument(
        'tag',
        type=str,
        help='Order parameter tag')
    parser.add_argument(
        'assembled_op',
        type=int,
        help='Value of order parameter in assembled state')
    parser.add_argument(
        '--tags',
        nargs='+',
        type=str,
        help='Order parameter tags')
    parser.add_argument(
        '--tag_pairs',
        nargs='+',
        type=str,
        help='Tags to calculate 2D pmf for (comma delim)')

    return parser.parse_args()


if __name__ == '__main__':
    main()
