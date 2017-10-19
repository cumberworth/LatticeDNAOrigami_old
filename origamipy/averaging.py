"""Functions for calculating ensemble averages"""

import numpy as np
from operator import itemgetter

from pymbar import timeseries

from origamipy.op_process import read_ops_from_file

def normalize(weights):
    """Normalize given weights"""
    N = 0
    for value in weights.values():
        N += value

    for point, value in weights.items():
        weights[point] = value / N

    return weights


def calc_weights_for_each(ops):
    """Return weights for given values of each op in given steps
    
    Instead of looking at the freqeuency of the combination of all ops, look at
    the freqeuncies of values of individual ops."""
    weights = {}
    for tag, points in ops.items():
        for point in points:
            if points in weights[tag]:
                weights[tag][point] += 1
            else:
                weights[tag][point] = 1

        for point, weight in weights[tag].items():
            weights[tag][point] = weight / len(points)

    return weights


def combine_sim_weights(op_weights, op_weights_per_rep, rep):
    """Combines the weights of one sims ops with multiple others

    Outputs a dictionary with order parameter tags as keys to dictionaries that
    have the order parameter values as keys and the arrays of weights for those
    op values across the simulations being considered as values.
    """
    for tag, weights in weights.items():
        for point, weight in weights.items():
            if point in op_weights_per_rep[tag]:
                op_weights_per_rep[tag][point].append(weight)
            else:
                op_weights_per_rep[tag][point] = [0]*rep + [weight]

    return op_weights_per_rep


def calc_rep_op_weights(tags, filebase_run, temp):
    """Calculate the weights of a set of order parameters for a set of replicas

    Outputs a dictionary with order parameter tags as keys to dictionaries that
    have the order parameter values as keys and the arrays of weights for those
    op values across the simulations being considered as values.
    """
    op_weights_per_rep = {}
    for rep in range(reps):
        filename = '{}_rep-{}-{}.ops'.format(filebase_run, rep, temp)
        ops = read_ops_from_file(filebasename, tags)
        op_weights = calc_weights(ops)
        op_weights_per_rep = combine_sim_weights(op_weights, op_weights_per_rep, rep)

    return op_weights_per_rep


def calc_mean_ops(tags, ops):
    """Calculate specified mean order parameters"""
    mean_ops = {tag: ops[tag].mean() for tag in tags}

    return mean_ops


def calc_effN_ops(tags, ops):
    """Calculate effective sample sizes for given order parameters"""
    effN_ops = {}
    for tag in tags:
        t, g, Neff = timeseries.detectEquilibration(ops[tag])
        t /= len(ops[tag])
        effN_ops[tag] = (t, g, Neff)

    return effN_ops


def calc_mean_ops_from_weights(weights):
    """Average over the weights of an order parameters values

    Assumes that weights is a dictionary from a single op value to its weight.
    Also assumes that the op values are stored in a tuple.
    """
    mean_op = 0
    for op, weight in weights.items():
        mean_op += op[0] * weight

    return mean_op


def marginalize_ops(weights, header_tags, sel_tags):
    """Marginalize over all other order parameters in weights"""
    if type(sel_tags) == str:
        sel_tags = [sel_tags]

    marg_weights = {}
    indices = [header_tags.index(tag) for tag in sel_tags]
    for point, value in weights.items():
        comps = tuple(point[i] for i in indices)
        if comps in marg_weights:
            marg_weights[comps] += value
        else:
            marg_weights[comps] = value

    return marg_weights


def marginalize_ops_over_reps(weights, marginalized_weights, rep, index):
    """Marginalize over all other indices in weights"""
    for comp in marginalized_weights.keys():
        marginalized_weights[comp].append(0)

    for point, value in weights.items():
        comp = point[index]
        if comp in marginalized_weights:
            marginalized_weights[comp][rep] += value
        else:
            marginalized_weights[comp] = [0]*(rep) + [value]

    return marginalized_weights


def fill_weights(weights, reps, x_lims, y_lims):
    """WHAT DO?"""
    for x in range(x_lims[0], x_lims[1] + 1):
        for y in range(y_lims[0], y_lims[1] + 1):
            if (x, y) not in weights.keys():
                weights[(x, y)] = [0]*reps

    return weights


def order_weights(weights):
    """Return tuple of points and weights in order of increasing order param
    
    COULD THE OP ORDER FUNCTION BE USED?"""
    sorted_weights = sorted(weights.items(), key=itemgetter(0))
    points = []
    weights_only = []
    for point, rep_weights in sorted_weights:
        points.append(point)
        weights_only.append(rep_weights)

    return points, weights_only


def calc_mean_std(weights):
    """Calculate means and standard deviations of weights"""
    means = []
    stds = []
    for rep_weights in weights:
        means.append(np.mean(rep_weights))
        stds.append(np.std(rep_weights))

    return means, stds


def calc_pmf(weights):
    """Convert weights to potential of mean force

    Uses largest weight as reference.
    """
    max_weight = max(weights)
    pmfs = []
    for weight in weights:
        if weight != 0:
            pmfs.append(np.log(max_weight / weight))
        else:
            pmfs.append('nan')

    return pmfs


def calc_mean_std_pmf(weights):
    """Return the mean pmfs and standard deviations of given weights

    So what is exactly happening?
    """

    # Calculate pmfs at each point
    pmfs = [[] for i in weights]
    for rep in range(len(weights[0])):
        all_weights = []
        for rep_weights in weights:
            all_weights.append(rep_weights[rep])

        local_pmfs = calc_pmf(all_weights)
        for i, pmf in enumerate(local_pmfs):
            pmfs[i].append(pmf)

    means = []
    stds = []
    for i, rep_pmfs in enumerate(pmfs):

        # Not counting non-sampled states towards mean
        samples = [i for i in rep_pmfs if i != 'nan']
        if len(samples) != 0:
            means.append(np.mean(samples))
            stds.append(np.std(samples))
        else:
            means.append('nan')
            stds.append('nan')

    return means, stds
