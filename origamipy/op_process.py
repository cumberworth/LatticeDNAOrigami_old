"""Functions for processing order parameter files"""

import numpy as np


def read_ops_from_file(filename, tags):
    """Read specified order parameters from file

    Returns a dictionary of tags to values.
    """
    with open(filename) as inp:
        header = inp.readline().split(', ')

    all_ops = np.loadtxt(filename, skiprows=1)
    ops = {}
    for i, tag in enumerate(header):
        if tag in tags:

            # First index is step, not in header
            ops[tag] = all_ops[:, i + 1]

    return ops


def read_weights_from_file(filename):
    """Read order parameter weights from enumeration output file

    The order parameters are stored in the first column in parantheses.
    """
    with open(filename) as inp:
        tags = inp.readline().split()
        lines = inp.readlines()

    points_weights = {}
    for line in lines:
        if line == '\n':
            continue

        start_p = line.find('(')
        end_p = line.find(')')
        point = tuple(int(i) for i in line[start_p + 1:end_p].split())
        weight = float(line.split()[-1])
        points_weights[point] = weight

    return tags, points_weights



def read_counts_file(filename):
    """Read counts file and return array
    
    Returns the number of fully bound domain pairs and the number of bound
    staples.
    """
    all_counts = np.loadtxt(filename)
    domain_counts = all_counts[:, 3]
    staple_counts = all_counts[:, 2]
    ops = np.array([domain_counts, staple_counts]).transpose()

    return ops


def find_closest_ops(point, ops):
    """Find closest ops in given list that is closest to specified value
    
    Returns a list of all ops that are have the minimum distance found."""
    dist = 0
    points = []
    while len(points) == 0:
        dist += 1
        for op in ops:
            calc_dist = 0
            for i in range(len(op)):
                calc_dist += abs(point[i] - op[i])

            if calc_dist <= dist:
                points.append(op)

    return points


def get_all_points(win, point, points, comp):
    """THIS DOES WHAT?"""
    if len(point) == len(win[0]):
        points.append(tuple(point))
    else:
        for i in range(win[0][comp], win[1][comp] + 1):
            point.append(i)
            comp += 1
            points = get_all_points(win, point, points, comp)
            point.pop()
            comp -= 1

    return points


def sort_by_ops(ops):
    """Return ???"""
    op_to_config = {}
    for i, op in enumerate(ops):
        op_key = tuple(op.tolist())
        if op_key in op_to_config:
            op_to_config[op_key].append(i)
        else:
            op_to_config[op_key] = [i]

    return op_to_config
