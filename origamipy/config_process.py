"""Functions for processing configuration files"""

import numpy as np

from origamipy import utility


def calc_dist_matrices(traj_file, skipo):
    """Calculate scaffold internal distance matrix for given trajectory
 
    Skip every skipo steps.
    """
    dist_matrices = []
    for i, step in enumerate(traj_file):
        if i % skipo != 0:
            continue

        config = np.array(step[0]['positions'])
        dist_matrix = calc_dist_matrix(config)
        dist_matrices.append(dist_matrix)

    return dist_matrices


def calc_dist_matrix(config):
    """Calculate scaffold internal distance matrix for given config.
    
    Does not include distances between domains adjacent along the scaffold as
    this will always be 1.
    """
    dists = []
    for i, pos_i in enumerate(config):
        for pos_j in config[i + 2:]:
            dist = calc_dist(pos_i, pos_j)
            dists.append(dist)

    return dists


def calc_dist(pos_i, pos_j):
    """Calculate distance between given two positions"""
    diff = np.abs(pos_j - pos_i)

    return np.sum(diff)


def calc_rmsds(ref_dists, dist_matrices):
    """Calculate RMSD between two distance matrices"""
    rmsds = []
    dist_matrices = np.array(dist_matrices)
    for dists in dist_matrices:
        rmsds.append(np.sqrt(((dists - ref_dists)**2).mean()))

    return rmsds


def calc_rmsd(config, ref):
    """Calculate RMSD between two configurations"""
    return np.sqrt(((config - ref)**2).sum(axis=1).mean())


def center_on_origin(config):
    """Align centroids"""
    scaffold_positions = config[0]['positions']
    centroid = np.array(scaffold_positions).mean(axis=0)
    for i, chain in enumerate(config):
        positions = chain['positions']
        positions = positions - centroid
        config[i]['positions'] = positions

    return config


def align_centroids(config, ref):
    """Align centroids"""
    diff_centroids = np.round(ref.mean(axis=0) - config.mean(axis=0))
    #diff_centroids = np.round(diff_centroids).astype(int)
    config = config + diff_centroids
    return config


def align_position(config, ref_config):
    scaffold_positions = config[0]['positions']
    scaffold_ref_positions = ref_config[0]['positions']
    rmsd = calc_rmsd(scaffold_positions, scaffold_ref_positions)
    test_config = np.copy(config)
    ref_axis = np.array([np.copy(utility.XHAT)])
    for i in range(4):
        ref_axis = utility.rotate_vectors_quarter(ref_axis, utility.YHAT, 1)
        for j, chain in enumerate(test_config):
            positions = chain['positions']
            positions = utility.rotate_vectors_quarter(positions, utility.YHAT, 1)
            test_config[j]['positions'] = positions

        config, rmsd = align_for_axis(ref_axis[0], test_config, config, rmsd, scaffold_ref_positions)

    ref_axis = utility.rotate_vectors_quarter(ref_axis, utility.ZHAT, 1)
    for j, chain in enumerate(test_config):
        positions = chain['positions']
        positions= utility.rotate_vectors_quarter(positions, utility.ZHAT, 1)
        test_config[j]['positions'] = positions

    config, rmsd = align_for_axis(ref_axis[0], test_config, config, rmsd, scaffold_ref_positions)

    ref_axis = utility.rotate_vectors_half(ref_axis, utility.ZHAT)
    for j, chain in enumerate(test_config):
        positions = chain['positions']
        positions = utility.rotate_vectors_half(positions, utility.ZHAT)
        test_config[j]['positions'] = positions

    config, rmsd = align_for_axis(ref_axis[0], test_config, config, rmsd, scaffold_ref_positions)

    return config, rmsd


def align_for_axis(ref_axis, test_config, config, rmsd, ref_positions):
    for i in range(4):
        for j, chain in enumerate(test_config):
            positions = chain['positions']
            positions = utility.rotate_vectors_quarter(positions, ref_axis, 1)
            test_config[j]['positions'] = positions

        scaffold_positions = test_config[0]['positions']
        test_rmsd = calc_rmsd(scaffold_positions, ref_positions)
        if test_rmsd < rmsd:
            config = np.copy(test_config)
            rmsd = test_rmsd

    return config, rmsd


def align_positions(config_file, ref_config):
    """Align configs in a file to a reference"""
    aligned_configs = []
    rmsds = []
    for step, config in enumerate(config_file):
        center_on_origin(config)
        aligned_config, rmsd = align_position(config, ref_config)
        aligned_configs.append(aligned_config)
        rmsds.append(rmsd)

    return aligned_configs, np.array(rmsds)


def calc_end_to_end_dists(configs_file):
    dists = []
    for step, config in enumerate(configs_file):
        positions = np.array(config[0]['positions'])
        dists.append(np.linalg.norm(positions[-1] - positions[0]))

    return dists

def calc_radius_of_gyration(configs_file):
    rgs = []
    for config in configs_file:
        center_on_origin(config)
        positions = np.array(config[0]['positions'])
        rgs.append(np.linalg.norm(positions))

    return rgs


def calc_stacked_pairs(configs_file, states, cyclic=False, extra_pairs=[]):
    """Calc stacked pairs for all chain."""
    stacked_pairs = []
    for i, config in enumerate(configs_file):
        step_stacked_pairs = []
        step_states = states[i]
        poses = np.array(config[0]['positions'])
        ores = np.array(config[0]['orientations'])
        for j in range(len(poses) - 1):
            stacked = check_stacked(step_states, poses, ores, j, j + 1)
            step_stacked_pairs.append(stacked)

        if cyclic:
            stacked = check_stacked(step_states, poses, ores, 0, -1)
            step_stacked_pairs.append(stacked)

        for pair in extra_pairs:
            stacked = check_stacked(step_states, poses, ores, pair[0], pair[1])
            step_stacked_pairs.append(stacked)

        stacked_pairs.append(step_stacked_pairs)

    return stacked_pairs


def check_stacked(states, poses, ores, i, j):
    state_1 = states[i]
    state_2 = states[j]
    if not (state_1 == 2 == state_2):
        return 0

    pos_1 = poses[i]
    pos_2 = poses[j]
    ndr = pos_2 - pos_1
    ore_1 = ores[i]
    ore_2 = ores[j]
    if (ore_1 == -ore_2).all() and (ore_1 != ndr).any() and (-ore_1 != ndr).any():
        return 1
    else:
        return 0
