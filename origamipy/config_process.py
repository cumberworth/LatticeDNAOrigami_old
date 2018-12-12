"""Functions for processing configuration files"""

import numpy as np

from origamipy import utility


def calc_dist_matrices(traj_file, skipo):
    """Calculate scaffold internal distance matrix for given trajectory
 
    Skip every skipo steps.
    """
    dist_matrices = []
    for step in range(0, traj_file.steps, skipo):
        config = np.array(traj_file.chains(step)[0]['positions'])
        dist_matrix = calc_dist_matrix(config)
        dist_matrices.append(dist_matrix)

    return dist_matrices


def calc_dist_matrix(config):
    """Calculate scaffold internal distance matrix for given config"""
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
    centroid = config.mean(axis=0)
    config = config - centroid
    return config


def align_centroids(config, ref):
    """Align centroids"""
    diff_centroids = np.round(ref.mean(axis=0) - config.mean(axis=0))
    #diff_centroids = np.round(diff_centroids).astype(int)
    config = config + diff_centroids
    return config


def align_position(config, ref_positions):
    rmsd = calc_rmsd(config, ref_positions)
    test_config = np.copy(config)
    ref_axis = np.array([np.copy(utility.XHAT)])
    for i in range(4):
        ref_axis = utility.rotate_vectors_quarter(ref_axis, utility.YHAT, 1)
        test_config = utility.rotate_vectors_quarter(test_config, utility.YHAT, 1)
        config, rmsd = align_for_axis(ref_axis[0], test_config, config, rmsd, ref_positions)

    ref_axis = utility.rotate_vectors_quarter(ref_axis, utility.ZHAT, 1)
    test_config = utility.rotate_vectors_quarter(test_config, utility.ZHAT, 1)
    config, rmsd = align_for_axis(ref_axis[0], test_config, config, rmsd, ref_positions)

    ref_axis = utility.rotate_vectors_half(ref_axis, utility.ZHAT)
    test_config = utility.rotate_vectors_half(test_config, utility.ZHAT)
    config, rmsd = align_for_axis(ref_axis[0], test_config, config, rmsd, ref_positions)

    return config, rmsd


def align_for_axis(ref_axis, test_config, config, rmsd, ref_positions):
    for i in range(4):
        test_config = utility.rotate_vectors_quarter(test_config, ref_axis, 1)
        test_rmsd = calc_rmsd(test_config, ref_positions)
        if test_rmsd < rmsd:
            config = np.copy(test_config)
            rmsd = test_rmsd

    return config, rmsd


def align_positions(config_file, ref_positions):
    """Align configs in a file to a reference"""
    aligned_positions = []
    rmsds = []
    for step, config in enumerate(config_file):
        positions = np.array(config[0]['positions'])
        positions = center_on_origin(positions)
        aligned_position, rmsd = align_position(positions, ref_positions)
        aligned_positions.append(aligned_position)
        rmsds.append(rmsd)

    return aligned_positions, np.array(rmsds)


def calc_end_to_end_dists(configs_file):
    dists = []
    for step, config in enumerate(configs_file):
        positions = np.array(config[0]['positions'])
        dists.append(np.linalg.norm(positions[-1] - positions[0]))

    return dists

def calc_radius_of_gyration(configs_file):
    rgs = []
    for step, config in enumerate(configs_file):
        positions = np.array(config[0]['positions'])
        center_on_origin(positions)
        rgs.append(np.linalg.norm(positions))

    return rgs
