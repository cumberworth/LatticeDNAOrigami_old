"""Functions for processing configuration files"""

import numpy as np

from origamipy.utility import XHAT, YHAT, ZHAT
from origamipy.utility import rotate_vectors_quarter


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


def align_config(config, ref):
    rmsd = calc_rmsd(config, ref)
    test_config = np.copy(config)
    for x_turns in range(4):
        test_config = rotate_vectors_quarter(test_config, XHAT, 1)
        for y_turns in range(4):
            test_config = rotate_vectors_quarter(test_config, YHAT, 1)
            for z_turns in range(4):
                test_config = rotate_vectors_quarter(test_config, ZHAT, 1)
                test_rmsd = calc_rmsd(test_config, ref)
                if test_rmsd < rmsd:
                    config = np.copy(test_config)
                    rmsd = test_rmsd
                else:
                    pass


    return config, rmsd


def align_configs(config_file, ref, skip, skipo):
    """Align configs in a file to a reference
    
    Skip is used to output the correct steps, while skipo is used to set the
    frequency with which to calculate the RMSD for the configs in the file.
    """
    aligned_configs = []
    rmsds = []
    steps = []
    for step in range(0, config_file.steps, skipo):
        #config = np.array(config_file.chains(step)[0]['positions'])[3:-4]
        config = np.array(config_file.chains(step)[0]['positions'])
        config = center_on_origin(config)
        aligned_config, rmsd = align_config(config, ref)
        aligned_configs.append(aligned_config)
        rmsds.append(rmsd)
        steps.append(step * skip)

    return aligned_configs, np.array(rmsds), steps
