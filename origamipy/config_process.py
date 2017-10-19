"""Functions for processing configuration files"""

import numpy as np


def calc_dist_matrices(traj_file, skip, skipo):
    """WRITE"""
    dist_matrices = []
    for step in range(0, traj_file.steps, skipo):
        config = np.array(traj_file.chains(step)[0]['positions'])
        dist_matrix = calc_dist_matrix(config)
        dist_matrices.append(dist_matrix)

    return dist_matrices


def calc_dist_matrix(config):
    """WRITE"""
    dists = []
    for i, pos_i in enumerate(config):
        for pos_j in config[i + 2:]:
            dist = calc_dist(pos_i, pos_j)
            dists.append(dist)

    return dists


def calc_dist(pos_i, pos_j):
    """WRITE"""
    diff = np.abs(pos_j - pos_i)

    return np.sum(diff)


def calc_rmsds(ref_dists, dist_matrices):
    """WRITE"""
    rmsds = []
    dist_matrices = np.array(dist_matrices)
    for dists in dist_matrices:
        rmsds.append(np.sqrt(((dists - ref_dists)**2).mean()))

    return rmsds


def calc_rmsd(config, ref):
    """WRITE"""
    return np.sqrt(((config - ref)**2).sum(axis=1).mean())


def align_centroids(config, ref):
    """WRITE"""
    diff_centroids = ref.mean(axis=0) - config.mean(axis=0)
    #diff_centroids = np.round(diff_centroids).astype(int)
    config = config + diff_centroids
    return config


def align_config(config, ref):
    """WRITE"""
    config = align_centroids(config, ref)
    rmsd = calc_rmsd(config, ref)
    while True:
        rotations = []
        for axis in [XHAT, YHAT, ZHAT]:
            axis_rmsd = rmsd
            rotation = ()
            for direction in [-1, 1]:
                test_config = rotate_vectors_quarter(config, axis, direction)
                test_rmsd = calc_rmsd(config, ref)
                if test_rmsd < axis_rmsd:
                    axis_rmsd = test_rmsd
                    rotation = (axis, direction)
                else:
                    pass

            if rotation != ():
                rotations.append(rotation)

        if rotations == []:
            break

        for rotation in rotations:
            config = rotate_vectors_quarter(config, *rotation)

        rmsd = calc_rmsd(config, ref)

    return config, rmsd


def align_configs(config_file, ref, skip, skipo):
    """WRITE"""
    aligned_configs = []
    rmsds = []
    steps = []
    for step in range(0, config_file.steps, skipo):
        #config = np.array(config_file.chains(step)[0]['positions'])[3:-4]
        config = np.array(config_file.chains(step)[0]['positions'])
        aligned_config, rmsd = align_config(config, ref)
        aligned_configs.append(aligned_config)
        rmsds.append(rmsd)
        steps.append(step * skip)

    return aligned_configs, rmsds, steps
