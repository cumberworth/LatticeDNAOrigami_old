"""Misc. constants and functions for analysis of simulations."""

import math
import itertools
import scipy.constants

import numpy as np

# Units vectors for euclidean space
XHAT = np.array([1, 0, 0])
YHAT = np.array([0, 1, 0])
ZHAT = np.array([0, 0, 1])

# All possible unit vectors
dimensions = [XHAT, YHAT, ZHAT]
directions = [-1, 1]
VECTORS = [np.array(j * i) for i in dimensions for j in directions]

# Avogadro's number
AN = scipy.constants.N_A


def value_is_multiple(value, multiple):
    """Test if given value is a multiple of second value."""
    is_multiple = False
    try:
        if value % multiple == 0:
            is_multiple = True
        else:
            pass

    except ZeroDivisionError:
        is_multiple = False

    return is_multiple


def molarity_to_lattice_volume(molarity, lattice_site_volume):
    """Given a molarity, calculate the volume that cancels the fugacity.

    Volume is in units of number of lattice sites.
    """
    # Number of lattice sites per L (1 L * (1000) cm^3 / L * m^3 / (10^2)^3 cm^3)
    sites_per_litre = 1e-3 / lattice_site_volume

    # u = KB*T*ln(p), where p is the number of particles per lattice site
    # g = exp(1/(-KB*T)*u) = exp(ln(p)) = p
    # V * p = 1, V = 1 / p
    # So just convert molarity to number of particles per lattice site
    V = 1 / (molarity * AN / sites_per_litre)
    return V


def rotate_vector_half(vector, rotation_axis):
    vector = np.copy(vector)
    if all(np.abs(rotation_axis) == XHAT):
        vector[1] = -vector[1]
        vector[2] = -vector[2]

    if all(np.abs(rotation_axis) == YHAT):
        vector[0] = -vector[0]
        vector[2] = -vector[2]

    if all(np.abs(rotation_axis) == ZHAT):
        vector[0] = -vector[0]
        vector[1] = -vector[1]

    return vector


def rotate_vectors_quarter(vectors, rotation_axis, direction):
    """Rotate given vectors pi/2 about given axis in given direction."""
    rotated_vectors = np.copy(vectors)
    if all(rotation_axis == XHAT):
        y = vectors[:, 1]
        z = vectors[:, 2]
        rotated_vectors[:, 1] = direction * -z
        rotated_vectors[:, 2] = direction * y

    if all(rotation_axis == -XHAT):
        y = vectors[:, 1]
        z = vectors[:, 2]
        rotated_vectors[:, 1] = direction * z
        rotated_vectors[:, 2] = direction * -y

    elif all(rotation_axis == YHAT):
        x = vectors[:, 0]
        z = vectors[:, 2]
        rotated_vectors[:, 2] = direction * -x
        rotated_vectors[:, 0] = direction * z

    elif all(rotation_axis == -YHAT):
        x = vectors[:, 0]
        z = vectors[:, 2]
        rotated_vectors[:, 2] = direction * x
        rotated_vectors[:, 0] = direction * -z

    elif all(rotation_axis == ZHAT):
        x = vectors[:, 0]
        y = vectors[:, 1]
        rotated_vectors[:, 0] = direction * -y
        rotated_vectors[:, 1] = direction * x

    elif all(rotation_axis == -ZHAT):
        x = vectors[:, 0]
        y = vectors[:, 1]
        rotated_vectors[:, 0] = direction * y
        rotated_vectors[:, 1] = direction * -x

    return rotated_vectors
