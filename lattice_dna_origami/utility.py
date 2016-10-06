#!/usr/bin/env python

"""Misc constants, functions, and classes for lattice origami model."""

import math
import itertools
import scipy.constants

import numpy as np

# Scaffold domain index
SCAFFOLD_INDEX = 0

# Occupancy states
UNASSIGNED = 0
UNBOUND = 1
BOUND = 2
MISBOUND = 2

# Move outcomes
REJECTED = 0
ACCEPTED = 1

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


# Exceptions
class MoveRejection(Exception):
    """Used for early move rejection."""
    pass

class ConstraintViolation(Exception):
    """Used for constraint violations in OrigamiSystem."""
    pass


class OrigamiMisuse(Exception):
    """Used for miscellaneous misuse of the origami objects."""
    pass


# Usefull general functions
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


# Vector manipulation
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


def rotate_vector_quarter(vector, rotation_axis, direction):
    """Rotate given vector pi/2 about given axis in given direction."""
    vector = np.copy(vector)
    if all(rotation_axis == XHAT):
        y = vector[1]
        z = vector[2]
        vector[1] = direction * -z
        vector[2] = direction * y

    if all(rotation_axis == -XHAT):
        y = vector[1]
        z = vector[2]
        vector[1] = direction * z
        vector[2] = direction * -y

    elif all(rotation_axis == YHAT):
        x = vector[0]
        z = vector[2]
        vector[2] = direction * -x
        vector[0] = direction * z

    elif all(rotation_axis == -YHAT):
        x = vector[0]
        z = vector[2]
        vector[2] = direction * x
        vector[0] = direction * -z

    elif all(rotation_axis == ZHAT):
        x = vector[0]
        y = vector[1]
        vector[0] = direction * -y
        vector[1] = direction * x

    elif all(rotation_axis == -ZHAT):
        x = vector[0]
        y = vector[1]
        vector[0] = direction * y
        vector[1] = direction * -x

    return vector


class IdealRandomWalks:

    def __init__(self):
        # Number of random walks indexed by tuple of end - start
        self._num_walks = {}

    def num_walks(self, start, end, N):
        """Calculate number of walks between two positions."""

        # Only work with one permutation of abs(DR)
        DR = tuple(end - start)
        walk = (DR, N)
        try:
            num_walks = self._num_walks[walk]
        except KeyError:
            pass
        else:
            return num_walks

        # Calculate num walks (see dijkstra1994)
        DX = DR[0]
        DY = DR[1]
        DZ = DR[2]
        Nminus = (N - DX - DY - DZ) // 2
        Nplus = (N - DX - DY + DZ) // 2
        num_walks = 0

        # Need some negative steps to reach a negative location
        for ybar in range(0, Nminus + 1):
            for xbar in range(0, Nminus + 1 - ybar):
                f1 = math.factorial(N)
                f2 = math.factorial(xbar)
                if (xbar + DX) < 0:
                    continue

                f3 = math.factorial(xbar + DX)
                f4 = math.factorial(ybar)
                if (ybar + DY) < 0:
                    continue

                f5 = math.factorial(ybar + DY)
                f6 = math.factorial(Nminus - xbar - ybar)
                if (Nplus - xbar - ybar) < 0:
                    continue

                f7 = math.factorial(Nplus - xbar - ybar)
                num_walks += f1 // (f2 * f3 * f4 * f5 * f6 * f7)

        # Add entries
        for perm in itertools.permutations((DX, DY, DZ)):
            self._num_walks[(perm, N)] = num_walks

        for perm in itertools.permutations((-DX, DY, DZ)):
            self._num_walks[(perm, N)] = num_walks

        for perm in itertools.permutations((DX, -DY, DZ)):
            self._num_walks[(perm, N)] = num_walks

        for perm in itertools.permutations((DX, DY, -DZ)):
            self._num_walks[(perm, N)] = num_walks

        for perm in itertools.permutations((-DX, -DY, DZ)):
            self._num_walks[(perm, N)] = num_walks

        for perm in itertools.permutations((DX, -DY, -DZ)):
            self._num_walks[(perm, N)] = num_walks

        for perm in itertools.permutations((-DX, DY, -DZ)):
            self._num_walks[(perm, N)] = num_walks

        for perm in itertools.permutations((-DX, -DY, -DZ)):
            self._num_walks[(perm, N)] = num_walks

        return num_walks
