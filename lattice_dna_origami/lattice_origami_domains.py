#!/usr/env python

"""Run MC simulations of the lattice domain-resolution DNA origami model."""

import json
import math
import sys
import os
import random
import pdb
import copy
import itertools
from enum import Enum

import h5py
import numpy as np
import scipy.constants

from lattice_dna_origami.nearest_neighbour import *

# Avogadro's number
AN = scipy.constants.N_A

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

class MOVETYPE(Enum):
    IDENTITY = 0
    ROTATE_ORIENTATION_VECTOR = 1
    EXCHANGE_STAPLE = 2
    REGROW_STAPLE = 3
    REGROW_SCAFFOLD = 4
    CB_EXCHANGE_STAPLE = 5
    CB_REGROW_STAPLE = 6
    CB_REGROW_SCAFFOLD = 7
    CB_CONSERVED_TOPOLOGY = 8
    MIS_EXCHANGE_STAPLE = 9
    MIS_REGROW_STAPLE = 10
    MIS_REGROW_SCAFFOLD = 11
    MIS_CB_EXCHANGE_STAPLE = 12
    MIS_CB_REGROW_STAPLE = 13
    MIS_CB_REGROW_SCAFFOLD = 14
    MIS_CB_CONSERVED_TOPOLOGY = 15


class MoveRejection(Exception):
    """Used for early move rejection."""
    pass

class ConstraintViolation(Exception):
    """Used for constraint violations in OrigamiSystem."""
    pass


class OrigamiMisuse(Exception):
    """Used for miscellaneous misuse of the origami objects."""
    pass


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

        # Check stored values
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


IDEAL_RANDOM_WALKS = IdealRandomWalks()


class OrigamiSystem:
    """Simple cubic lattice model of DNA origami at domain level resolution.

    See reports/modelSpecs/domainResModelSpecs for exposition on the model.

    I've used get methods instead of properties as properties don't take
    indices. I would have to index the internal structure directly, which
    I don't want to do.
    """

    def __init__(self, input_file, step, temp, cation_M, misbinding=True):
        self.temp = temp

        # Fully bound only
        self._bind_misbound_domain = self._no_misbound

        # Scaffold cyclic
        self.cyclic = input_file.cyclic

        # Domain identities of each chain
        self.identities = input_file.identities

        # Scaffold strand indices of complementary domains to staple domains,
        # indexed by staple identity
        self.staple_to_scaffold_domains = {}
        for staple_identity, staple in enumerate(self.identities[1:]):
            staple_identity += 1
            scaffold_indices = []
            for staple_domain_ident in staple:
                scaffold_domain_i = self.identities[SCAFFOLD_INDEX].index(
                        -staple_domain_ident)
                scaffold_indices.append(scaffold_domain_i)

            self.staple_to_scaffold_domains[staple_identity] = scaffold_indices

        # Domain sequences
        self.sequences = input_file.sequences

        # Calculate and store hybridization energies
        self._hybridization_energies = []
        for chain_i in self.sequences:
            chain_j_energies = []
            for chain_j in self.sequences:
                seq_i_energies = []
                for seq_i in chain_i:
                    seq_j_energies = []
                    for seq_j in chain_j:
                        comp_seqs = find_longest_contig_complement(seq_i, seq_j)
                        comp_seq_energies = []
                        for comp_seq in comp_seqs:
                           energy = calc_hybridization_energy(comp_seq, temp, cation_M)
                           comp_seq_energies.append(energy)
                        energy = np.array(comp_seq_energies).mean()
                        seq_j_energies.append(energy)
                    seq_i_energies.append(seq_j_energies)
                chain_j_energies.append(seq_i_energies)
            self._hybridization_energies.append(chain_j_energies)

        # Unique indices and mapping dictionaries
        self._unique_to_working = {}
        self._working_to_unique = []

        # Indices to identities list for current chains
        self._chain_identities = []

        # Working indices indexed by identity
        self._identity_to_index = [[] for i in range(len(self.identities))]

        # Configuration arrays
        self._positions = []
        self._orientations = []

        # Next and previous domain vectors
        self._next_domains = []
        self._prev_next_domains = []
        #self._prev_domains = []

        # Dictionary with position keys and state values
        self._position_occupancies = {}

        # Dictionary with domain keys and state values
        self._domain_occupancies = {}

        # Dictionary with bound domain keys and values
        self._bound_domains = {}

        # Number of fully bound domains
        self._fully_bound_domains = 0

        # Dictionary with position keys and unbound domain values
        self._unbound_domains = {}

        # Set configuration to specified input file step
        self.chain_lengths = []
        for chain_index, chain in enumerate(input_file.chains(step)):
            unique_index = chain['index']
            self._unique_to_working[unique_index] = chain_index
            self._working_to_unique.append(unique_index)

            identity = chain['identity']
            self._identity_to_index[identity].append(unique_index)
            self._chain_identities.append(identity)
            num_domains = len(self.identities[identity])
            self._positions.append([[]] * num_domains)
            self._orientations.append([[]] * num_domains)
            self._next_domains.append([[]] * num_domains)
            #self._prev_domains.append([[]] * num_domains)
            self.chain_lengths.append(num_domains)
            previous_position = np.array(chain['positions'][0])
            for domain_index in range(num_domains):
                position = np.array(chain['positions'][domain_index])
                orientation = np.array(chain['orientations'][domain_index])
                self.set_domain_configuration(chain_index, domain_index,
                        position, orientation)

        # Keep track of unique chain index
        self._current_chain_index = max(self._working_to_unique)

        # Close input file to prevent corruption
        input_file.close()

    @property
    def chains(self):
        """Standard format for passing chain configuration."""
        chains = []
        for working_index, unique_index in enumerate(self._working_to_unique):
            chain = {}
            chain['index'] = unique_index
            chain['identity'] = self._chain_identities[working_index]
            positions = []
            for position in self._positions[working_index]:
                position = position.tolist()
                positions.append(position)

            chain['positions'] = positions
            orientations = []
            for orientation in self._orientations[working_index]:
                orientation = orientation.tolist()
                orientations.append(orientation)

            chain['orientations'] = orientations
            chains.append(chain)
        return chains

    @property
    def num_staples(self):
        return len(self.chain_lengths) - 1

    @property
    def num_bound_domains(self):
        return self._fully_bound_domains

    @property
    def energy(self):
        """System energy in 1/K."""
        energy = 0

        # Iterate over all scaffold domains and sum hybridization energies
        for domain_i in range(self.chain_lengths[SCAFFOLD_INDEX]):
            if self.get_domain_occupancy(SCAFFOLD_INDEX, domain_i) == BOUND:
                bound_domain = self.get_bound_domain(SCAFFOLD_INDEX, domain_i)
                energy += self.get_hybridization_energy(SCAFFOLD_INDEX, domain_i,
                        *bound_domain)

        return energy

    def get_num_staples(self, identity):
        """Return number of staples in system of given identity."""
        Ni = len(self._identity_to_index[identity])
        return Ni

    def get_complementary_domains(self, staple_i):
        """Return list of scaffold domains complementary to given staple."""
        staple_identity = self._chain_identities[staple_i]
        return self.staple_to_scaffold_domains[staple_identity]

    def get_domain_position(self, chain_index, domain_index):
        """Return domain position as numpy array."""
        return self._positions[chain_index][domain_index]

    def get_domain_orientation(self, chain_index, domain_index):
        """Return domain orientation as numpy array."""
        return self._orientations[chain_index][domain_index]

    def get_position_occupancy(self, position):
        """Return occupancy of given position."""
        try:
            position = tuple(position)
            occupancy = self._position_occupancies[position]
        except KeyError:
            occupancy = UNASSIGNED
        return occupancy

    def get_domain_occupancy(self, chain_index, domain_index):
        """Return occupancy of given domain."""

        unique_index = self._working_to_unique[chain_index]
        try:
            occupancy = self._domain_occupancies[(unique_index, domain_index)]
        except KeyError:
            if chain_index < len(self._working_to_unique):
                occupancy = UNASSIGNED
            else:
                raise

        return occupancy

    def get_bound_domain(self, chain_index, domain_index):
        """Return domain bound to given domain, otherwise return empty tuple.

        Consider failing instead to be consistent with get_unbound_domain.
        """
        unique_index = self._working_to_unique[chain_index]
        try:
            unique_index, domain_index = self._bound_domains[(unique_index,
                domain_index)]
            chain_index = self._unique_to_working[unique_index]
            domain = (chain_index, domain_index)
        except KeyError:
            domain = ()

        return domain

    def get_unbound_domain(self, position):
        """Return domain at position with unbound state."""
        position = tuple(position)
        unique_index, domain_index = self._unbound_domains[position]
        chain_index = self._unique_to_working[unique_index]
        return chain_index, domain_index

    def get_random_staple_identity(self):
        """Return random staple identity."""
        staple_identity = random.randrange(1, len(self.identities))
        domain_identities = self.identities[staple_identity]
        return staple_identity, domain_identities

    def get_random_staple_of_identity(self, identity):
        """Return random staple of given identity."""
        staples = self._identity_to_index[identity]
        if staples == []:
            raise IndexError
        else:
            staple_index = random.choice(staples)

        staple_index = self._unique_to_working[staple_index]
        return staple_index

    def get_hybridization_energy(self, c_1, d_1, c_2, d_2):
        """Return precalculated hybridization energy."""
        c_1_ident = self._chain_identities[c_1]
        c_2_ident = self._chain_identities[c_2]
        return self._hybridization_energies[c_1_ident][c_2_ident][d_1][d_2]

    def check_all_constraints(self):
        """Check all constraints by rebuilding system."""
        chains = self.chains

        # Unassign everything
        for chain_i, chain in enumerate(chains):
            for domain_i in range(len(chain['positions'])):
                self.unassign_domain(chain_i, domain_i)

        # Reset configuration
        for chain_i, chain in enumerate(chains):
            for domain_i in range(len(chain['positions'])):
                pos = np.array(chain['positions'][domain_i])
                ore = np.array(chain['orientations'][domain_i])
                try:
                    self.set_domain_configuration(chain_i, domain_i, pos, ore)
                except ConstraintViolation:
                    raise OrigamiMisuse

        # Check distance constraints
        for chain_ndrs in self._next_domains:
            for ndr in chain_ndrs[:-1]:
                if (ndr**2).sum() != 1:
                    raise ConstraintViolation

    def check_domain_configuration(self, chain_index, domain_index, position,
                orientation, step):
        """Check if constraints are obeyed and return energy change."""
        domain = (chain_index, domain_index)
        delta_e = 0

        # Constraint violation if position in bound state
        occupancy = self.get_position_occupancy(position)
        if occupancy in [BOUND, MISBOUND]:

            raise ConstraintViolation
        else:
            pass

        self._positions[chain_index][domain_index] = position
        self._orientations[chain_index][domain_index] = orientation

        # Update next domain vectors
        try:
            self._update_next_domain(*domain)
        except ConstraintViolation:

            # Remove attempted configuration
            self._positions[chain_index][domain_index] = []
            self._orientations[chain_index][domain_index] = []
            self._revert_next_domain()
            raise

        # Attempt binding if position occupied in unbound state
        if occupancy == UNBOUND:
            try:
                delta_e = self._bind_domain(*domain)
            except ConstraintViolation:

                # Remove attempted configuration
                self._positions[chain_index][domain_index] = []
                self._orientations[chain_index][domain_index] = []
                self._revert_next_domain()
                raise
        else:
            pass

        # Remove checked configuration
        self._positions[chain_index][domain_index] = []
        self._orientations[chain_index][domain_index] = []
        self._revert_next_domain()

        return delta_e

    def set_checked_domain_configuration(self, chain_index, domain_index,
                position, orientation):
        """Set domain to previously checked configuration."""
        domain = (chain_index, domain_index)
        self._positions[chain_index][domain_index] = position
        self._orientations[chain_index][domain_index] = orientation
        self._update_next_domain(*domain)
        occupancy = self.get_position_occupancy(position)
        unique_index = self._working_to_unique[chain_index]
        domain_key = (unique_index, domain_index)
        if occupancy == UNBOUND:
            self._update_occupancies_bound(position, *domain)
        else:
            self._update_occupancies_unbound(position, *domain)

    def set_domain_configuration(self, chain_index, domain_index, position,
                orientation):
        """Set domain configuration and return change in energy.

        Assumes that given domain has already been unassigned. Consider adding a
        check.

        Will raise exception ConstraintViolation if constraints violated.
        """
        domain = (chain_index, domain_index)
        delta_e = 0

        # Constraint violation if position in bound state
        occupancy = self.get_position_occupancy(position)
        if occupancy in [BOUND, MISBOUND]:

            raise ConstraintViolation
        else:
            pass

        self._positions[chain_index][domain_index] = position
        self._orientations[chain_index][domain_index] = orientation

        # Update next domain vectors
        try:
            self._update_next_domain(*domain)
        except ConstraintViolation:

            # Remove attempted configuration
            self._positions[chain_index][domain_index] = []
            self._orientations[chain_index][domain_index] = []
            self._revert_next_domain()
            raise

        # Attempt binding if position occupied in unbound state
        if occupancy == UNBOUND:
            try:
                delta_e = self._bind_domain(*domain)
            except ConstraintViolation:

                # Remove attempted configuration
                self._positions[chain_index][domain_index] = []
                self._orientations[chain_index][domain_index] = []
                self._revert_next_domain()
                raise
            else:
                self._update_occupancies_bound(position, *domain)

        # Move to empty site and update occupancies
        else:
            self._update_occupancies_unbound(position, *domain)

        return delta_e

    def set_domain_orientation(self, chain_index, domain_index, orientation):
        """Set domain orientation.

        There is a orientation setter and not an indepenent public position
        setter because a new orientation is always generated when a new position
        is generated (easier than somehow trying to keep the same relative
        orientation), while the orientation is set indepentnly, so not all the
        checks of set_domain_configuration are necessary anymore.
        """
        if self.get_domain_occupancy(chain_index, domain_index) == BOUND:
            raise ConstraintViolation
        else:
            self._orientations[chain_index][domain_index] = orientation

    def unassign_domain(self, chain_index, domain_index):
        """Deletes positions, orientations, and removes/unassigns occupancies.

        Intended for use when regrowing parts of the system to allow correct
        testing of constraints (actually MUST be used).

        Consider adding a global check that all defined domains are not in
        unassigned states.
        """
        unique_index = self._working_to_unique[chain_index]
        domain = (chain_index, domain_index)
        domain_key = (unique_index, domain_index)
        occupancy = self._domain_occupancies[domain_key]

        delta_e = 0
        if occupancy == BOUND:
            self._fully_bound_domains -= 1

        if occupancy in [BOUND, MISBOUND]:

            # Collect energy
            bound_domain = self.get_bound_domain(*domain)
            bound_c_ui = self._working_to_unique[bound_domain[0]]
            bound_domain_key = (bound_c_ui, bound_domain[1])
            delta_e = -self.get_hybridization_energy(*domain, *bound_domain)
            position = tuple(self._positions[chain_index][domain_index])
            self._positions[chain_index][domain_index] = []
            self._orientations[chain_index][domain_index] = []
            del self._bound_domains[domain_key]
            del self._bound_domains[bound_domain_key]
            del self._domain_occupancies[domain_key]
            self._unbound_domains[position] = bound_domain_key
            self._position_occupancies[position] = UNBOUND
            self._domain_occupancies[bound_domain_key] = UNBOUND
        elif occupancy == UNBOUND:
            position = tuple(self._positions[chain_index][domain_index])
            self._positions[chain_index][domain_index] = []
            self._orientations[chain_index][domain_index] = []
            del self._unbound_domains[position]
            del self._position_occupancies[position]
            del self._domain_occupancies[domain_key]
        else:
            pass

        # Delete next/prev domain vectors
        self._next_domains[chain_index][domain_index] = []
        #self._prev_domains[chain_index][domain_index] = []

        prev_domain_i = domain_index - 1
        if prev_domain_i >= 0:
            ndr = self._next_domains[chain_index][prev_domain_i] = []
        else:
            if self.cyclic and chain_index == SCAFFOLD_INDEX:
                prev_domain_i = self.wrap_cyclic_scaffold(domain_index - 1)
                self._next_domains[chain_index][prev_domain_i] = []
            else:
                pass

#        next_domain_i = domain_index + 1
#        if next_domain_i < self.chain_lengths[chain_index]:
#            ndr = self._prev_domains[chain_index][next_domain_i] = []
#        else:
#            if self.cyclic and chain_index == SCAFFOLD_INDEX:
#                next_domain_i = self.wrap_cyclic_scaffold(domain_index + 1)
#                self._prev_domains[chain_index][next_domain_i] = []
#            else:
#                pass

        return delta_e

    def add_chain(self, identity):
        """Add chain with domains in unassigned state and return chain index."""
        self._current_chain_index += 1
        chain_index = len(self.chain_lengths)
        self._identity_to_index[identity].append(self._current_chain_index)
        self._working_to_unique.append(self._current_chain_index)
        self._unique_to_working[self._current_chain_index] = chain_index
        self._chain_identities.append(identity)
        chain_length = len(self.identities[identity])
        self._positions.append([[]] * chain_length)
        self._orientations.append([[]] * chain_length)
        self._next_domains.append([[]] * chain_length)
#        self._prev_domains.append([[]] * chain_length)
        self.chain_lengths.append(chain_length)
        return chain_index

    def readd_chain(self, identity, unique_index):
        """Add chain but with give unique index."""
        chain_index = len(self.chain_lengths)
        self._identity_to_index[identity].append(unique_index)
        self._working_to_unique.append(unique_index)
        self._unique_to_working[unique_index] = chain_index
        self._chain_identities.append(identity)
        chain_length = len(self.identities[identity])
        self._positions.append([[]] * chain_length)
        self._orientations.append([[]] * chain_length)
        self._next_domains.append([[]] * chain_length)
        self.chain_lengths.append(chain_length)
        return chain_index

    def delete_chain(self, chain_index):
        """Delete chain."""

        unique_index = self._working_to_unique[chain_index]
        identity = self._chain_identities[chain_index]
        self._identity_to_index[identity].remove(unique_index)
        del self._working_to_unique[chain_index]
        del self._unique_to_working[unique_index]

        # Update map from unique to working indices
        for unique_index, working_index in self._unique_to_working.items():
            if working_index > chain_index:
                self._unique_to_working[unique_index] = working_index - 1
            else:
                pass

        del self._chain_identities[chain_index]
        del self._positions[chain_index]
        del self._orientations[chain_index]
        del self._next_domains[chain_index]
#        del self._prev_domains[chain_index]
        del self.chain_lengths[chain_index]

        return

    def center(self):
        """Translates system such that first domain of scaffold on origin."""

        # New occupancy dicts
        position_occupancies = {}
        unbound_domains = {}

        # Translation vector
        r_t = self._positions[0][0]
        for chain_index, chain_positions in enumerate(self._positions):
            unique_index = self._working_to_unique[chain_index]
            for domain_index, chain_positions in enumerate(chain_positions):
                r_o = self._positions[chain_index][domain_index]
                r_n = r_o - r_t
                self._positions[chain_index][domain_index] = r_n

                # Update occupancies
                domain = (unique_index, domain_index)
                r_o = tuple(r_o)
                r_n = tuple(r_n)
                occupancy = self._domain_occupancies[domain]

                if occupancy == BOUND:
                    position_occupancies[r_n] = BOUND
                elif occupancy == MISBOUND:
                    position_occupancies[r_n] = MISBOUND
                elif occupancy == UNBOUND:
                    position_occupancies[r_n] = UNBOUND
                    unbound_domains[r_n] = domain

        self._position_occupancies = position_occupancies
        self._unbound_domains = unbound_domains

    def wrap_cyclic_scaffold(self, domain_i):
        """Wrap given domain to correct domain index."""
        scaffold_length = self.chain_lengths[0]
        if domain_i >= scaffold_length:
            wrapped_domain_i = domain_i - scaffold_length
        elif domain_i < 0:
            wrapped_domain_i = scaffold_length + domain_i

        else:
            wrapped_domain_i = domain_i

        return wrapped_domain_i

    def _update_occupancies_bound(self, position, tc_i, td_i):
        
        # Consider having checking these identities in it's own method
        position = tuple(position)
        oc_ui, od_i = self._unbound_domains[position]
        oc_i = self._unique_to_working[oc_ui]
        tc_ident = self._chain_identities[tc_i]
        td_ident = self.identities[tc_ident][td_i]
        oc_ident = self._chain_identities[oc_i]
        od_ident = self.identities[oc_ident][od_i]
        if td_ident == -od_ident:
            state = BOUND
            self._fully_bound_domains += 1
        else:
            state = MISBOUND

        unique_index = self._working_to_unique[tc_i]
        domain_key = (unique_index, td_i)
        od_key = (oc_ui, od_i)
        del self._unbound_domains[position]
        self._domain_occupancies[domain_key] = state
        self._domain_occupancies[od_key] = state
        self._position_occupancies[position] = state
        self._bound_domains[od_key] = domain_key
        self._bound_domains[domain_key] = od_key

    def _update_occupancies_unbound(self, position, tc_i, td_i):
        unique_index = self._working_to_unique[tc_i]
        domain_key = (unique_index, td_i)
        position = tuple(position)
        self._domain_occupancies[domain_key] = UNBOUND
        self._position_occupancies[position] = UNBOUND
        self._unbound_domains[position] = domain_key

    def _bind_domain(self, trial_chain_index, trial_domain_index):
        """Bind given domain in preset trial config and return change in energy.
        """
        position = tuple(self._positions[trial_chain_index][trial_domain_index])

        # Test if complementary (and has correct orientation for binding)
        try:
            occupying_domain = self.get_unbound_domain(position)
        except KeyError:

            # This would only happen if the caller didn't check the state of
            # position first.
            raise

        # Convenience variable
        trial_domain = (trial_chain_index, trial_domain_index)

        complementary = self._domains_match(*trial_domain, *occupying_domain)
        if not complementary:
            delta_e = self._bind_misbound_domain(trial_domain, occupying_domain)
            return delta_e
        else:
            pass

        # Check constraints between domains and neighbours
        domains = [trial_domain, occupying_domain]
        for domains_i, domain in enumerate(domains):
            chain_i, domain_i = domain

            # Occupancies and domain indices of the 6 relevant neighbouring domains
            occupancies = []
            domain_is = []
            for i in [-3, -2, -1, 1, 2, 3]:
                domain_j = domain_i + i
                if self.cyclic and chain_i == SCAFFOLD_INDEX:
                    domain_j = self.wrap_cyclic_scaffold(domain_j)

                domain_is.append(domain_j)

                # Get occupancy will return incorrect things if negative index given
                if domain_j >= 0:
                    occupancy = self.get_domain_occupancy(chain_i, domain_j)
                else:

                    # More like undefined
                    occupancy = UNASSIGNED

                occupancies.append(occupancy)

            # Check pairs from left to right
            if occupancies[2] == BOUND:
                bound_domain = self.get_bound_domain(chain_i, domain_is[2])
                self._helical_pair_constraints_obeyed(chain_i, domain_is[2],
                    domain_i, bound_domain)
                if occupancies[1] == BOUND:
                    self._linear_helix(chain_i, domain_is[1], domain_is[2])

            if occupancies[2] == occupancies[3] == BOUND:
                self._linear_helix(chain_i, domain_is[2], domain_i)

            if occupancies[3] == BOUND:
                bound_domain = domains[domains_i - 1]
                self._helical_pair_constraints_obeyed(chain_i, domain_i,
                    domain_is[3], bound_domain)
                if occupancies[4] == BOUND:
                    self._linear_helix(chain_i, domain_i, domain_is[3])

            # Check doubly contiguous all bound constraint
            if occupancies[2] == BOUND and occupancies[3] == BOUND and occupancies[4] == BOUND:
                pdr_bound = self._calc_prev_domain(*bound_domain)
                self._check_doubly_contiguous_constraint(chain_i, domain_is[2],
                        trial_domain[1], domain_is[3], domain_is[4], pdr_bound)

            if occupancies[1] == BOUND and occupancies[2] == BOUND and occupancies[3] == BOUND:
                ndr_bound = self._next_domains[bound_domain[0]][bound_domain[1]]
                self._check_doubly_contiguous_constraint(chain_i, domain_is[1],
                        domain_is[2], trial_domain[1], domain_is[4], ndr_bound)

            if occupancies[3] == BOUND and occupancies[4] == BOUND and occupancies[5] == BOUND:
                b_domain = self.get_bound_domain(chain_i, domain_is[3])
                pdr_next_bound = self._calc_prev_domain(*b_domain)
                self._check_doubly_contiguous_constraint(chain_i, trial_domain[1],
                        domain_is[3], domain_is[4], domain_is[5], pdr_next_bound)

            if occupancies[0] == BOUND and occupancies[1] == BOUND and occupancies[2] == BOUND:
                pb_domain = self.get_bound_domain(chain_i, domain_is[2])
                ndr_pb = self._next_domains[pb_domain[0]][pb_domain[1]]
                self._check_doubly_contiguous_constraint(chain_i, domain_is[0],
                        domain_is[1], domain_is[2], trial_domain[1], ndr_pb)

        # Add new binding energies
        delta_e = self.get_hybridization_energy(*trial_domain, *occupying_domain)

        return delta_e

    def _bind_misbound_domain(self, trial_domain, occupying_domain):
        delta_e = self.get_hybridization_energies(*trial_domain, *occupying_domain)
        return delta_e

    def _no_misbound(self, *args):
        raise ConstraintViolation

    def _check_doubly_contiguous_constraint(self, chain_i, d_1, d_2, d_3, d_4, d_b_dr):
        d_2_ndr = self._next_domains[chain_i][d_2]
        if all(d_2_ndr != np.zeros(3)) and all(d_2_ndr == d_b_dr):
            d_1_pos = self._positions[chain_i][d_1]
            d_4_pos = self._positions[chain_i][d_4]
            if (d_1_pos - d_4_pos).sum().abs() != 1:
                raise ConstraintViolation

    def _domains_match(self, chain_index_1, domain_index_1,
                chain_index_2, domain_index_2):
        """Return True if domains have correct orientation and sequence."""

        # Determine domain identities
        chain_identity_1 = self._chain_identities[chain_index_1]
        domain_identity_1 = self.identities[chain_identity_1][domain_index_1]
        chain_identity_2 = self._chain_identities[chain_index_2]
        domain_identity_2 = self.identities[chain_identity_2][domain_index_2]

        # Complementary if domain identities sum to 0
        complementary = domain_identity_1 + domain_identity_2

        # Check if orientations are correct
        if complementary == 0:
            orientation_1 = self._orientations[chain_index_1][domain_index_1]
            orientation_2 = self._orientations[chain_index_2][domain_index_2]

            # They should be opposite vectors, thus correct if sum to 0
            complementary_orientations = orientation_1 + orientation_2
            if all(complementary_orientations == np.zeros(3)):
                match = True
            else:
                match = False
        else:
            match = False

        return match

    def _helical_pair_constraints_obeyed(self, chain_i, domain_i_1, domain_i_2,
            bound_domain):
        """Return True if domains not in same helix or twist constraints obeyed.

        Assumes domain occupancies are bound.

        bound_domain -- domain bound to domain_i_1
        """

        # Next domain vector
        next_dr = self._next_domains[chain_i][domain_i_1]
        o_1 = self._orientations[chain_i][domain_i_1]
        o_2 = self._orientations[chain_i][domain_i_2]

        # Only one allowed configuration not in the same helix
        if all(next_dr == o_1):

            # If bound chain's next domain bound to current chain next
            # domain, can't be a new helix
            ndr2 = self._next_domains[bound_domain[0]][bound_domain[1]]
            if ndr2 == []:
                constraints_obeyed = True
            elif all(next_dr == ndr2):
                raise ConstraintViolation
            else:
                constraints_obeyed = True

            return constraints_obeyed

        # Can't be in the same helix if next domain vector equals bound domain
        # previous domain vector
        prd = self._calc_prev_domain(*bound_domain)
        if prd == []:
            pass
        elif all(next_dr == prd):
            raise ConstraintViolation
        else:
            pass

        # Next domain being in opposite direction of orientation vector not allowed
        if all(next_dr == -o_1):
            raise ConstraintViolation
        else:
            pass

        # Check twist constraint if same helix
        constraints_obeyed = self._check_twist_constraint(next_dr, o_1, o_2)
        if not constraints_obeyed:
            raise ConstraintViolation

        return constraints_obeyed

    def _linear_helix(self, chain_i, domain_i_1, domain_i_2):
        """Returns true if the three domains are linear.
        
        Assumes domain occupancies are bound."""
        o_1 = self._orientations[chain_i][domain_i_1]
        o_2 = self._orientations[chain_i][domain_i_2]
        next_dr_1 = self._next_domains[chain_i][domain_i_1]
        next_dr_2 = self._next_domains[chain_i][domain_i_2]
        new_helix_1 = all(next_dr_1 == o_1)
        new_helix_2 = all(next_dr_2 == o_2)
        if new_helix_1 or new_helix_2:
            return True

        linear = all(next_dr_1 == next_dr_2)
        if not linear:
            raise ConstraintViolation

        return linear

    def _update_next_domain(self, chain_i, domain_i):
        """Update next domain vectors."""
        self._prev_next_domains = []
        for d_i in [domain_i - 1, domain_i]:
            if d_i < 0 or d_i >= self.chain_lengths[chain_i]:
                if self.cyclic and chain_i == SCAFFOLD_INDEX:
                    d_i = self.wrap_cyclic_scaffold(d_i)
                else:
                    continue

            p_i = m_position[c_i][d_i];
            d_j = d_i + 1;
            if d_j < 0 or d_j >= self.chain_lengths[chain_i]:
                if self.cyclic and chain_i == SCAFFOLD_INDEX:
                    d_j = self.wrap_cyclic_scaffold(d_j)
                else:
                    prev_ndr = self._next_domains[chain_i][d_i]
                    self._prev_next_domains.append(((chain_i, d_i), prev_ndr))
                    ndr = np.zeros(3)
                    self._next_domains[chain_i][d_i] = ndr
                    continue

            p_j = self.get_domain_position(chain_i, d_j)
            if p_i == [] or p_j == []:
                prev_ndr = self._next_domains[chain_i][d_i]
                self._prev_next_domains.append(((chain_i, d_i), prev_ndr))
                ndr = np.zeros(3)
                self._next_domains[chain_i][d_i] = ndr
                #self._next_domains[chain_i][d_i] = []
                continue

            ndr = p_j - p_i

            prev_ndr = self._next_domains[chain_i][d_i]
            self._prev_next_domains.append(((chain_i, d_i), prev_ndr))
            self._next_domains[chain_i][d_i] = ndr

    def _revert_next_domain(self):
        for domain, ndr in self._prev_next_domains:
            self._next_domains[domain[0]][domain[1]] = ndr

        self._prev_next_domains = []

#    def _update_prev_domain(self, chain_i, domain_i):
#        """Update next domain vectors."""
#        for d_i in [domain_i, domain_i + 1]:
#            if d_i >= 0 and d_i < self.chain_lengths[0]:
#                if self.cyclic and chain_i == SCAFFOLD_INDEX:
#                    d_i = self.wrap_cyclic_scaffold(d_i)
#                else:
#                    continue
#
#            p_i = self.get_domain_position(chain_i, d_i)
#
#            d_j = d_i - 1
#            if d_i < 0 or d_i >= self.chain_lengths[0]:
#                if self.cyclic and chain_i == SCAFFOLD_INDEX:
#                    d_j = self.wrap_cyclic_scaffold(d_j)
#                else:
#                    pdr = np.zeros(3)
#                    self._prev_domains[chain_i][d_i] = pdr
#                    continue
#
#            p_j = self.get_domain_position(chain_i, d_j)
#            if p_i == [] or p_j == []:
#                pdr = np.zeros(3)
#                self._prev_domains[chain_i][d_i] = pdr
#                continue
#
#            pdr = p_j - p_i
#
#            self._prev_domains[chain_i][d_i] = pdr

    def _calc_prev_domain(self, chain_i, d_i):
        """Update next domain vectors.
        
        Assumes that d_i is always an actual index.
        """

        p_i = self.get_domain_position(chain_i, d_i)
        d_j = d_i - 1
        if d_j < 0 or d_j >= self.chain_lengths[chain_i]:
            if self.cyclic and chain_i == SCAFFOLD_INDEX:
                d_j = self.wrap_cyclic_scaffold(d_j)
            else:
                return np.zeros(3)

        p_j = self.get_domain_position(chain_i, d_j)
        if p_i == [] or p_j == []:
            return np.zeros(3)

        pdr = p_j - p_i

        return pdr

    def _check_twist_constraint(self, *args):
        raise NotImplementedError


# Should merge these in and have init check sequence length
class OrigamiSystemEight(OrigamiSystem):
    """Origami systems with 8 bp domains."""

    def __init__(self, input_file, step, temp, strand_M, cation_M, misbinding=True):

        # Volume of lattice site (m)
        self.lattice_site_volume = 0.332e-9 * 8 * 2e-9 * 2e-9

        # System volume
        V = molarity_to_lattice_volume(strand_M, self.lattice_site_volume)
        self.volume = V
        super().__init__(input_file, step, temp, cation_M, misbinding)

    def _check_twist_constraint(self, next_dr, orientation_1, orientation_2):
        orientation_1_r = (rotate_vector_quarter(orientation_1, next_dr, -1))
        if all(orientation_1_r == orientation_2):
            constraints_obeyed = True
        else:
            constraints_obeyed = False

        return constraints_obeyed


class OrigamiSystemSixteen(OrigamiSystem):
    """Origami systems with 16 bp domains."""

    def __init__(self, input_file, step, temp, strand_M, cation_M, misbinding=True):

        # Volume of lattice site (m)
        #self.lattice_site_volume = 0.332e-9 * 16 * 2e-9 * 2e-9
        self.lattice_site_volume = 4e-28

        # System volume
        V = molarity_to_lattice_volume(strand_M, self.lattice_site_volume)
        self.volume = V
        super().__init__(input_file, step, temp, cation_M, misbinding)

    def _check_twist_constraint(self, next_dr, orientation_1, orientation_2):
        orientation_1_r = rotate_vector_half(orientation_1, next_dr)
        if all(orientation_1_r == orientation_2):
            constraints_obeyed = True
        else:
            constraints_obeyed = False

        return constraints_obeyed


class OutputFile:
    """Base output file class to allow check_and_write to be shared."""

    def check_and_write(self, origami_system, step):
        """Check property write frequencies and write accordingly."""
        if value_is_multiple(step, self._config_write_freq):
            self.write_configuration(origami_system, step)
        if value_is_multiple(step, self._count_write_freq):
            self.write_staple_domain_count(origami_system)
        if value_is_multiple(step, self._energy_write_freq):
            self.write_energy(origami_system)

        # Move types and acceptances


class JSONOutputFile(OutputFile):
    """JSON output file class."""

    def __init__(self, filename, origami_system, config_write_freq=1):
        self._filename = filename
        self._config_write_freq = config_write_freq

        self.json_origami = {'origami':{'identities':{}, 'configurations':[]}}
        self.json_origami['origami']['identities'] = origami_system.identities
        self.json_origami['origami']['sequences'] = origami_system.sequences
        self.json_origami['origami']['cyclic'] = origami_system.cyclic

    def write_seed(self, seed):
        self.json_origami['origami']['seed'] = seed

    def write_configuration(self, origami_system, step):
        self.json_origami['origami']['configurations'].append({})
        current_config = self.json_origami['origami']['configurations'][-1]
        current_config['step'] = step
        current_config['chains'] = []
        for chain in origami_system.chains:
            current_config['chains'].append({})
            json_chain = current_config['chains'][-1]
            json_chain['identity'] = chain['identity']
            json_chain['index'] = chain['index']
            json_chain['positions'] = chain['positions']
            json_chain['orientations'] = chain['orientations']

        json.dump(self.json_origami, open(self._filename, 'w'), indent=4,
                    separators=(',', ': '))

    def close(self):
        """Perform cleanup."""
        pass


class JSONInputFile:
    """Input file taking json formatted origami system files in constructor."""

    def __init__(self, filename):
        json_origami = json.load(open(filename))

        self._filename = filename
        self._json_origami = json_origami

    @property
    def cyclic(self):
        return self._json_origami['origami']['cyclic']

    @property
    def identities(self):
        """Standard format for passing origami domain identities."""
        return self._json_origami['origami']['identities']

    @property
    def sequences(self):
        """Standard format for passing origami domain sequences.

        Only includes sequences of scaffold in 5' to 3' orientation.
        """
        return self._json_origami['origami']['sequences']

    def chains(self, step):
        """Standard format for passing chain configuration."""
        return self._json_origami['origami']['configurations'][step]['chains']

    def close(self):
        pass


class HDF5OutputFile(OutputFile):
    """HDF5 output file class.

    Custom format; not compatable with VMD (not H5MD).
    """

    def __init__(self, filename, origami_system, config_write_freq=1,
                count_write_freq=0, energy_write_freq=1):
        self.hdf5_origami = h5py.File(filename, 'w')
        self.hdf5_origami.create_group('origami')
        self.filename = filename
        self._config_write_freq = config_write_freq
        self._config_writes = 0
        self._count_write_freq = count_write_freq
        self._count_writes = 0
        self._energy_write_freq = energy_write_freq
        self._energy_writes = 0

        # HDF5 does not allow variable length lists; fill with 0
        max_domains = 0
        for domain_identities in origami_system.identities:
            domains = len(domain_identities)
            if domains > max_domains:
                max_domains = domains

        filled_identities = []
        for domain_identities in origami_system.identities:
            remainder = max_domains - len(domain_identities)
            filled_identities.append(domain_identities[:])
            for i in range(remainder):
                filled_identities[-1].append(0)

        self.hdf5_origami.create_dataset('origami/identities',
                data=filled_identities)

        # Fill attributes
        self.hdf5_origami.attrs['cyclic'] = origami_system.cyclic
        self.hdf5_origami.attrs['temp'] = origami_system.temp
        self.hdf5_origami.attrs['config_write_freq'] = config_write_freq
        self.hdf5_origami.attrs['count_write_freq'] = count_write_freq

        # HDF5 does not allow lists of strings
        linear_seqs = [j for i in origami_system.sequences for j in i]
        sequences = np.array(linear_seqs, dtype='a')
        self.hdf5_origami.create_dataset('origami/sequences',
                data=sequences)

        # Setup configuration datasets
        if config_write_freq > 0:
            self.hdf5_origami.create_group('origami/configurations')
            for chain in origami_system.chains:
                self._create_chain(chain)

        # Setup up analysis datasets:
        if self._count_write_freq > 0:
            self.hdf5_origami.create_dataset('origami/staple_domain_count',
                    (1, 2),
                    maxshape=(None, 2),
                    chunks=(1, 2),
                    dtype=int)

        if self._energy_write_freq > 0:
            self.hdf5_origami.create_dataset('origami/energies',
                    (1, 1),
                    maxshape=(None, 1),
                    chunks=(1, 1),
                    dtype=float)

        # Create array of chains present at each step
        dt = h5py.special_dtype(vlen=np.dtype('int32'))
        self.hdf5_origami.create_dataset('origami/chain_ids',
                (1,),
                maxshape=(None,),
                chunks=(1,),
                dtype=dt)

    def write_seed(self, seed):

        # h5py docs recommend wrapping byte strings in np.void
        self.hdf5_origami['origami'].attrs['seed'] = np.void(seed)

    def write_staple_domain_count(self, origami_system):
        write_index = self._count_writes
        self._count_writes += 1
        num_staples = origami_system.num_staples
        num_domains = origami_system.num_bound_domains
        count_key = 'origami/staple_domain_count'
        self.hdf5_origami[count_key].resize(self._count_writes, axis=0)
        self.hdf5_origami[count_key][write_index] = (num_staples, num_domains)

    def write_energy(self, origami_system):
        write_index = self._energy_writes
        self._energy_writes += 1
        energy = origami_system.energy
        energy_key = 'origami/energies'
        self.hdf5_origami[energy_key].resize(self._energy_writes, axis=0)
        self.hdf5_origami[energy_key][write_index] = energy

    def write_configuration(self, origami_system, step):
        write_index = self._config_writes
        self._config_writes += 1
        chain_ids = []
        for chain in origami_system.chains:
            chain_index = chain['index']
            chain_ids.append(chain_index)
            base_key = 'origami/configurations/{}'.format(chain_index)
            try:
                self.hdf5_origami[base_key]
            except KeyError:
                self._create_chain(chain)

            step_key = base_key + '/step'
            pos_key = base_key + '/positions'
            orient_key = base_key + '/orientations'
            self.hdf5_origami[step_key].resize(self._config_writes, axis=0)
            self.hdf5_origami[step_key][write_index] = step
            self.hdf5_origami[pos_key].resize(self._config_writes, axis=0)
            self.hdf5_origami[pos_key][write_index] = chain['positions']
            self.hdf5_origami[orient_key].resize(self._config_writes, axis=0)
            self.hdf5_origami[orient_key][write_index] = chain['orientations']

        self.hdf5_origami['origami/chain_ids'].resize(self._config_writes, axis=0)
        self.hdf5_origami['origami/chain_ids'][write_index] = chain_ids

    def close(self):
        """Perform any cleanup."""
        self.hdf5_origami.close()

    def _create_chain(self, chain):
        chain_length = len(chain['positions'])
        chain_index = chain['index']
        base_key = 'origami/configurations/{}'.format(chain_index)
        step_key = base_key + '/step'
        pos_key = base_key + '/positions'
        orient_key = base_key + '/orientations'

        self.hdf5_origami.create_group(base_key)
        self.hdf5_origami[base_key].attrs['index'] = (chain_index)
        self.hdf5_origami[base_key].attrs['identity'] = (chain['identity'])
        self.hdf5_origami.create_dataset(step_key,
                (1, 1),
                chunks=(1, 1),
                maxshape=(None, 1),
                dtype='i')
        self.hdf5_origami.create_dataset(pos_key,
                (1, chain_length, 3),
                chunks=(1, chain_length, 3),
                maxshape=(None, chain_length, 3),
                compression='gzip',
                dtype='i')
        self.hdf5_origami.create_dataset(orient_key,
                (1, chain_length, 3),
                chunks=(1, chain_length, 3),
                maxshape=(None, chain_length, 3),
                compression='gzip',
                dtype='i')


class HDF5InputFile:
    """Input file taking hdf5 formatted origami system files in constructor."""

    def __init__(self, filename):
        hdf5_origami = h5py.File(filename, 'r')

        self._filename = filename
        self._hdf5_origami = hdf5_origami
        self.config_write_freq = hdf5_origami.attrs['config_write_freq']
        self.count_write_freq = hdf5_origami.attrs['config_write_freq']

    @property
    def cyclic(self):
        return self._hdf5_origami.attrs['cyclic']

    @property
    def identities(self):
        """Standard format for passing origami domain identities."""

        # HDF5 does not allow variable length lists; fill with 0
        raw = np.array(self._hdf5_origami['origami/identities']).tolist()
        identities = []
        for raw_domain_identities in raw:
            identities.append([i for i in raw_domain_identities if i != 0])

        return identities

    @property
    def sequences(self):
        """Standard format for passing origami system sequences.

        Only includes sequences of scaffold in 5' to 3' orientation.
        """

        # H5py outputs as type 'S', need type 'U'
        identities = self.identities
        seqs = []
        linear_i = 0
        linear_seqs = np.array(self._hdf5_origami['origami/sequences']).astype(
                'U').tolist()
        for c_i, c_idents in enumerate(identities):
            for d_i in range(len(c_idents)):
                seqs[c_i][d_i] = linear_seqs[linear_i]
                linear_i += 1

        return seqs

    @property
    def temp(self):
        return self._hdf5_origami.attrs['temp']

    @property
    def steps(self):
        return len(self._hdf5_origami['origami/chain_ids'])

    @property
    def staple_domain_counts(self):
        return self._hdf5_origami['origami/staple_domain_count']

    @property
    def energy(self):
        return self._hdf5_origami['origami/energies'][:].flatten()

    def chains(self, step):
        """Standard format for passing chain configuration."""

        # For consistency, do not autoconvert step to index
        #step = step / self.config_write_freq
        chains = []
        chain_ids = self._hdf5_origami['origami/chain_ids'][step]
        base_key = 'origami/configurations'
        for chain in chain_ids:
            chain_key = base_key + '/' + str(chain)
            chain_group = self._hdf5_origami[chain_key]
            chain = {}
            chain['index'] = chain_group.attrs['index']
            chain['identity'] = chain_group.attrs['identity']
            chain['positions'] = chain_group['positions'][step].tolist()
            chain['orientations'] = chain_group['orientations'][step].tolist()
            chains.append(chain)

        return chains

    def close(self):
        self._hdf5_origami.close()


class GCMCSimulation:
    """GCMC sim for domain-res origami model with bound staples only.

    Grand cannonical Monte Carlo simulations on a simple cubic lattice of the
    origami model defined by the origami class and associated documentation. The
    simulations run with this class do not include any moves that can lead to
    free staples.
    """

    def __init__(self, origami_system, move_settings, output_file, center_freq=1):
        self._origami_system = origami_system
        self._output_file = output_file

        # Set seed for python's Mersenne Twister (64 bits from os)
        seed = os.urandom(8)
        random.seed(seed)
        output_file.write_seed(seed)

        # Frequency for translating system back to origin
        self.center_freq = center_freq

        # Current movetype
        self._movetype = -1

        # Create cumulative probability distribution for movetypes
        # List to associate movetype method with index in distribution
        self._movetype_classes = []
        self._movetypes = []
        self._movetype_probabilities = []
        cumulative_probability = 0
        for movetype, probability in move_settings.items():

            # I still wonder if there is a way to avoid all the if statements
            if movetype == MOVETYPE.IDENTITY:
                movetype_class = IdentityMCMovetype
            if movetype == MOVETYPE.EXCHANGE_STAPLE:
                movetype_class = ExchangeMMCMovetype
            elif movetype == MOVETYPE.REGROW_STAPLE:
                movetype_class = StapleRegrowthMMCMovetype
            elif movetype == MOVETYPE.REGROW_SCAFFOLD:
                movetype_class = ScaffoldRegrowthMMCMovetype
            elif movetype == MOVETYPE.ROTATE_ORIENTATION_VECTOR:
                movetype_class = OrientationRotationMCMovetype
            if movetype == MOVETYPE.CB_EXCHANGE_STAPLE:
                movetype_class = ExchangeCBMCMovetype
            elif movetype == MOVETYPE.CB_REGROW_STAPLE:
                movetype_class = StapleRegrowthCBMCMovetype
            elif movetype == MOVETYPE.CB_REGROW_SCAFFOLD:
                movetype_class = ScaffoldRegrowthCBMCMovetype
            elif movetype == MOVETYPE.CB_CONSERVED_TOPOLOGY:
                movetype_class = ConservedTopologyCBMCMovetype
            if movetype == MOVETYPE.MIS_EXCHANGE_STAPLE:
                movetype_class = MisbindingExchangeMMCMovetype
            elif movetype == MOVETYPE.MIS_REGROW_STAPLE:
                movetype_class = MisbindingStapleRegrowthMMCMovetype
            elif movetype == MOVETYPE.MIS_REGROW_SCAFFOLD:
                movetype_class = MisbindingScaffoldRegrowthMMCMovetype
            if movetype == MOVETYPE.MIS_CB_EXCHANGE_STAPLE:
                movetype_class = MisbindingExchangeCBMCMovetype
            elif movetype == MOVETYPE.MIS_CB_REGROW_STAPLE:
                movetype_class = MisbindingStapleRegrowthCBMCMovetype
            elif movetype == MOVETYPE.MIS_CB_REGROW_SCAFFOLD:
                movetype_class = MisbindingScaffoldRegrowthCBMCMovetype
            elif movetype == MOVETYPE.MIS_CB_CONSERVED_TOPOLOGY:
                movetype_class = MisbindingConservedTopologyCBMCMovetype

            cumulative_probability += probability
            self._movetype_classes.append(movetype_class)
            self._movetypes.append(movetype)
            self._movetype_probabilities.append(cumulative_probability)

        # Check movetype probabilities are normalized
        # This could break from rounding errors
        # Might be better to normalize whatever numbers given
        if cumulative_probability != 1:
            print('Movetype probabilities not normalized')
            sys.exit()

    def run(self, num_steps, logging=1):
        """Run simulation for given number of steps."""

        for step in range(1, num_steps + 1):
            movetype_object = self._select_movetype(step)
            try:
                outcome = movetype_object.attempt_move()
            except MoveRejection:
                outcome = False

            if outcome == REJECTED:
                movetype_object.reset_origami()

            self._origami_system = movetype_object.origami_system
            if value_is_multiple(step, self.center_freq):
                self._origami_system.center()
                self._origami_system.check_all_constraints()

            self._output_file.check_and_write(self._origami_system, step)

            # Loggging hack
            if value_is_multiple(step, logging):
                print(step, self._movetype, outcome,
                        len(self._origami_system.chain_lengths))
            else:
                pass

    def _select_movetype(self, step):
        """Return movetype method according to distribution."""
        random_number = random.random()
        lower_boundary = 0
        for movetype_index, upper_boundary in enumerate(
                self._movetype_probabilities):
            if lower_boundary <= random_number < upper_boundary:
                movetype_class = self._movetype_classes[movetype_index]
                self._movetype = self._movetypes[movetype_index]
                break
            else:
                lower_boundary = upper_boundary

        movetype_object = movetype_class(self._origami_system, step)
        return movetype_object


class MCMovetype:
    """Base class for all movetype classes."""

    def __init__(self, origami_system, step):
        self.origami_system = origami_system
        self._step = step
        self._modifier = 1

        # List of chains/domains that have been modified
        self._modified_domains = []
        self._assigned_domains = []
        self._added_chains = []
        self._added_domains = []
        self._deleted_chains = []
        self._deleted_domains = []

        # Dictionary of previous configs as (pos, ore)
        self._prev_configs = {}

        # Convienience variables
        self.scaffold_length = self.origami_system.chain_lengths[0]

    def attempt_move(self):
        raise NotImplementedError

    def reset_origami(self):
        """Reset modified domains to old config."""

        # Re-add chains that were deleted
        for staple_identity, unique_index in self._deleted_chains:
            staple_index = self.origami_system.readd_chain(staple_identity,
                    unique_index)

        # First unassign those that were assigned
        for domain in self._modified_domains:
            if domain in self._assigned_domains:
                self.origami_system.unassign_domain(*domain)
            else:
                pass

        # Unassign added domains
        for domain in self._added_domains:
            if domain in self._assigned_domains:
                self.origami_system.unassign_domain(*domain)
            else:
                pass

        # Revert scaffold domains to previous positions
        for domain in self._modified_domains:
            p_old, o_old = self._prev_configs[domain]
            self.origami_system.set_checked_domain_configuration(*domain,
                    p_old, o_old)

        # Revert deleted domains with updated chain index (assumes only 1 
        # chain deleted)
        for domain in self._deleted_domains:
            p_old, o_old = self._prev_configs[domain]
            self.origami_system.set_checked_domain_configuration(staple_index,
                    domain[1], p_old, o_old)

        # Delete chains that were added
        for staple_i in self._added_chains:
            self.origami_system.delete_chain(staple_i)

    def _test_acceptance(self, ratio):
        """Metropolis acceptance test for given ratio."""
        p_accept = min(1, ratio) * self._modifier
        if p_accept == 1:
            accept = True
        else:
            if p_accept > random.random():
                accept = True
            else:
                accept = False

        return accept

    def _find_bound_staples_with_complements(self, scaffold_indices):
        """Find all staples bound to scaffold segment.

        Returns dictionary of staple indices and complementary scaffold domains
        in provided scaffold indices, with corresponding staple domain index.
        """
        staples = {}
        for domain_index in scaffold_indices:

            # Check if scaffold domain bound
            staple_domain = self.origami_system.get_bound_domain(
                    SCAFFOLD_INDEX, domain_index)
            if staple_domain == ():
                continue
            else:
                pass

            # Find remaining complementary domains to bound staple
            staple_index = staple_domain[0]
            staple_domain_i = staple_domain[1]
            if staple_index in staples:
                continue
            else:
                pass

            staples[staple_index] = []
            comp_domains = self.origami_system.get_complementary_domains(
                    staple_index)
            for staple_domain_i, scaffold_domain_i in enumerate(comp_domains):
                if scaffold_domain_i in scaffold_indices:
                    staples[staple_index].append((staple_domain_i,
                            scaffold_domain_i))

        return staples

    def _find_bound_staples(self, scaffold_indices):
        """Return set of all staples bound to scaffold segment."""
        staples = {}
        for domain_index in scaffold_indices:

            # Check if scaffold domain bound
            staple_domain = self.origami_system.get_bound_domain(
                    SCAFFOLD_INDEX, domain_index)
            if staple_domain == ():
                continue
            else:

                # 0 is just a placeholder, using a dictionary for consistency
                # with above method
                staples[staple_domain[0]] = 0

        return staples

    def _find_and_pick_externally_bound_staples(self, staples, scaffold_indices):
        """Find staples bound to scaffold domain outside selected segment.

        Also picks whether to grow them from that segment or leave them bound
        to the external domain, and modifies the staples dictionary accordingly.
        """
        externally_bound = []
        for staple_index in staples.keys():
            staple_length = self.origami_system.chain_lengths[staple_index]
            num_bound = 0
            for staple_d_i in range(staple_length):
                bound_domain = self.origami_system.get_bound_domain(
                        staple_index, staple_d_i)
                if bound_domain == () or bound_domain[1] in scaffold_indices:
                    continue
                else:
                    occupancy = self.origami_system.get_domain_occupancy
                    if occupancy in [BOUND, MISBOUND]:
                        num_bound += 1
                    else:
                        pass

            if num_bound > 0:
                externally_bound.append((staple_index, staple_length,
                    num_bound))
            else:
                pass

        # Pick to grow from scaffold segment or leave
        for staple_index, staple_length, num_bound in externally_bound:
            if random.random() < (num_bound / staple_length):
                del staples[staple_index]
            else:
                pass

        return staples

    def _set_staple_growth_point(self, staple_c_i, staple_d_i,
                scaffold_c_i, scaffold_d_i):
        """Given scaffold and staple, attempt to bind with correct orientation.
        """
        p_growth = self.origami_system.get_domain_position(scaffold_c_i,
                scaffold_d_i)
        o_growth = -self.origami_system.get_domain_orientation(
                scaffold_c_i, scaffold_d_i)

        # Attempt to set growth domain
        try:
            delta_e = self.origami_system.set_domain_configuration(
                    staple_c_i, staple_d_i, p_growth, o_growth)
            self._assigned_domains.append((staple_c_i, staple_d_i))
        except ConstraintViolation:
            raise MoveRejection

        return delta_e

    def _set_new_staple_growth_point(self, staple_c_i, staple_d_i,
            growth_c_i, growth_d_i):
        """Given growth points, attempt to bind with random orientation."""
        p_growth = self.origami_system.get_domain_position(growth_c_i,
                growth_d_i)
        o_growth = self._select_random_orientation()

        # Attempt to set growth domain
        try:
            delta_e = self.origami_system.set_domain_configuration(
                    staple_c_i, staple_d_i, p_growth, o_growth)
            self._assigned_domains.append((staple_c_i, staple_d_i))
        except ConstraintViolation:
            raise MoveRejection

        return delta_e

    def _select_random_orientation(self):
        """Select a random orientation."""
        return random.choice(VECTORS)

    def _select_random_position(self, p_prev):
        """Select a random position."""
        return p_prev + random.choice(VECTORS)

    def _calc_overcount(self, chain_index, domain_i):
        bound_domain = self.origami_system.get_bound_domain(chain_index,
                domain_i)
        if bound_domain != ():
            if chain_index == SCAFFOLD_INDEX:
                staple_l = self.origami_system.chain_lengths[bound_domain[0]]
            else:
                staple_l = self.origami_system.chain_lengths[chain_index]

            self._modifier *= (1 / staple_l)
        else:
            pass

    def _add_prev_config(self, chain_i, domain_i):
        p_old = self.origami_system.get_domain_position(chain_i, domain_i)
        o_old = self.origami_system.get_domain_orientation(chain_i, domain_i)
        self._prev_configs[(chain_i, domain_i)] = (p_old, o_old)

class IdentityMCMovetype(MCMovetype):
    """Identity movetype."""

    def attempt_move(self):
        """Accept current configuration."""
        accepted = True
        return accepted


class OrientationRotationMCMovetype(MCMovetype):
    """Class for rotating the orientation vectors."""

    def attempt_move(self):
        """Randomly rotate random domain."""

        # Select random chain and domain
        chain_lengths = self.origami_system.chain_lengths
        chain_index = random.randrange(len(chain_lengths))
        domain_index = random.randrange(chain_lengths[chain_index])

        # Reject if in bound state
        occupancy = self.origami_system.get_domain_occupancy(chain_index,
                domain_index)
        if occupancy == BOUND:
            accepted = False
            return accepted
        else:
            pass

        # Select random orientation and update (always accepted)
        o_new = self._select_random_orientation()
        self.origami_system.set_domain_orientation(chain_index, domain_index,
                o_new)

        accepted = True
        return accepted


class MMCMovetype(MCMovetype):
    """Base class for Metropolis MC movetypes."""

    def __init__(self, *args, **kwargs):
        self._delta_e = 0
        super().__init__(*args, **kwargs)

    def attempt_move(self):
        raise NotImplementedError

    def _grow_staple(self, staple_length, staple_index, domain_index):
        """Randomly grow staple out from given domain in both directions."""

        # Grow in three-prime direction
        staple_indices = range(domain_index, staple_length)
        self._grow_chain(staple_index, staple_indices)

        # Grow in five-prime direction
        staple_indices = range(domain_index, -1, -1)
        self._grow_chain(staple_index, staple_indices)

    def _grow_chain(self, chain_index, domain_indices):
        """Randomly grow out chain from given domain indices.

        Updates changes in energy as binding events occur."""

        # Iterate through given indices, growing next domain from current index
        for i, domain_i in enumerate(domain_indices[1:]):

            # Position vector of previous domain
            prev_domain_i = domain_indices[i]
            p_prev = self.origami_system.get_domain_position(chain_index,
                    prev_domain_i)

            # Randomly select neighbour lattice site for new position
            p_new = self._select_random_position(p_prev)

            # Randomly select new orientation
            o_new = self._select_random_orientation()

            # Attempt to set position
            try:
                self._delta_e += self.origami_system.set_domain_configuration(
                        chain_index, domain_i, p_new, o_new)
                self._assigned_domains.append((chain_index, domain_i))
            except ConstraintViolation:
                raise MoveRejection

            # Update overcounts
            self._calc_overcount(chain_index, domain_i)

        return


class ExchangeMMCMovetype(MMCMovetype):
    """Simple staple exchange movetype with Metropolis acceptance.
    
    Default is for fully bound domains only.
    """

    # Number of pre-constrained internal degrees of freedom
    preconstrained_df = 1
    
    # Number of possible insertion sites for selected domain
    insertion_sites = 1

    def attempt_move(self):
        if random.random() < 0.5:
            accepted = self._delete_staple()
        else:
            accepted = self._insert_staple()

        return accepted

    def _staple_insertion_accepted(self, identity):
        """Metropolis acceptance test for particle insertion."""
        T = self.origami_system.temp
        boltz_factor = math.exp(-self._delta_e / T)
        Ni_new = self.origami_system.get_num_staples(identity)

        # Correct for extra states from additional staple domains
        staple_length = len(self.origami_system.identities[identity])
        extra_states = 6**(2 * staple_length - 1 - self.preconstrained_df)
        ratio = extra_states / (Ni_new) * boltz_factor

        # Correct for insertion to subset of volume
        V = self.origami_system.volume
        self._modifier *= (self.insertion_sites / V)

        # Correct for only considering 1 of 2 ways insertion could happen
        self._modifier *= 2

        return self._test_acceptance(ratio)

    def _staple_deletion_accepted(self, identity):
        """Metropolis acceptance test for particle deletion."""
        T = self.origami_system.temp
        boltz_factor = math.exp(-self._delta_e / T)
        Ni_new = self.origami_system.get_num_staples(identity)

        # Correct for extra states from additional staple domains
        staple_length = len(self.origami_system.identities[identity])
        extra_states = 6**(2 * staple_length - 1 - self.preconstrained_df)
        ratio = (Ni_new + 1) / extra_states * boltz_factor
        return self._test_acceptance(ratio)

    def _insert_staple(self):
        """Insert staple at random scaffold domain and grow."""

        # Randomly select staple identity and add chain
        staple_c_i_ident, domain_identities = (
                self.origami_system.get_random_staple_identity())

        staple_c_i = self.origami_system.add_chain(staple_c_i_ident)
        self._added_chains.append(staple_c_i)
        for d_i in range(len(domain_identities)):
            self._added_domains.append((staple_c_i, d_i))

        staple_length = self.origami_system.chain_lengths[staple_c_i]

        # Select staple domain
        staple_d_i = random.randrange(staple_length)

        # Set growth point domain
        self._set_growth_point(staple_c_i, staple_d_i, domain_identities)

        # Grow staple
        self._grow_staple(staple_length, staple_c_i, staple_d_i)

        # Test acceptance
        if self._staple_insertion_accepted(staple_c_i_ident):
            accepted = True
        else:
            accepted = False

        return accepted

    def _delete_staple(self):
        """Delete random staple."""

        # Randomly select staple identity
        staple_identity, domain_identities = (
                self.origami_system.get_random_staple_identity())

        # Randomly select staple
        try:
            staple_index = self.origami_system.get_random_staple_of_identity(
                    staple_identity)

        # No staples in system
        except IndexError:
            raise MoveRejection

        # Delete staple
        for domain_i in range(len(domain_identities)):
            self._add_prev_config(staple_index, domain_i)
            self._deleted_domains.append((staple_index, domain_i))
            self._delta_e += self.origami_system.unassign_domain(staple_index, domain_i)

        unique_index = self.origami_system._working_to_unique[staple_index]
        self.origami_system.delete_chain(staple_index)
        self._deleted_chains.append((staple_identity, unique_index))

        # Test acceptance
        if self._staple_deletion_accepted(staple_identity):
            self.origami_system = self.origami_system
            accepted = True
        else:
            accepted = False

        return accepted

    def _set_growth_point(self, staple_c_i, staple_d_i, domain_identities):
        """Select complemenetrary scaffold domain as growth point."""
        staple_d_i_ident = domain_identities[staple_d_i]
        scaffold_d_i = self.origami_system.identities[SCAFFOLD_INDEX].index(
                -staple_d_i_ident)
        self._delta_e += self._set_staple_growth_point(staple_c_i,
                    staple_d_i, SCAFFOLD_INDEX, scaffold_d_i)


class MisbindingExchangeMMCMovetype(ExchangeMMCMovetype):
    """MMC staple exchange movetype for origami with misbinding allowed."""

    # Number of pre-constrained internal degrees of freedom
    preconstrained_df = 0

    # Number of possible insertion sites for selected domain
    insertion_sites = 1

    def _set_growth_point(self, staple_c_i, staple_d_i, *args):
        """Select a random growth site."""
        growth_c_i = staple_c_i
        while growth_c_i == staple_c_i:
            growth_c_i = random.randrange(len(self.origami_system.chain_lengths))

        growth_d_i = random.randrange(self.origami_system.chain_lengths[
                growth_c_i])
        self._delta_e += self._set_new_staple_growth_point(staple_c_i,
                    staple_d_i, growth_c_i, growth_d_i)


class RegrowthMMCMovetype(MMCMovetype):
    """Base class for simple Metropolis conformational change movetypes."""

    def attempt_move(self):
        raise NotImplementedError

    def _configuration_accepted(self):
        """Metropolis acceptance test for configuration change."""
        T = self.origami_system.temp
        boltz_factor = math.exp(-self._delta_e / T)
        return self._test_acceptance(boltz_factor)


class StapleRegrowthMMCMovetype(RegrowthMMCMovetype):
    """Simple Metropolis staple regrowth movetype."""

    _set_growth_point = MMCMovetype._set_staple_growth_point
 
    def attempt_move(self):
        """Regrow random staple from randomly chosen complementary domain."""

        # Randomly select staple
        try:
            staple_c_i = random.randrange(1,
                    len(self.origami_system.chain_lengths))

        # No staples in system
        except ValueError:
            raise MoveRejection

        # Find all growth points and randomly select growth point
        staple_length = self.origami_system.chain_lengths[staple_c_i]
        staple_d_i = random.randrange(staple_length)
        growth_cd_i = self._select_growth_point(staple_c_i, staple_d_i)

        # Unassign domains
        for d_i in range(staple_length):
            self._add_prev_config(staple_c_i, d_i)
            self._modified_domains.append((staple_c_i, d_i))
            self._delta_e += self.origami_system.unassign_domain(staple_c_i,
                    d_i)

        # Grow staple
        self._delta_e += self._set_growth_point(staple_c_i, staple_d_i,
                *growth_cd_i)
        self._grow_staple(staple_length, staple_c_i, staple_d_i)

        # Test acceptance
        if self._configuration_accepted():
            accepted = True
        else:
            accepted = False

        return accepted

    def _select_growth_point(self, staple_c_i, staple_d_i):
        comp_domains = self.origami_system.get_complementary_domains(staple_c_i)
        growth_c_i = SCAFFOLD_INDEX
        growth_d_i = comp_domains[staple_d_i]
        return growth_c_i, growth_d_i


class MisbindingStapleRegrowthMMCMovetype(StapleRegrowthMMCMovetype):
    
    _set_growth_point = MCMovetype._set_new_staple_growth_point

    def _select_growth_point(self, staple_c_i, staple_d_i):
        growth_c_i = staple_c_i
        while growth_c_i == staple_c_i:
            growth_c_i = random.randrange(len(self.origami_system.chain_lengths))

        growth_d_i = random.randrange(self.origami_system.chain_lengths[
            growth_c_i])
        return growth_c_i, growth_d_i


class ScaffoldRegrowthMMCMovetype(RegrowthMMCMovetype):
    """Simple Metropolis scaffold and bound staple regrowth movetype."""

    _find_staples = MCMovetype._find_bound_staples_with_complements

    def attempt_move(self):
        """Randomly regrow terminal section of scaffold and bound staples.

        From a randomly selected scaffold domain in a random direction to the
        end.
        """

        # Pick section of scaffold to regrow
        scaffold_indices = self._select_scaffold_indices()

        # Find bound staples and all complementary domains
        staples = self._find_staples(scaffold_indices)
        staples = self._find_and_pick_externally_bound_staples(staples,
                scaffold_indices)

        # Unassign scaffold domains
        for d_i in scaffold_indices[1:]:
            self._add_prev_config(SCAFFOLD_INDEX, d_i)
            self._modified_domains.append((SCAFFOLD_INDEX, d_i))
            self._delta_e += self.origami_system.unassign_domain(SCAFFOLD_INDEX,
                    d_i)

        # Unassign staples
        for c_i in staples.keys():
            for d_i in range(self.origami_system.chain_lengths[c_i]):
                self._add_prev_config(c_i, d_i)
                self._modified_domains.append((c_i, d_i))
                self._delta_e += self.origami_system.unassign_domain(c_i, d_i)

        # Regrow scaffold and staples
        self._grow_chain(SCAFFOLD_INDEX, scaffold_indices)
        self._grow_staples(staples)

        # Test acceptance
        if self._configuration_accepted():
            accepted = True
        else:
            accepted = False

        return accepted

    def _grow_staples(self, staples):
        for staple_c_i, comp_domains in staples.items():

            # Pick domain on scaffold and staple to grow from
            staple_d_i, scaffold_d_i = random.choice(comp_domains)
            self._delta_e += self._set_staple_growth_point(staple_c_i,
                    staple_d_i, SCAFFOLD_INDEX, scaffold_d_i)

            # Grow remainder of staple
            staple_length = self.origami_system.chain_lengths[staple_c_i]
            self._grow_staple(staple_length, staple_c_i, staple_d_i)

    def _select_scaffold_indices(self):
        """Return scaffold indices from random segment to end."""

        # Randomly select starting scaffold domain
        start_d_i = random.randrange(self.scaffold_length)

        # Select direction to regrow, create index list
        direction = random.randrange(2)
        if direction == 1:
            scaffold_indices = range(start_d_i, self.scaffold_length)
        else:
            scaffold_indices = range(start_d_i, -1, -1)

        return scaffold_indices

class MisbindingScaffoldRegrowthMMCMovetype(ScaffoldRegrowthMMCMovetype):

    _find_staples = MCMovetype._find_bound_staples

    def _grow_staples(self, staples):
        for staple_c_i in staples.keys():

            # Pick domain on scaffold and staple to grow from
            staple_length = self.origami_system.chain_lengths[staple_c_i]
            staple_d_i = random.randrange(staple_length)
            scaffold_d_i = random.randrange(self.scaffold_length)
            self._delta_e += self._set_new_staple_growth_point(staple_c_i,
                    staple_d_i, 0, scaffold_d_i)

            # Grow remainder of staple
            self._grow_staple(staple_length, staple_c_i, staple_d_i)

class CBMCMovetype(MCMovetype):
    """Base class for configurational bias movetypes."""

    def attempt_move(self):
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        self._bias = 1
        self._old_configs = {}
        super().__init__(*args, **kwargs)

    def _calc_rosenbluth(self, weights, *args):
        """calculate rosenbluth weight and return normalized weights."""
        rosenbluth_i = sum(weights)

        # deadend
        if rosenbluth_i == 0:
            raise MoveRejection

        weights = (np.array(weights) / rosenbluth_i).tolist()
        self._bias *= rosenbluth_i
        return weights

    def _select_config_with_bias(self, weights, configs, **kwargs):
        """Select configuration according to Rosenbluth weights."""
        random_n = random.random()
        cumulative_prob = 0
        for i, weight in enumerate(weights):
            cumulative_prob += weight
            if random_n < cumulative_prob:
                config = configs[i]
                break

        return config

    def _select_old_config(self, *args, domain=None):
        """Select old configuration."""
        p_old, o_old = self._old_configs[domain]
        return (p_old, o_old)

    def _grow_staple(self, staple_i, growth_domain_i, regrow_old=False,
            overcount_cor=True):
        """Grow segment of a staple chain with configurational bias."""

        staple_length = self.origami_system.chain_lengths[staple_i]
        self._calc_bias = self._calc_rosenbluth
        if regrow_old:
            self._select_config = self._select_old_config
        else:
            self._select_config = self._select_config_with_bias

        self._update_endpoints = lambda x: None

        # Grow in three-prime direction
        staple_indices = range(growth_domain_i, staple_length)
        for i, domain_i in enumerate(staple_indices[1:]):
            prev_domain_i = staple_indices[i]
            self._select_position(staple_i, domain_i, prev_domain_i)

            # Update overcounts
            if overcount_cor and not regrow_old:
                self._calc_overcount(staple_i, domain_i)

        # Grow in five-prime direction
        staple_indices = range(growth_domain_i, -1, -1)
        for i, domain_i in enumerate(staple_indices[1:]):
            prev_domain_i = staple_indices[i]
            self._select_position(staple_i, domain_i, prev_domain_i)

            # Update overcounts
            if overcount_cor and not regrow_old:
                self._calc_overcount(staple_i, domain_i)

    def _select_position(self, chain_index, domain_i, prev_domain_i):
        """Select next domain configuration with CB."""

        # Position vector of previous domain
        p_prev = self.origami_system.get_domain_position(chain_index,
                prev_domain_i)

        # List of tuples of position and orientations (if bound, orientation
        # vector, otherwise the number of possible orientation (6)
        configs = []

        # List of associated boltzmann factors
        bfactors = []

        # Iterate through all possible new positions
        for vector in VECTORS:

            # Trial position vector
            p_new = p_prev + vector

            # Check energies of each configuration
            occupancy = self.origami_system.get_position_occupancy(p_new)

            # No contribution from blocked domain
            if occupancy in (BOUND, MISBOUND):
                continue

            # Add energy and configuration if binding possible
            elif occupancy == UNBOUND:
                unbound_domain = self.origami_system.get_unbound_domain(p_new)
                orientation = -self.origami_system.get_domain_orientation(
                        *unbound_domain)
                try:
                    delta_e = self.origami_system.check_domain_configuration(
                            chain_index, domain_i, p_new, orientation,
                            self._step)
                except ConstraintViolation:
                    pass
                else:
                    configs.append((p_new, orientation))
                    bfactor = math.exp(-delta_e / self.origami_system.temp)
                    bfactors.append(bfactor)

            # If unoccupied site, all 6 orientations possible
            else:
                num_orientations = 6
                configs.append((p_new, num_orientations))
                bfactors.append(6)

        # Check if dead end
        if configs == []:
            raise MoveRejection

        # Calculate bias and select position
        weights = self._calc_bias(bfactors, configs, p_prev)
        domain = (chain_index, domain_i)
        selected_config = self._select_config(weights, configs, domain=domain)

        # If unnoccupied lattice site, randomly select orientation
        p_new = selected_config[0]
        if isinstance(selected_config[1], int):
            o_new = self._select_random_orientation()
            self.origami_system.check_domain_configuration(chain_index,
                    domain_i, p_new, o_new, self._step)
            self.origami_system.set_checked_domain_configuration(
                    chain_index, domain_i, p_new, o_new)

        # Otherwise use complementary orientation
        else:
            o_new = selected_config[1]

            # Check again in case old configuration selected
            self.origami_system.check_domain_configuration(chain_index,
                    domain_i, p_new, o_new, self._step)
            self.origami_system.set_checked_domain_configuration(
                    chain_index, domain_i, p_new, o_new)

        # Update endpoints
        self._update_endpoints(domain_i)

        # Updated assigned domain list
        self._assigned_domains.append((chain_index, domain_i))

        return

    def _get_bound_domains(self, staple_index):
        """Unassign domains and collect bound domain indices."""

        # List of staple_domain_i and scaffold_domain_i
        bound_domains = []
        for domain_i in range(self.origami_system.chain_lengths[staple_index]):
            bound_domain = self.origami_system.get_bound_domain(staple_index,
                    domain_i)
            if bound_domain != ():
                bound_domains.append((domain_i, bound_domain[1]))
            else:
                pass

        return bound_domains

#
#class ExchangeCBMCMovetype(CBMCMovetype):
#    """CB staple exchange movetype."""
#
#    def attempt_move(self):
#        if random.random() < 0.5:
#            accepted = self._delete_staple()
#        else:
#            accepted = self._insert_staple()
#
#        return accepted
#
#    def _staple_insertion_accepted(self, identity, length, overcounts):
#        """Metropolis acceptance test for particle insertion."""
#        Ni_new = self.origami_system.get_num_staples(identity)
#
#        # Number of neighbouring lattice sites
#        k = 6
#
#        # Subtract one as I don't multiply the first weight by k
#        ratio = self._bias / (Ni_new) / k**(length - 1)
#
#        # Correct for overcounts and insertint to subset of volume
#        p_accept = min(1, ratio) / overcounts / self.origami_system.volume
#        if p_accept == 1:
#            accept = True
#        else:
#            if p_accept > random.random():
#                accept = True
#            else:
#                accept = False
#
#        return accept
#
#    def _staple_deletion_accepted(self, identity):
#        """Metropolis acceptance test for particle deletion."""
#        Ni_new = self.origami_system.get_num_staples(identity)
#
#        # Number of neighbouring lattice sites
#        k = 6
#
#        # Subtract one as I don't multiply the first weight by k
#        staple_length = len(self.origami_system.sequences[identity])
#        ratio = (Ni_new - 1) * k**(staple_length - 1) / self._bias
#        return self._test_acceptance(ratio)
#
#    def _insert_staple(self):
#        """Insert staple at random scaffold domain and grow.
#        
##        Not finished. Exchange wrong and missing updates on old
#        configs and modified domans.
#        """
#
#        # Randomly select staple identity and add chain
#        staple_identity, domain_identities = (
#                self.origami_system.get_random_staple_identity())
#
#        staple_index = self.origami_system.add_chain(staple_identity)
#        staple_length = self.origami_system.chain_lengths[staple_index]
#
#        # Select staple domain
#        staple_domain = random.randrange(staple_length)
#        domain_identity = domain_identities[staple_domain]
#
#        # Select complementary scaffold domain
#        scaffold_domain = self.origami_system.identities[SCAFFOLD_INDEX].index(
#                -domain_identity)
#
#        # Number of bound domains in system (for calculating overcounts)
#        init_num_bound_domains = self.origami_system.num_bound_domains
#
#        # Set growth point domain and grow staple
#        delta_e = self._set_staple_growth_point(staple_index, staple_domain,
#                scaffold_domain)
#        self._bias *= math.exp(-delta_e / self.origami_system.temp)
#        self._grow_staple(staple_index, staple_domain)
#
#        # If the configuration is such that the staple can bind with the other
#        # domain, then there are two ways this can happen, so the ratio should
#        # be halved to prevent overcounting. If staple ends in multiply bound
#        # state, save resulting overcounts.
#        cur_num_bound_domains = self.origami_system.num_bound_domains
#        D_bind_state = cur_num_bound_domains - init_num_bound_domains
#        overcounts = D_bind_state
#
#        # Test acceptance
#        if self._staple_insertion_accepted(staple_identity, staple_length,
#                overcounts):
#            self.origami_system = self.origami_system
#            accepted = True
#        else:
#            accepted = False
#
#        return accepted
#
#    def _delete_staple(self):
#        """Delete random staple.
#
#        Not finished. Exchange wrong and missing updates on old
#        configs and modified domans.
#        """
#        
#        # Randomly select staple identity
#        staple_identity, domain_identities = (
#                self.origami_system.get_random_staple_identity())
#
#        # Randomly select staple
#        try:
#            staple_index = self.origami_system.get_random_staple_of_identity(
#                    staple_identity)
#
#        # No staples in system
#        except IndexError:
#            raise MoveRejection
#
#        # Select domain to regrow from
#        # Note the unassign method does not add unassigning energy to bias
#        bound_domains = self._unassign_staple_and_collect_bound(staple_index)
#        self._bias = 1
#        staple_domain_i, scaffold_domain_i = random.choice(bound_domains)
#
#        # Set growth point and regrow
#        delta_e = self._set_staple_growth_point(staple_index, staple_domain_i,
#                scaffold_domain_i)
#        self._bias *= math.exp(-delta_e / self.origami_system.temp)
#        self._grow_staple(staple_index, staple_domain_i,
#                regrow_old=True)
#
#        # Delete chain to create correct trial system config
#        self.origami_system.delete_chain(staple_index)
#
#        # Test acceptance
#        if self._staple_deletion_accepted(staple_identity):
#            self.origami_system = self.origami_system
#            accepted = True
#        else:
#            accepted = False
#
#        return accepted
#

class RegrowthCBMCMovetype(CBMCMovetype):

    def __init__(self, origami_system, *args):
        self._endpoints = {}
        self._endpoints['indices'] = []
        self._endpoints['positions'] = []
        self._endpoints['steps'] = np.array([], dtype=int)
        if origami_system.cyclic:
            self._select_scaffold_indices = self._select_scaffold_indices_cyclic
        else:
            self._select_scaffold_indices = self._select_scaffold_indices_linear

        super().__init__(origami_system, *args)

    def attempt_move(self):
        raise NotImplementedError

    def _calc_fixed_end_rosenbluth(self, weights, configs, p_prev):
        """Return fixed endpoint weights."""

        # Bias weights with number of walks
        for i, config in enumerate(configs):
            start_point = config[0]
            num_walks = 1
            for endpoint_i in range(len(self._endpoints['indices'])):
                endpoint_p = self._endpoints['positions'][endpoint_i]
                endpoint_s = self._endpoints['steps'][endpoint_i]
                num_walks *= IDEAL_RANDOM_WALKS.num_walks(start_point,
                        endpoint_p, endpoint_s)

            weights[i] *= num_walks

        # Calculate number of walks for previous position
        num_walks = 1
        for endpoint_i in range(len(self._endpoints['indices'])):
            endpoint_p = self._endpoints['positions'][endpoint_i]
            endpoint_s = self._endpoints['steps'][endpoint_i] + 1
            num_walks *= IDEAL_RANDOM_WALKS.num_walks(p_prev,
                    endpoint_p, endpoint_s)

        # Modified Rosenbluth
        weights_sum = sum(weights)
        bias = weights_sum / num_walks
        self._bias *= bias
        if bias == 0:
            raise MoveRejection

        weights = (np.array(weights) / weights_sum).tolist()
        return weights

    def _select_scaffold_indices_linear(self):
        """Return scaffold indices between two random segments."""

        # Randomly select end points
        start_domain_i = end_domain_i = random.randrange(
                self.scaffold_length)
        while start_domain_i == end_domain_i:
            end_domain_i = random.randrange(self.scaffold_length)

        # Select direction to regrow, create index list
        if start_domain_i < end_domain_i:
            scaffold_indices = range(start_domain_i, end_domain_i + 1)
        else:
            scaffold_indices = range(start_domain_i, end_domain_i - 1, -1)

        # No endpoint if regrowing to end
        if end_domain_i in (0, self.scaffold_length - 1):
            pass

        # Set endpoint
        else:
            self._endpoints['indices'].append(end_domain_i)
            endpoint_p = self.origami_system.get_domain_position(SCAFFOLD_INDEX,
                    end_domain_i)
            self._endpoints['positions'].append(endpoint_p)
            steps = scaffold_indices.index(end_domain_i) - 1
            self._endpoints['steps'] = np.concatenate([self._endpoints['steps'],
                    [steps]])

        return scaffold_indices

    def _select_scaffold_indices_cyclic(self):
        """Return scaffold indices between two random segments."""

        # Randomly select end points
        start_domain_i = end_domain_i = random.randrange(
                self.scaffold_length)
        while start_domain_i == end_domain_i:
            end_domain_i = random.randrange(self.scaffold_length)

        # Ensure start domain is 5'
        if start_domain_i > end_domain_i:
            start_domain_i, end_domain_i = end_domain_i, start_domain_i

        # Select direction to regrow
        direction = random.randrange(2)
        if direction == 1:
            pass
        else:
            start_domain_i, end_domain_i = end_domain_i, start_domain_i

        # Create index list
        if start_domain_i > end_domain_i:
            scaffold_indices = []
            for domain_i in range(start_domain_i, self.scaffold_length +
                    end_domain_i + 1):
                wd_i = self.origami_system.wrap_cyclic_scaffold(domain_i)
                scaffold_indices.append(wd_i)
        else:
            scaffold_indices = range(start_domain_i, end_domain_i + 1)

        # Set endpoint
        endpoint_i = scaffold_indices[-1]
        self._endpoints['indices'].append(end_domain_i)
        endpoint_p = self.origami_system.get_domain_position(SCAFFOLD_INDEX,
                endpoint_i)
        self._endpoints['positions'].append(endpoint_p)
        steps = len(scaffold_indices) - 2
        self._endpoints['steps'] = np.concatenate([self._endpoints['steps'],
                [steps]])

        return scaffold_indices

    def _find_bound_staples_with_bound(self, scaffold_indices):
        """Find all bound staples and return with bound scaffold domains.

        Includes only bound scaffold domains in given indices.
        """
        staples = {}
        for domain_index in scaffold_indices:

            # Check if scaffold domain bound
            staple_domain = self.origami_system.get_bound_domain(
                    SCAFFOLD_INDEX, domain_index)
            if staple_domain == ():
                continue
            else:
                staple_index, staple_domain_i = staple_domain
                if not staple_index in staples:
                    staples[staple_index] = []
                else:
                    pass

                staples[staple_index].append((staple_domain_i, domain_index))

        return staples


class StapleRegrowthCBMCMovetype(RegrowthCBMCMovetype):
    """CB staple regrowth movetype."""

    def attempt_move(self):
        """Regrow random staple."""

        # Randomly select staple
        try:
            staple_index = random.randrange(1,
                    len(self.origami_system.chain_lengths))

        # No staples in system
        except ValueError:
            raise MoveRejection

        # Find all bound domains (for regrowing old)
        bound_domains = self._get_bound_domains(staple_index)

        # Find all complementary domains and randomly select growth point
        comp_domains = self.origami_system.get_complementary_domains(staple_index)
        staple_length = self.origami_system.chain_lengths[staple_index]
        staple_domain_i = random.randrange(staple_length)
        scaffold_domain_i = comp_domains[staple_domain_i]

        # Unassign domains
        for domain_index in range(staple_length):
            self._add_prev_config(staple_index, domain_index)
            self._modified_domains.append((staple_index, domain_index))
            self.origami_system.unassign_domain(staple_index,
                    domain_index)
            # Do not count as would be double counting
            #self._bias *= math.exp(-delta_e / self.origami_system.temp)

        # Grow staple
        delta_e = self._set_staple_growth_point(staple_index,
                staple_domain_i, 0, scaffold_domain_i)
        self._bias *= math.exp(-delta_e / self.origami_system.temp)
        self._grow_staple(staple_index, staple_domain_i)

        # Regrow staple in old conformation
        new_bias = self._bias
        self._bias = 1
        self._modified_domains = []
        self._assigned_domains = []
        self._old_configs = copy.deepcopy(self._prev_configs)

        # Unassign and pick a starting point at random (will average over
        # simulation)
        for domain_index in range(staple_length):
            self._add_prev_config(staple_index, domain_index)
            self._modified_domains.append((staple_index, domain_index))
            self.origami_system.unassign_domain(staple_index,
                    domain_index)

        staple_domain_i, scaffold_domain_i = random.choice(bound_domains)

        # Grow staple
        delta_e = self._set_staple_growth_point(staple_index, staple_domain_i,
                0, scaffold_domain_i)
        self._bias *= math.exp(-delta_e / self.origami_system.temp)
        self._grow_staple(staple_index, staple_domain_i,
                regrow_old=True)

        # Test acceptance
        ratio = new_bias / self._bias
        if self._test_acceptance(ratio):
            self.reset_origami()
            accepted = True
        else:
            self._modified_domains = []
            self._assigned_domains = []
            accepted = False

        return accepted


class ScaffoldRegrowthCBMCMovetype(RegrowthCBMCMovetype):
    """CB scaffold regrowth movetype."""

    def attempt_move(self):
        """Randomly regrow terminal section of scaffold and bound staples.

        From a randomly selected scaffold domain in a random direction to the
        end.
        """

        # Pick section of scaffold to regrow
        scaffold_indices = self._select_scaffold_indices()
        initial_endpoints = copy.deepcopy(self._endpoints)

        # Find bound staples and all complementary domains
        staples = self._find_bound_staples_with_complements(scaffold_indices)
        staples = self._find_and_pick_externally_bound_staples(staples,
                scaffold_indices)

        # Find bound staples and bound domains
        staples_bound = self._find_bound_staples_with_bound(scaffold_indices)
        staples = self._find_and_pick_externally_bound_staples(
                staples_bound, scaffold_indices)

        # Regrow scaffold
        self._unassign_domains(scaffold_indices, staples)
        self._grow_scaffold(scaffold_indices)

        # Regrow staples
        self._regrow_staples(staples)

        # Regrow in old conformation
        new_bias = self._bias
        self._bias = 1
        self._endpoints = initial_endpoints
        self._modified_domains = []
        self._assigned_domains = []
        self._old_configs = copy.deepcopy(self._prev_configs)

        # Regrow scaffold
        self._unassign_domains(scaffold_indices, staples_bound)
        self._grow_scaffold(scaffold_indices, regrow_old=True)

        # Regrow staples
        self._regrow_staples(staples_bound, regrow_old=True)

        # Test acceptance
        ratio = new_bias / self._bias
        if self._test_acceptance(ratio):
            self.reset_origami()
            accepted = True
        else:
            self._modified_domains = []
            self._assigned_domains = []
            accepted = False

        return accepted

    def _unassign_domains(self, scaffold_indices, staples):
        """Unassign scaffold and staple domains in selected region."""

        # Note do not count energies here as would double count
        # Unassign scaffold
        #delta_e = 0
        for domain_index in scaffold_indices[1:]:
            self._add_prev_config(SCAFFOLD_INDEX, domain_index)
            self._modified_domains.append((SCAFFOLD_INDEX, domain_index))
            self.origami_system.unassign_domain(SCAFFOLD_INDEX,
                    domain_index)

        # Unassign staples
        for staple_index in staples.keys():
            for domain_index in range(self.origami_system.chain_lengths[staple_index]):
                self._add_prev_config(staple_index, domain_index)
                self._modified_domains.append((staple_index, domain_index))
                self.origami_system.unassign_domain(staple_index,
                    domain_index)

        #self._bias *= math.exp(-delta_e / self.origami_system.temp)

    def _regrow_staples(self, staples, regrow_old=False):
        for staple_index, comp_domains in staples.items():

            # Pick domain on scaffold and staple to grow from
            staple_domain_i, scaffold_domain_i = random.choice(
                    comp_domains)
            delta_e = self._set_staple_growth_point(staple_index,
                    staple_domain_i, 0, scaffold_domain_i)
            self._bias *= math.exp(-delta_e / self.origami_system.temp)
            self._grow_staple(staple_index, staple_domain_i,
                    regrow_old=regrow_old)

    def _grow_scaffold(self, scaffold_indices, regrow_old=False):
        """Grow segment of scaffold with configurational bias."""
        self._calc_bias = self._calc_fixed_end_rosenbluth
        if regrow_old:
            self._select_config = self._select_old_config
        else:
            self._select_config = self._select_config_with_bias

        self._update_endpoints = self._update_scaffold_endpoint
        for i, domain_i in enumerate(scaffold_indices[1:]):
            prev_domain_i = scaffold_indices[i]
            self._select_position(SCAFFOLD_INDEX, domain_i, prev_domain_i)

            # Update overcounts
            if not regrow_old:
                self._calc_overcount(SCAFFOLD_INDEX, domain_i)

    def _update_scaffold_endpoint(self, *args):
        self._endpoints['steps'] -= 1


class ConservedTopologyCBMCMovetype(RegrowthCBMCMovetype):
    """CB constant topology scaffold/staple regrowth movetype."""

    def attempt_move(self):
        """Regrow scaffold and staples with fixed topology."""

        # Pick scaffold segment to regrow
        scaffold_indices = self._select_scaffold_indices()

        # Find and classify all bound staples
        staples = self._find_bound_staples_with_bound(scaffold_indices)
        staple_types = self._classify_staples(staples, scaffold_indices)
        initial_staples = copy.deepcopy(staples)
        initial_endpoints = copy.deepcopy(self._endpoints)
        initial_staple_types = copy.deepcopy(staple_types)

        # Unassign scaffold and non-externally bound staples
        self._unassign_domains(scaffold_indices, staples)

        # Regrow scaffold and staples
        self._grow_scaffold_and_staples(scaffold_indices, staples, staple_types)

        # Regrow in old conformation
        new_bias = self._bias
        self._bias = 1
        self._endpoints = initial_endpoints
        staples = initial_staples
        staple_types = initial_staple_types
        self._modified_domains = []
        self._assigned_domains = []
        self._old_configs = copy.deepcopy(self._prev_configs)

        # Unassign scaffold and non-externaly bound staples
        self._unassign_domains(scaffold_indices, staples)

        # Regrow scaffold and staples
        self._grow_scaffold_and_staples(scaffold_indices, staples, staple_types,
                regrow_old=True)

        # Test acceptance
        ratio = new_bias / self._bias
        if self._test_acceptance(ratio):
            self.reset_origami()
            accepted = True
        else:
            self._modified_domains = []
            self._assigned_domains = []
            accepted = False

        self.origami_system.check_all_constraints()
        return accepted

    def _classify_staples(self, staples, scaffold_indices):
        """Find and classify all staples bound to give indices.

        Returns a dictionary with single, multiple, and external keys to bound
        staples.
        """
        staple_types = {}

        # Both indexed by scaffold domain index, contains bound staple index
        # and domain
        staple_types['singly_bound'] = {}
        staple_types['multiply_bound'] = {}

        # Delete externally bound staples from staples dict
        staples_to_delete = []
        for staple_index, staple in staples.items():
            staple_length = self.origami_system.chain_lengths[staple_index]

            # Check unbound domains if bound to another part of scaffold
            for staple_domain_i in range(staple_length):

                # If bound to current scaffold segment, continue
                bcur = any([x[0] == staple_domain_i for x in staple]) == True
                if bcur:
                    continue

                domain = (staple_index, staple_domain_i)
                bound_domain = self.origami_system.get_bound_domain(*domain)

                # If unbound, continue
                if bound_domain == ():
                    continue

                # Since externally bound, add bound scaffold domains to endpoints
                for staple_domain_j, scaffold_domain_i in staple:

                    # Don't add endpoints for first domain:
                    if scaffold_domain_i == scaffold_indices[0]:
                        continue

                    self._endpoints['indices'].append(scaffold_domain_i)
                    position = self.origami_system.get_domain_position(
                            SCAFFOLD_INDEX, scaffold_domain_i)
                    self._endpoints['positions'].append(position)
                    Ni = scaffold_indices.index(scaffold_domain_i)
                    steps = np.concatenate([self._endpoints['steps'], [Ni]])
                    self._endpoints['steps'] = steps

                staples_to_delete.append(staple_index)
                break

            # If no externally bound domains, save possible growth points
            else:
                if len(staple) == 1:
                    staple_domain_i = staple[0][0]
                    scaffold_domain_i = staple[0][1]
                    staple_types['singly_bound'][scaffold_domain_i] = (
                            staple_index, staple_domain_i)
                else:
                    for pair in staple:
                        staple_domain_i = pair[0]
                        scaffold_domain_i = pair[1]
                        staple_types['multiply_bound'][scaffold_domain_i] = (
                                (staple_index, staple_domain_i))
        
        # Delete after as cannot during iteration over the dict
        for staple_index in staples_to_delete:
            del staples[staple_index]

        return staple_types

    def _unassign_domains(self, scaffold_indices, staples):
        """Unassign all give scaffold and non-externaly bound staple domains."""

        # Note do not count energies here because is double counting
        # Unassign scaffold domains
        #delta_e = 0
        for domain_index in scaffold_indices[1:]:
            self._add_prev_config(SCAFFOLD_INDEX, domain_index)
            self._modified_domains.append((SCAFFOLD_INDEX, domain_index))
            self.origami_system.unassign_domain(SCAFFOLD_INDEX,
                    domain_index)

        # Unassign staples
        for staple_index in staples.keys():
            for domain_index in range(self.origami_system.chain_lengths[staple_index]):
                self._add_prev_config(staple_index, domain_index)
                self._modified_domains.append((staple_index, domain_index))
                self.origami_system.unassign_domain(staple_index,
                    domain_index)

        #self._bias *= math.exp(-delta_e / self.origami_system.temp)

    def _grow_staple_and_update_endpoints(self, scaffold_domain_i, staple_types,
            staples, scaffold_indices, regrow_old=False):
        """Grow staples, update endpoints, and return modified staple_types."""

        # Grow singly bound staple if present
        if scaffold_domain_i in staple_types['singly_bound'].keys():
            staple_i, staple_domain_i = staple_types['singly_bound'][
                    scaffold_domain_i]
            delta_e = self._set_staple_growth_point(staple_i, staple_domain_i,
                    0, scaffold_domain_i)
            self._bias *= math.exp(-delta_e / self.origami_system.temp)
            self._grow_staple(staple_i, staple_domain_i, regrow_old=regrow_old,
                    overcount_cor=False)
            del staple_types['singly_bound'][scaffold_domain_i]

        # Grow multiply bound staple and add endpoints if present
        elif scaffold_domain_i in staple_types['multiply_bound'].keys():
            staple_i, staple_domain_i= staple_types['multiply_bound'][
                     scaffold_domain_i]
            delta_e = self._set_staple_growth_point(staple_i, staple_domain_i,
                    0, scaffold_domain_i)
            self._bias *= math.exp(-delta_e / self.origami_system.temp)
            self._grow_staple(staple_i, staple_domain_i, regrow_old=regrow_old,
                    overcount_cor=False)
            del staple_types['multiply_bound'][scaffold_domain_i]

            # Add remaining staple domains to endpoints
            staples[staple_i].remove((staple_domain_i, scaffold_domain_i))
            for staple_domain_j, scaffold_domain_j in staples[staple_i]:
                self._endpoints['indices'].append(scaffold_domain_j)
                position = self.origami_system.get_domain_position(
                        staple_i, staple_domain_j)
                self._endpoints['positions'].append(position)
                Ni = (scaffold_indices.index(scaffold_domain_j) -
                        scaffold_indices.index(scaffold_domain_i) - 1)
                steps = np.concatenate([self._endpoints['steps'], [Ni]])
                self._endpoints['steps'] = steps
                del staple_types['multiply_bound'][scaffold_domain_j]

        # Otherwise continue with scaffold
        else:
            pass

        return staple_types

    def _update_scaffold_endpoints(self, domain_i):
        try:
            endpoint_i = self._endpoints['indices'].index(domain_i)
        except ValueError:
            pass
        else:
            del self._endpoints['indices'][endpoint_i]
            del self._endpoints['positions'][endpoint_i]
            steps = np.delete(self._endpoints['steps'], endpoint_i)
            self._endpoints['steps'] = steps

        self._endpoints['steps'] -= 1

    def _grow_scaffold_and_staples(self, scaffold_indices, staples,
                staple_types, regrow_old=False):
        """Grow scaffold and staple chains."""
        self._calc_bias = self._calc_fixed_end_rosenbluth
        if regrow_old:
            self._select_config = self._select_old_config
        else:
            self._select_config = self._select_config_with_bias

        # Grow staples
        domain_i = scaffold_indices[0]
        self._grow_staple_and_update_endpoints(domain_i, staple_types,
                staples, scaffold_indices, regrow_old=regrow_old)
        for i, domain_i in enumerate(scaffold_indices[1:]):

            # Reset bias calc and endpoint update methods
            self._calc_bias = self._calc_fixed_end_rosenbluth
            self._update_endpoints = self._update_scaffold_endpoints

            prev_domain_i = scaffold_indices[i]
            self._select_position(SCAFFOLD_INDEX, domain_i, prev_domain_i)

            # Grow staples
            self._grow_staple_and_update_endpoints(domain_i, staple_types,
                    staples, scaffold_indices, regrow_old=regrow_old)

    def _calc_rosenbluth(self, weights, configs, *args):
        """calculate rosenbluth weight and return normalized weights."""

        # Set weights of complementary domains to 0
        for i, weight in enumerate(weights):

            # WARNING: This assumes all weights are either 6 for an unbound
            # state, otherwise it's a bound state
            if weight != 6:
                weights[i] = 0

        # deadend
        rosenbluth_i = sum(weights)
        if rosenbluth_i == 0:
            raise moverejection

        weights = (np.array(weights) / rosenbluth_i).tolist()
        self._bias *= rosenbluth_i
        return weights

    def _calc_fixed_end_rosenbluth(self, weights, configs, p_prev):
        """Return fixed endpoint weights."""
        
        # Set weights of complementary domains to 0
        for i, weight in enumerate(weights):

            # WARNING: This assumes all weights are either 6 for an unbound
            # state, otherwise it's a bound state
            if weight != 6:
                config = configs[i][0]
                endpoint_ps = self._endpoints['positions']
                if any((config == endpoint).all() for endpoint in endpoint_ps):
                    continue
                else:
                    weights[i] = 0

        # Bias weights with number of walks
        for i, config in enumerate(configs):
            start_point = config[0]
            num_walks = 1
            for endpoint_i in range(len(self._endpoints['indices'])):
                endpoint_p = self._endpoints['positions'][endpoint_i]
                endpoint_s = self._endpoints['steps'][endpoint_i]
                num_walks *= IDEAL_RANDOM_WALKS.num_walks(start_point,
                        endpoint_p, endpoint_s)

            weights[i] *= num_walks

        # Calculate number of walks for previous position
        num_walks = 1
        for endpoint_i in range(len(self._endpoints['indices'])):
            endpoint_p = self._endpoints['positions'][endpoint_i]
            endpoint_s = self._endpoints['steps'][endpoint_i] + 1
            num_walks *= IDEAL_RANDOM_WALKS.num_walks(p_prev,
                    endpoint_p, endpoint_s)

        # Modified Rosenbluth
        weights_sum = sum(weights)
        bias = weights_sum / num_walks
        self._bias *= bias
        if bias == 0:
            raise MoveRejection

        weights = (np.array(weights) / weights_sum).tolist()
        return weights
