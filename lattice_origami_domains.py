#!/usr/env python

"""Run MC simulations of the lattice domain-resolution DNA origami model."""

import json
import math
import sys
import os
import random
import pdb
import copy
from enum import Enum

import h5py
import numpy as np
import scipy.constants

# Scaffold domain index
SCAFFOLD_INDEX = 0

# Occupancy states
UNASSIGNED = 0
UNBOUND = 1
BOUND = 2

# Move outcomes
REJECTED = 0
ACCEPTED = 1

# Units vectors for euclidean space
XHAT = np.array([1, 0, 0])
YHAT = np.array([0, 1, 0])
ZHAT = np.array([0, 0, 1])

# Boltzmann constant in J/K
KB = scipy.constants.k

# Avogadro's number
AN = scipy.constants.Avogadro

# Molar gas constant
R = scipy.constants.gas_constant

# J/cal
J_PER_CAL = 4.184

# santalucia2004; kcal/mol
NN_ENTHALPY = {
    'AA/TT': -7.6,
    'AT/TA': -7.2,
    'TA/AT': -7.2,
    'CA/GT': -8.5,
    'GT/CA': -8.4,
    'CT/GA': -7.8,
    'GA/CT': -8.2,
    'CG/GC': -10.6,
    'GC/CG': -9.8,
    'GG/CC': -8.0,
    'INITIATION': 0.2,
    'TERMINAL_AT_PENALTY': 2.2,
    'SYMMETRY_CORRECTION': 0}

# kcal/mol/K
NN_ENTROPY = {
    'AA/TT': -0.0213,
    'AT/TA': -0.0204,
    'TA/AT': -0.0213,
    'CA/GT': -0.0227,
    'GT/CA': -0.0224,
    'CT/GA': -0.0210,
    'GA/CT': -0.0222,
    'CG/GC': -0.0272,
    'GC/CG': -0.0244,
    'GG/CC': -0.0199,
    'INITIATION': -0.0057,
    'TERMINAL_AT_PENALTY': 0.0069,
    'SYMMETRY_CORRECTION': -0.0014}

COMPLIMENTARY_BASE_PAIRS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


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


def rotate_vector_half(vector, rotation_axis):
    vector = np.copy(vector)
    if all(np.abs(rotation_axis == XHAT)):
        y = -vector[1]
        z = -vector[2]

    if all(np.abs(rotation_axis == YHAT)):
        x = -vector[0]
        z = -vector[2]

    if all(np.abs(rotation_axis == ZHAT)):
        x = -vector[0]
        y = -vector[1]

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


def calc_hybridization_energy(sequence, T, cation_M):
    """Calculate hybridization energy of domains with NN model.

    OUtputs energies in K (avoid multiplying by KB when calculating acceptances.
    Sequences are assumed to be 5' to 3'.

    cation_M -- Total cation molarity.
    """
    complimentary_sequence = calc_complimentary_sequence(sequence)

    # Initiation free energy
    DH_init = NN_ENTHALPY['INITIATION']
    DS_init = NN_ENTROPY['INITIATION']

    # Symmetry penalty for palindromic sequences
    if sequence_is_palindromic(sequence):
        DS_sym = NN_ENTROPY['SYMMETRY_CORRECTION']
    else:
        DS_sym = 0

    # NN pair energies
    DH_stack = 0
    DS_stack = 0
    for base_index in range(0, len(sequence), 2):
        first_pair = sequence[base_index : base_index + 2]
        second_pair = complimentary_sequence[base_index : base_index + 2]
        key = first_pair + '/' + second_pair

        # Not all permutations are included in dict as some reversals have
        # identical energies
        try:
            DH_stack += NN_ENTHALPY[key]
            DS_stack += NN_ENTROPY[key]
        except KeyError:
            key = key[::-1]
            DH_stack += NN_ENTHALPY[key]
            DS_stack += NN_ENTROPY[key]

    # Terminal AT penalties
    terminal_AT_pairs = 0
    for sequence_index in [0, -1]:
        if sequence[sequence_index] in ['A', 'T']:
            terminal_AT_pairs += 1

    if terminal_AT_pairs > 0:
        DH_at = NN_ENTHALPY['TERMINAL_AT_PENALTY'] * terminal_AT_pairs
        DS_at = NN_ENTROPY['TERMINAL_AT_PENALTY'] * terminal_AT_pairs
    else:
        DH_at = 0
        DS_at = 0

    DH_hybrid = DH_init + DH_stack + DH_at
    DS_hybrid = DS_init + DS_sym + DS_stack + DS_at

    # Apply salt correction
    DS_hybrid = DS_hybrid + (0.368 * (len(sequence) / 2) * math.log(cation_M))/1000

    DG_hybrid = DH_hybrid - T * DS_hybrid

    # Convert from kcal/mol to K (so avoid KB later)
    DG_hybrid = DG_hybrid * J_PER_CAL * 1000 / R

    return DG_hybrid


def calc_complimentary_sequence(sequence):
    """Return the complimentary DNA sequence."""
    complimentary_seq_list = []
    for base in sequence:
        complimentary_seq_list.append(COMPLIMENTARY_BASE_PAIRS[base])

    complimentary_sequence = ''.join(complimentary_seq_list)
    return complimentary_sequence


def sequence_is_palindromic(sequence):
    """True if reverse complimenet is equal to given sequence."""
    complimentary_sequence = calc_complimentary_sequence(sequence)
    reverse_complimentary_sequence = complimentary_sequence[::-1]
    palindromic = reverse_complimentary_sequence == sequence
    return palindromic


class IdealRandomWalks:

    def __init__():
        # Number of random walks indexed by tuple of end - start
        self._num_walks = {}

    def num_walks(start, end, N):
        """Calculate number of walks between two positions."""

        # Check stored values
        walk = tuple(end - start)
        try:
            num_walks = self._num_walks[walk]
        except KeyError:
            pass
        else:
            return num_walks

        # Calculate num walks (see dijkstra1994)
        DX = walk[0]
        DY = walk[1]
        DZ = walk[2]
        Nminus = (N - DX - DY - DZ) // 2
        Nplus = (N - DX - DY + DZ) // 2
        for ybar in range(0, Nminus + 1):
            for xbar in range(0, Nminus + 1 - ybar):
                f1 = math.factorial(N)
                f2 = math.factorial(xbar)
                f3 = math.factorial(xbar + DX)
                f4 = math.factorial(ybar)
                f5 = math.factorial(ybar + DY)
                f6 = math.factorial(Nminus - xbar - ybar)
                f7 = math.factorial(Nplus - xbar - ybar)
                num_walks = f1 // (f2 * f3 * f4 * f5 * f6 * f7)

        return num_walks


class OrigamiSystem:
    """Simple cubic lattice model of DNA origami at domain level resolution.

    See reports/modelSpecs/domainResModelSpecs for exposition on the model.

    I've used get methods instead of properties as properties don't take
    indices. I would have to index the internal structure directly, which
    I don't want to do.
    """

    def __init__(self, input_file, step, temp, cation_M):
        self.temp = temp

        # Scaffold cyclic
        cyclic = input_file.cyclic

        # Domain identities of each chain
        self.identities = input_file.identities

        # Scaffold strand indices of complimentary domains to staple domains,
        # indexed by staple identity
        self.staple_to_scaffold_domains = {}
        for staple_indentity, staple in enumerate(identities[1:]):
            scaffold_domain_indices = []
            for staple_domain_indent in staple:
                scaffold_domain_i = self.identities.index(-staple_domain_ident)
                scaffold_domain_indices.append(scaffold_domain_i)
            self.staple_to_scaffold_domains[staple_index] = 
            
        # Domain sequences
        self.sequences = input_file.sequences
        
        # Check all sequences are the same length
        seq_l = len(self.sequences[0])
        for sequence in self.sequences:
            if seq_l != len(sequence):
                print('Domain sequences not all equal length.')
                sys.exit()

        # Calculate and store hybridization energies
        self._hybridization_energies = []
        for sequence in self.sequences:
            energy = calc_hybridization_energy(sequence, temp, cation_M)
            self._hybridization_energies.append(energy)

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

        # Dictionary with position keys and state values
        self._position_occupancies = {}

        # Dictionary with domain keys and state values
        self._domain_occupancies = {}

        # Dictionary with bound domain keys and values
        self._bound_domains = {}

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
            self.chain_lengths.append(num_domains)
            previous_position = np.array(chain['positions'][0])
            for domain_index in range(num_domains):
                position = np.array(chain['positions'][domain_index])
                orientation = np.array(chain['orientations'][domain_index])

                # Check domains are within one
                if ((position - previous_position)**2).sum() > 1:
                    raise ConstraintViolation

                self.set_domain_configuration(chain_index, domain_index,
                        position, orientation)

                previous_position = np.array(chain['positions'][domain_index])

        # Check first and last domains of cyclic scaffold within 1
        p_first = self_.positions[0][0]
        p_last = self_.positions[0][-1]
        if ((p_first - p_last)**2).sum() > 1:
            raise ConstraintViolation

        # Keep track of unique chain index
        self._current_chain_index = max(self._working_to_unique)

        # Bookeeping for configuration bias
        self._checked_list = []
        self._current_domain = ()

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

    def get_num_staples(self, identity):
        """Return number of staples in system of given identity."""
        Ni = len(self._identity_to_index[identity])
        return Ni

    def get_complimentary_scaffold_domains(self, staple_i):
        """Return list of scaffold domains complimentary to given staple."""
        staple_identity = self._chain_identities[staple_i]
        return self.staple_to_scaffold[staple_identity]

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

    def get_hybridization_energy(self, chain_index, domain_index):
        """Return precalculated hybridization energy."""

        chain_identity = self._chain_identities[chain_index]
        domain_identity = self.identities[chain_identity][domain_index]

        # Because identites start at 1
        energy_index = abs(domain_identity) - 1
        return self._hybridization_energies[energy_index]

    def check_domain_configuration(self, chain_index, domain_index, position,
                orientation):
        """Check if constraints are obeyed and return energy change."""
        domain = (chain_index, domain_index)
        delta_e = 0

        # If checked list empty, set current domain identity
        if self._checked_list == []:
            self._current_domain = domain
        else:

        # Check if given domain identity matches current
            if domain != self._current_domain:
                raise OrigamiMisuse
            else:
                pass

        # Constraint violation if position in bound state
        occupancy = self.get_position_occupancy(position)
        if occupancy == BOUND:

            raise ConstraintViolation
        else:
            pass

        # Save current positions and set to trial
        cur_r = self._positions[chain_index][domain_index]
        self._positions[chain_index][domain_index] = position
        cur_o = self._orientations[chain_index][domain_index]
        self._orientations[chain_index][domain_index] = orientation

        # Attempt binding if position occupied in unbound state
        if occupancy == UNBOUND:
            try:
                delta_e = self._bind_domain(*domain)
            except ConstraintViolation:

                # Revert to current position
                self._positions[chain_index][domain_index] = cur_r
                self._orientations[chain_index][domain_index] = cur_o
                raise
        else:
            pass

        # Revert to current position
        self._positions[chain_index][domain_index] = cur_r
        self._orientations[chain_index][domain_index] = cur_o

        # Update checked list
        self._checked_list.append(tuple(position))
        return delta_e

    def set_checked_domain_configuration(self, chain_index, domain_index,
                position, orientation):
        """Set domain to previously checked configuration."""
        domain = (chain_index, domain_index)

        # Check if give domain identity matches current
        if domain != self._current_domain:
            raise OrigamiMisuse
        else:
            pass

        if tuple(position) in self._checked_list:

            # Set domain configuration without further checks
            self._positions[chain_index][domain_index] = position
            self._orientations[chain_index][domain_index] = orientation
            occupancy = self.get_position_occupancy(position)
            unique_index = self._working_to_unique[chain_index]
            domain_key = (unique_index, domain_index)
            if occupancy == UNBOUND:
                self._update_occupancies_bound(position, domain_key)
            else:
                self._update_occupancies_unbound(position, domain_key)
        else:
            raise OrigamiMisuse

        self._checked_list = []
        self._current_domain = ()

    def set_domain_configuration(self, chain_index, domain_index, position,
                orientation):
        """Set domain configuration and return change in energy.

        Assumes that given domain has already been unassigned. Consider adding a
        check.

        Will raise exception ConstraintViolation if constraints violated.
        """
        domain = (chain_index, domain_index)
        unique_index = self._working_to_unique[chain_index]
        domain_key = (unique_index, domain_index)
        delta_e = 0

        # Constraint violation if position in bound state
        occupancy = self.get_position_occupancy(position)
        if occupancy == BOUND:

            raise ConstraintViolation
        else:
            pass

        # Save current positions and set to trial
        cur_r = self._positions[chain_index][domain_index]
        self._positions[chain_index][domain_index] = position
        cur_o = self._orientations[chain_index][domain_index]
        self._orientations[chain_index][domain_index] = orientation

        # Attempt binding if position occupied in unbound state
        if occupancy == UNBOUND:
            try:
                delta_e = self._bind_domain(*domain)
            except ConstraintViolation:

                # Revert to current position
                self._positions[chain_index][domain_index] = cur_r
                self._orientations[chain_index][domain_index] = cur_o
                raise
            else:
                self._update_occupancies_bound(position, domain_key)

        # Move to empty site and update occupancies
        else:
            self._update_occupancies_unbound(position, domain_key)

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

            # Collect energy
            delta_e = -self.get_hybridization_energy(*domain)
            bound_domain = self._bound_domains[domain_key]
            position = tuple(self._positions[chain_index][domain_index])
            self._positions[chain_index][domain_index] = []
            self._orientations[chain_index][domain_index] = []
            del self._bound_domains[domain_key]
            del self._bound_domains[bound_domain]
            del self._domain_occupancies[domain_key]
            self._unbound_domains[position] = bound_domain
            self._position_occupancies[position] = UNBOUND
            self._domain_occupancies[bound_domain] = UNBOUND
        elif occupancy == UNBOUND:
            position = tuple(self._positions[chain_index][domain_index])
            self._positions[chain_index][domain_index] = []
            self._orientations[chain_index][domain_index] = []
            del self._unbound_domains[position]
            del self._position_occupancies[position]
            del self._domain_occupancies[domain_key]
        else:
            pass

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
        self.chain_lengths.append(chain_length)
        return chain_index

    def delete_chain(self, chain_index):
        """Delete chain."""

        # Change in energy
        delta_e = 0
        for domain_index in range(self.chain_lengths[chain_index]):
            delta_e += self.unassign_domain(chain_index, domain_index)

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
        del self.chain_lengths[chain_index]

        return delta_e

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
                elif occupancy == UNBOUND:
                    position_occupancies[r_n] = UNBOUND
                    unbound_domains[r_n] = domain

        self._position_occupancies = position_occupancies
        self._unbound_domains = unbound_domains

    def _update_occupancies_bound(self, position, domain_key):
        position = tuple(position)
        occupying_domain = self._unbound_domains[position]
        del self._unbound_domains[position]
        self._domain_occupancies[domain_key] = BOUND
        self._domain_occupancies[occupying_domain] = BOUND
        self._position_occupancies[position] = BOUND
        self._bound_domains[occupying_domain] = domain_key
        self._bound_domains[domain_key] = occupying_domain

    def _update_occupancies_unbound(self, position, domain_key):
        position = tuple(position)
        self._domain_occupancies[domain_key] = UNBOUND
        self._position_occupancies[position] = UNBOUND
        self._unbound_domains[position] = domain_key

    def _bind_domain(self, trial_chain_index, trial_domain_index):
        """Bind given domain in preset trial config and return change in energy.
        """
        position = tuple(self._positions[trial_chain_index][trial_domain_index])

        # Test if complimentary (and has correct orientation for binding)
        try:
            occupying_domain = self.get_unbound_domain(position)
        except KeyError:

            # This would only happen if the caller didn't check the state of
            # position first.
            raise

        # Convenience variable
        trial_domain = (trial_chain_index, trial_domain_index)

        complimentary = self._domains_match(*trial_domain, *occupying_domain)
        if not complimentary:
            raise ConstraintViolation
        else:
            pass

        # Check constraints between domains and neighbours
        for chain_index, domain_index in [trial_domain, occupying_domain]:
            for direction in [-1, 1]:
                adjacent_domain_i = domain_index + direction
                neighbour_domain = (chain_index, adjacent_domain_i)
                try:
                    occupancy = self.get_domain_occupancy(*neighbour_domain)
                except KeyError:
                    if self.cyclic:
                        scaffold_length = self.chain_lengths[0]
                        if adjacent_domain_i == scaffold_length:
                            adjacent_domain_i = 0
                        elif adjacent_domain_i == -1:
                            adjacent_domain_i = scaffold_length - 1

                        neighbour_domain = (chain_index, adjacent_domain_i)
                        occupancy = self.get_domain_occupancy(*neighbour_domain)
                    else:
                        continue

                if occupancy == BOUND:
                    if self._helical_pair_constraints_obeyed(*neighbour_domain,
                            domain_index):
                        pass
                    else:
                        raise ConstraintViolation
                else:
                    pass

        # Add new binding energies
        delta_e = self.get_hybridization_energy(*trial_domain)

        return delta_e

    def _domains_match(self, chain_index_1, domain_index_1,
                chain_index_2, domain_index_2):
        """Return True if domains have correct orientation and sequence."""

        # Determine domain identities
        chain_identity_1 = self._chain_identities[chain_index_1]
        domain_identity_1 = self.identities[chain_identity_1][domain_index_1]
        chain_identity_2 = self._chain_identities[chain_index_2]
        domain_identity_2 = self.identities[chain_identity_2][domain_index_2]

        # Complimentary if domain identities sum to 0
        complimentary = domain_identity_1 + domain_identity_2

        # Check if orientations are correct
        if complimentary == 0:
            orientation_1 = self._orientations[chain_index_1][domain_index_1]
            orientation_2 = self._orientations[chain_index_2][domain_index_2]

            # They should be opposite vectors, thus correct if sum to 0
            complimentary_orientations = orientation_1 + orientation_2
            if all(complimentary_orientations == np.zeros(3)):
                match = True
            else:
                match = False
        else:
            match = False

        return match

    def _helical_pair_constraints_obeyed(self, chain_index, domain_index_1,
            domain_index_2):
        """Return True if domains not in same helix or twist constraints obeyed.

        Does not check if they are in bound states.
        """

        # Set domain_index_1 to be 5' domain if scaffold and 3' domain if staple
        if domain_index_2 > domain_index_1:
            pass
        else:
            domain_index_1, domain_index_2 = domain_index_2, domain_index_1

        orientation_1 = (self._orientations[chain_index][domain_index_1])
        orientation_2 = (self._orientations[chain_index][domain_index_2])

        # Calculate next domain vectors
        position_1 = self._positions[chain_index][domain_index_1]
        position_2 = self._positions[chain_index][domain_index_2]
        next_dr_1 = position_2 - position_1
        if domain_index_2 == self.chain_lengths[chain_index] - 1:
            next_dr_2 = np.zeros(3)
        else:
            position_3 = self._positions[chain_index][domain_index_2 + 1]
            next_dr_2 = position_3 - position_2

        # Only one allowed configuration not in the same helix
        if all(next_dr_1 == orientation_1):
            constraints_obeyed = True

        # If next domain is not new helix, must be linear
        elif all(next_dr_2 != orientation_2) and not (
                all(next_dr_2 == np.zeros(3)) or all(next_dr_1 == next_dr_2)):
            constraints_obeyed = False

        # Check twist constraint if same helix
        else:
            constraints_obeyed = self._check_twist_constraint(
                    next_dr_1, orientation_1, orientation_2)

        return constraints_obeyed


class OrigamiSystemEight(OrigamiSystem):
    """Origami systems with 8 bp domains."""

    def _check_twist_constraint(self, next_dr, orientation_1, orientation_2):
        orientation_1_r = (rotate_vector_quarter(orientation_1, next_dr, -1))
        if all(orientation_1_r == orientation_2):
            constraints_obeyed = True
        else:
            constraints_obeyed = False

        return constraints_obeyed


class OrigamiSystemSixteen(OrigamiSystem):
    """Origami systems with 16 bp domains."""

    def _check_twist_constraint(self, next_dr, orientation_1, orientation_2):
        orientation_1_r = (rotate_vector_half(orientation_1, next_dr))
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
            self._write_configuration(origami_system, step)
        else:
            pass
        if value_is_multiple(step, self._count_write_freq):
            self._write_staple_domain_count(origami_system)

        # Move types and acceptances


class JSONOutputFile(OutputFile):
    """JSON output file class."""

    def __init__(self, filename, origami_system, config_write_freq=1):
        json_origami = {'origami':{'identities':{}, 'configurations':[]}}
        json_origami['origami']['identities'] = origami_system.identities
        json_origami['origami']['sequences'] = origami_system.sequences
        json_origami['origami']['cyclic'] = origami_system.cyclic

        self._filename = filename
        self._config_write_freq = config_write_freq
        self.json_origami = json_origami

    def write_seed(self, seed):
        json_origami['origami']['seed'] = seed

    def _write_configuration(self, origami_system, step):
        self.json_origami['origami']['configurations'].append({})
        current_config = self.json_origami['origami']['configurations'][-1]
        current_config['step'] = step
        current_config['chains'] = []
        chain_index = -1
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
        self._json_origami['origami']['cyclic']

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


class HDF5OutputFile(OutputFile):
    """HDF5 output file class.

    Custom format; not compatable with VMD (not H5MD).
    """

    def __init__(self, filename, origami_system, config_write_freq=1,
                count_write_freq=0):
        self.hdf5_origami = h5py.File(filename, 'w')
        self.hdf5_origami.create_group('origami')
        self.filename = filename
        self._config_write_freq = config_write_freq
        self._config_writes = 0
        self._count_write_freq = count_write_freq
        self._count_writes = 0

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

        # Fill attributes
        self.hdf5_origami.attrs['identities'] = filled_identities
        self.hdf5_origami.attrs['cyclicl'] = origami_system.cyclic
        self.hdf5_origami.attrs['temp'] = origami_system.temp
        self.hdf5_origami.attrs['config_write_freq'] = config_write_freq
        self.hdf5_origami.attrs['count_write_freq'] = count_write_freq

        # HDF5 does not allow lists of strings
        sequences = np.array(origami_system.sequences, dtype='a')
        self.hdf5_origami.attrs['sequences'] = sequences

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

    def _write_staple_domain_count(self, origami_system):
        write_index = self._count_writes
        self._count_writes += 1
        num_staples = len(origami_system.chain_lengths) - 1
        num_domains = len(origami_system._bound_domains) // 2
        count_key = 'origami/staple_domain_count'
        self.hdf5_origami[count_key].resize(self._count_writes, axis=0)
        self.hdf5_origami[count_key][write_index] = (num_staples, num_domains)

    def _write_configuration(self, origami_system, step):
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

    def close(self):
        """Perform any cleanup."""
        self.hdf5_origami.close()


class HDF5InputFile:
    """Input file taking hdf5 formatted origami system files in constructor."""

    def __init__(self, filename):
        hdf5_origami = h5py.File(filename, 'r')

        self._filename = filename
        self._hdf5_origami = hdf5_origami

        # Try for backwards compatibility
        try:
            self.config_write_freq = hdf5_origami.attrs['config_write_freq']
            self.count_write_freq = hdf5_origami.attrs['config_write_freq']
        except KeyError:
            pass

    @property
    def cyclic(self):
        self._hdf5_origami.attrs['cyclic']

    @property
    def identities(self):
        """Standard format for passing origami domain identities."""

        # HDF5 does not allow variable length lists; fill with 0
        raw =  self._hdf5_origami.attrs['identities'].tolist()
        identities = []
        for raw_domain_identities in raw:
            identities.append([i for i in raw_domain_identities if i !=0])

        return identities

    @property
    def sequences(self):
        """Standard format for passing origami system sequences.

        Only includes sequences of scaffold in 5' to 3' orientation.
        """

        # H5py outputs as type 'S', need type 'U'
        return self._hdf5_origami.attrs['sequences'].astype('U').tolist()

    @property
    def temp(self):
        return self._hdf5_origami.attrs['temp']

    @property
    def steps(self):
        return len(self._hdf5_origami['origami/chain_ids'])

    @property
    def staple_domain_counts(self):
        return self._hdf5_origami['origami/staple_domain_count']

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


class GCMCSimulation:
    """GCMC sim for domain-res origami model with bound staples only.

    Grand cannonical Monte Carlo simulations on a simple cubic lattice of the
    origami model defined by the origami class and associated documentation. The
    simulations run with this class do not include any moves that can lead to
    free staples.
    """

    def __init__(self, origami_system, move_settings, output_file):
        self._origami_system = origam_system
        self._output_file = output_file

        # Set seed for python's Mersenne Twister (64 bits from os)
        seed = os.urandom(8)
        random.seed(seed)
        output_file.write_seed(seed)

        # Frequency for translating system back to origin
        self.center_freq = 1

        # Current movetype
        self._movetype = -1

        # Create cumalative probability distribution for movetypes
        # List to associate movetype method with index in distribution
        self._movetype_methods = []
        self._movetypes = []
        self._movetype_probabilities = []
        cumulative_probability = 0
        for movetype, probability in move_settings.items():

            # I still wonder if there is a way to avoid all the if statements
            if movetype == MOVETYPE.EXCHANGE_STAPLE:
                movetype_class = ExchangeMMCMovetype
            elif movetype == MOVETYPE.REGROW_STAPLE:
                movetype_class = StapleRegrowthMMCMovetype
            elif movetype == MOVETYPE.REGROW_SCAFFOLD:
                movetype_class = ScaffoldRegrowthMMCMovetype
            elif movetype == MOVETYPE.ROTATE_ORIENTATION_VECTOR:
                movetype_class = OrientationRotationMCMovetype
            if movetype == MOVETYPE.CB_EXCHANGE_STAPLE:
                movetype_class = ExchangeMMCMovetype
            elif movetype == MOVETYPE.CB_REGROW_STAPLE:
                movetype_class = StapleRegrowthMMCMovetype
            elif movetype == MOVETYPE.CB_REGROW_SCAFFOLD:
                movetype_class = ScaffoldRegrowthMMCMovetype
            elif movetype == MOVETYPE.CB_CONSERVED_TOPOLOGY:
                movetype_class = ConservedTopologyCBMCMovetype

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

    def run(self, num_steps, logging=True):
        """Run simulation for given number of steps."""

        for step in range(1, num_steps + 1):
            movetype_object = self._select_movetype()
            outcome = movetype_object.attempt_move()
            self._origami_system = movetype_object.accepted_system
            if value_is_multiple(step, self.center_freq):
                self._origami_system.center()

            self._output_file.check_and_write(self._origami_system, step)

            # Loggging hack
            if logging:
                print(step, self._movetype, outcome,
                        len(self._origami_system.chain_lengths))
            else:
                pass

    def _select_movetype(self):
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

        movetype_object = movetype_class(self._origami_system)
        return movetype_object


class MCMovetype:
    """Base class for all movetype classes."""

    def __init__(self, origami_system):
        self.accepted_system = origami_system
        self.trial_system = copy.deepcopy(origami_system)

    def attempt_move(self):
        raise NotImplementedError

    def _test_acceptance(self, ratio):
        """Metropolis acceptance test for given ratio."""

        ratio = 1
        p_accept = min(1, ratio)
        if p_accept == 1:
            accept = True
        else:
            if p_accept > random.random():
                accept = True
            else:
                accept = False

        return accept

    def _find_bound_staples_with_compliments(self, domain_indices):
        """Find all staples bound to scaffold segment (includes repeats).

        Returns dictionary of staple indices and complimentary scaffold domains
        in provided scaffold indices, with corresponding staple domain index.
        """
        staples = {}
        for domain_index in scaffold_indices:

            # Check if scaffold domain bound
            staple_domain = self.accepted_system.get_bound_domain(
                    SCAFFOLD_INDEX, domain_index)
            if staple_domain == ():
                continue
            else:

                # Find remaining complimentary domains to bound staple
                staple_index = staple_domain[0]
                staple_identity = self.accepted_system._chain_indentities[
                        staple_index]
                staple_domain_i = staple_domain[1]
                if staple_index in staples:
                    continue
                else:
                    staples[staple_index] = []
                    scaffold_domains = self.trial_system.staple_to_scaffold
                    for staple_i, scaffold_i in scaffold_domains:
                        if scaffold_i in domain_indices:
                            staples[staple_index].append(staple_i, scaffold_i)

        return staples

    def _set_staple_growth_point(self, staple_index, staple_domain_index,
                scaffold_domain_index):
        """Given scaffold and staple, attempt to bind with correct orientation.
        """
        position = self.trial_system.get_domain_position(SCAFFOLD_INDEX,
                scaffold_domain_index)
        orientation = self.trial_system.get_domain_orientation(
                SCAFFOLD_INDEX, scaffod_domain_index)

        # Attempt to set growth domain
        try:
            delta_e = self.trial_system.set_domain_configuration(
                    staple_index, staple_domain_index, position, o_new)
        except ConstraintViolation:
            raise MoveRejection

        return delta_e

    def _grow_staple(self, staple_length, staple_index, domain_index, *args,
            **kwargs):
        """Randomly grow staple out from given domain in both directions."""

        # Grow in three-prime direction
        staple_indices = range(domain_index, staple_length)
        try:
            self._grow_chain(staple_index, staple_indices, *args, **kwargs)
        except MoveRejection:
            raise

        # Grow in five-prime direction
        staple_indices = range(domain_index, -1, -1)
        try:
            self._grow_chain(staple_index, staple_indices, *args, **kwargs)
        except MoveRejection:
            raise

    def _select_random_orientation(self):
        """Select a random orientation."""
        dimensions = [XHAT, YHAT, ZHAT]
        directions = [-1, 1]
        positions = [i * j for i in dimensions for j in directions]
        return random.choice(positions)

    def _select_random_position(self, p_prev):
        """Select a random position."""
        dimensions = [XHAT, YHAT, ZHAT]
        directions = [-1, 1]
        positions = [i * j for i in dimensions for j in directions]
        return p_prev + random.choice(positions)


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
        chain_lengths = self.accepted_system.chain_lengths
        chain_index = random.randrange(len(chain_lengths))
        domain_index = random.randrange(chain_lengths[chain_index])

        # Reject if in bound state
        occupancy = self.accepted_system.get_domain_occupancy(chain_index,
                domain_index)
        if occupancy == BOUND:
            accepted = False
            return accepted
        else:
            pass

        # Select random orientation and update (always accepted)
        o_new = self._select_random_orientation()
        self.accepted_system.set_domain_orientation(chain_index, domain_index,
                o_new)

        accepted = True
        return accepted


class MMCMovetype(Movetype):
    """Base class for Metropolis MC movetypes."""

    def __init__(self, *args, **kwargs):
        self._delta_e = 0
        super().__init__(self, *args, **kwargs)

    def attempt_move(self):
        raise NotImplementedError

    def _grow_chain(self, chain_index, domain_indices):
        """Randomly grow out chain from given domain indices.

        Updates changes in energy as binding events occur."""

        # Iterate through given indices, growing next domain from current index
        for i, domain_index in enumerate(domain_indices[:-1]):
            new_domain_index = domain_indices[i + 1]

            # Position vector of previous domain
            p_prev = self.trial_system.get_domain_position(chain_index, domain_index)

            # Randomly select neighbour lattice site for new position
            p_new = self._select_random_position(p_prev)

            # Randomly select new orientation
            o_new = self._select_random_orientation

            # Attempt to set position
            try:
                self._delta_e += self.trial_system.set_domain_configuration(
                        chain_index, new_domain_index, p_new, o_new)
            except ConstraintViolation:
                raise MoveRejection

        return


class ExchangeMMCMovetype(MMCMovetype):
    """Simple staple exchange movetype with Metropolis acceptance."""

    def attempt_move(self):
        if random.random() < 0.5:
            accepted = self._delete_staple()
        else:
            accepted = self._insert_staple()

        return accepted

    def _staple_insertion_accepted(self, identity, overcounts):
        """Metropolis acceptance test for particle insertion."""
        T = self.accepted_system.temp
        boltz_factor = math.exp(-self._delta_e / T)
        Ni = self.accepted_system.get_num_staples(identity)
        ratio = boltz_factor / (Ni + 1)

        # Correct for overcounts (can't call normal method)
        p_accept = min(1, ratio) / overcounts
        if p_accept == 1:
            accept = True
        else:
            if p_accept > random.random():
                accept = True
            else:
                accept = False

        return accept

    def _staple_deletion_accepted(self, identity):
        """Metropolis acceptance test for particle deletion."""
        T = self.accepted_system.temp
        boltz_factor = math.exp(-self._delta_e / T)
        Ni = self.accepted_system.get_num_staples(identity)
        return self._test_acceptance(Ni * boltz_factor)

    def _insert_staple(self):
        """Insert staple at random scaffold domain and grow."""

        # Randomly select staple identity and add chain
        staple_identity, domain_identities = (
                self.accepted_system.get_random_staple_identity())

        staple_index = self.trial_system.add_chain(staple_identity)
        staple_length = self.trial_system.chain_lengths[staple_index]

        # Select staple domain
        staple_domain = random.randrange(staple_length)
        domain_identity = domain_identities[staple_domain]

        # Select complimentary scaffold domain
        scaffold_domain = self.trial_system.identities.index(-domain_identity)
        
        # Number of bound domains in system (for calculating overcounts)
        init_num_bound_domains = self.trial_system.num_bound_domains

        # Set growth point domain
        try:
            self._delta_e += self._set_staple_growth_point(staple_index,
                    staple_domain, scaffold_domain):
        except MoveRejection:
            accepted = False
            return accepted

        # Grow staple
        try:
            self._grow_staple(staple_length, staple_index, staple_domain)
        except MoveRejection:
            accepted = False
            return accepted

        # If the configuration is such that the staple can bind with the other
        # domain, then there are two ways this can happen, so the ratio should
        # be halved to prevent overcounting. If staple ends in multiply bound
        # state, save resulting overcounts.
        cur_num_bound_domains = self.trial_system.num_bound_domains
        D_bind_state = init_num_bound_domains - cur_num_bound_domains
        overcounts = D_bind_state

        # Test acceptance
        if self._staple_insertion_accepted(staple_identity, overcounts):
            self.accepted_system = self._trial_system
            accepted = True
        else:
            accepted = False

        return accepted

    def _delete_staple(self):
        """Delete random staple."""

        # Randomly select staple identity
        staple_identity, domain_identities = (
                self.accepted_system.get_random_staple_identity())

        # Randomly select staple
        try:
            staple_index = self.trial_system.get_random_staple_of_identity(
                    staple_identity)

        # No staples in system
        except IndexError:
            accepted = False
            return accepted

        # Delete staple
        self._delta_e += self.trial_system.delete_chain(staple_index)

        # Test acceptance
        if self._staple_deletion_accepted(staple_identity):
            self.accepted_system = self._trial_system
            accepted = True
        else:
            accepted = False

        return accepted


class RegrowthMMCMovetype(MMCMovetype):
    """Base class for simple Metropolis conformational change movetypes."""

    def attempt_move(self):
        raise NotImplementedError

    def _conformation_accepted(self):
        """Metropolis acceptance test for configuration change."""
        T = self.accepted_system.temp
        boltz_factor = math.exp(-self._delta_e / T)
        return self._test_acceptance(boltz_factor)


class StapleRegrowthMMCMovetype(RegrowthMMCMovetype):
    """Simple Metropolis staple regrowth movetype."""

    def attempt_move(self):
        """Regrow random staple from randomly chosen complimentary domain."""

        # Randomly select staple
        try:
            staple_index = random.randrange(1,
                    len(self.trial_system.chain_lengths))

        # No staples in system
        except ValueError:
            accepted = False
            return accepted

        # Find all complimentary domains and randomly select growth point
        comp_domains = self.trial_system.get_complimentary_domains(staple_index)

        staple_length = self.accepted_system.chain_lengths[staple_index]
        staple_domain_i = random.randrange(staple_length)
        scaffold_domain_i = comp_domains[staple_domain_i]

        # Unassign domains
        for domain_index in range(staple_length):
            self._delta_e += self.trial_system.unassign_domain(staple_index,
                    domain_index)

        # Grow staple
        try:
            self._grow_staple(staple_length, staple_index, bound_domain_index)
        except MoveRejection:
            accepted = False
            return accepted
        
        # Test acceptance
        if self._configuration_accepted():
            self.accepted_system = self._trial_system
            accepted = True
        else:
            accepted = False

        return accepted


class ScaffoldRegrowthMMCMovetype(RegrowthMMCMovetype):
    """Simple Metropolis scaffold and bound staple regrowth movetype."""

    def attempt_move(self):
        """Randomly regrow terminal section of scaffold and bound staples.

        From a randomly selected scaffold domain in a random direction to the
        end.
        """

        # Pick section of scaffold to regrow
        scaffold_indices = self._select_scaffold_indices()

        # Find bound staples and all complimentary domains
        staples = self._find_bound_staples_with_compliments(scaffold_indices)

        # Unassign scaffold domains
        for domain_index in scaffold_indices[1:]:
            self._delta_e += self.trial_system.unassign_domain(SCAFFOLD_INDEX,
                    domain_index)

        # Unassign staples
        for staple_index in staples.keys():
            for domain_index in range(self.trial_system.chain_lengths[staple_index]):
                self._delta_e += self.trial_system.unassign_domain(staple_index,
                    domain_index)

        # Regrow scaffold
        try:
            self._grow_chain(SCAFFOLD_INDEX, scaffold_indices)
        except MoveRejection:
            accepted = False
            return accepted

        # Regrow staples
        try:
            self._grow_staples(staples)
        except MoveRejection:
            accepted = False
            return accepted

        # Test acceptance
        if self._configuration_accepted():
            self.accepted_system = self.trial_system
            accepted = True
        else:
            accepted = False

        return accepted

    def _grow_staples(staples):
        for staple_i, comp_domains in staples.items():

            # Pick domain on scaffold and staple to grow from
            scaffold_domain_i, staple_domain_i = random.choice(comp_domains)
            self._delta_e += self._set_staple_growth_point(staple_index,
                    staple_domain_i, scaffold_domain_i):

            # Grow remainder of staple
            staple_length = self.trial_system.chain_lengths[staple_index]
            self._grow_staple(staple_length, staple_index, staple_domain_i)

    def _select_scaffold_indices(self):
        """Return scaffold indices from random segment to end."""

        # Randomly select starting scaffold domain
        start_domain_index = random.randrange(self.scaffold_length)

        # Select direction to regrow, create index list
        direction = random.randrange(2)
        if direction == 1:
            scaffold_indices = range(start_domain_index, self.scaffold_length)
        else:
            scaffold_indices = range(start_domain_index, -1, -1)
        
        return scaffold_indices


class CBMCMovetype(MCMovetype):
    """Base class for configurational bias movetypes."""

    def __init__(self, cyclic, *args, **kwargs):
        self._bias = 0
        if cyclic:
            self._select_scaffold_indices_cyclic()
        else:
            self._select_scaffold_indices_linear()
        super().__init__(*args, **kwargs)

    def _calc_rosenbluth(self, weights, *args):
        """Calculate rosenbluth weight and return normalized weights."""
        rosenbluth_i = np.array(np.sum(weights))
        weights /= rosenbluth_i
        self._bias *= rosenbluth_i
        return weights

    def _select_config_with_bias(self, weights, configs, **kwargs)
        """Select configuration according to Rosenbluth weights."""
        random_n = random.random()
        cumalative_prob = 0
        for i, weight in enumerate(weights):
            cumalative_prob += weight
            if random_n < cumalative_prob:
                break

        return configs[i]

    def _select_old_config(self, *args, domain=domain):
        """Select old configuration."""
        p_old = self._accepted_system.get_domain_position(*domain)
        o_old = self._accepted_system.get_domain_orientation(*domain)
        return (p_old, o_old)
                
    def _grow_chain(self, chain_index, domain_indices, regrow_old=False):
        """Grow chain with configurational bias."""

        # Set method for calculating weights and selecting domain
        if endpoint == {}:
            calc_bias = self._calc_rosenbluth
        else:
            calc_bias = self._calc_fixed_end_rosenbluth

        if regrow_old:
            select_config = self._select_old_config
        else:
            select_config = self._select_config_with_bias

        # All position combinations for new domains
        dimensions = [XHAT, YHAT, ZHAT]
        directions = [-1, 1]
        positions = [np.array(j * i) for i in dimensions for j in directions]

        # Position vector of previous domain
        p_prev = self.trial_system.get_domain_position(chain_index,
                domain_indices[0])

        # Iterate through given indices, growing next domain from current index
        for i, domain_i in enumerate(domain_indices[:-1]):
            new_domain_i = domain_indices[i + 1]

            # List possible positions and associated boltzmann factors
            configs = []
            bfactors = []

            # Iterate through all possible new positions
            for position in positions:

                # Trial position vector
                p_new = np.array(p_prev + direction * dimension)

                # Check energies of each configuration
                occupancy = self.trial_system.get_position_occupancy(p_new)

                # No contribution from blocked domain
                if occupancy == BOUND:
                    continue

                # Add energy and configuration if binding possible
                elif occupancy == UNBOUND:
                    unbound_domain = self.trial_system.get_unbound_domain(p_new)
                    orientation = -self.trial_sytem.get_domain_orientation(
                            unbound_domain)
                    try:
                        delta_e = self.trial_system.check_domain_configuration(
                                p_new)
                    except ConstraintViolation:
                        pass
                    else:
                        configs.append((p_new, orientation))
                        bfactor = math.exp(-delta_e / self.trial_system.temp)
                        bfactors.append(bfactor)

                # If unoccupied site, all 6 orientations possible
                else:
                    num_orientations = 6
                    possible_configs.append((p_new, num_orientations))
                    boltzmann_factors.append(6)

            # Check if dead end
            if possible_configs == []:
                raise MoveRejection

            # Calculate bias and select position
            weights = calc_bias(boltzmann_factors, possible_configs)
            domain = (chain_index, domain_i)
            selected_config = select_config(domain, weights, possible_configs)

            # If unnoccupied lattice site, randomly select orientation
            p_new = p_prev = selected_config[0]
            if type(selected_config[1]) == int:
                o_new = self._select_random_orientation()
                self.trial_system.set_domain_configuration(
                        chain_index, new_domain_i, p_new, o_new)

            # Otherwise use complimentary orientation
            else:
                o_new = selected_config[1]
                self.trial_system.set_checked_domain_configuration(
                        chain_index, new_domain_i, p_new, o_new)

            # Update endpoints
            endpoints['steps'] -= 1

        return


class ExchangeCBMCMovetype(CBMCMovetype):
    """CB staple exchange movetype."""

    NORMLIZE THE BIAS

    def attempt_move(self):
        if random.random() < 0.5:
            accepted = self._delete_staple()
        else:
            accepted = self._insert_staple()

        return accepted

    def _staple_insertion_accepted(self, identity, overcounts):
        """Metropolis acceptance test for particle insertion."""
        N = len(self.accepted_system._identity_to_index[identity])
        ratio = self.bias / (N + 1)

        # Correct for overcounts (can't call normal method)
        p_accept = min(1, ratio) / overcounts
        if p_accept == 1:
            accept = True
        else:
            if p_accept > random.random():
                accept = True
            else:
                accept = False

        return accept

    def _staple_deletion_accepted(self, identity):
        """Metropolis acceptance test for particle deletion."""
        N = len(self.accepted_system._identity_to_index[identity])
        ratio = N / self.bia
        return self._test_acceptance(ratio)

    def _insert_staple(self):
        """Insert staple at random scaffold domain and grow."""

        # Randomly select staple identity and add chain
        staple_identity, domain_identities = (
                self.accepted_system.get_random_staple_identity())

        staple_index = self.trial_system.add_chain(staple_identity)
        staple_length = self.trial_system.chain_lengths[staple_index]

        # Select staple domain
        staple_domain = random.randrange(staple_length)
        domain_identity = domain_identities[staple_domain]

        # Select complimentary scaffold domain
        scaffold_domain = self.trial_system.identities.index(-domain_identity)
        
        # Number of bound domains in system (for calculating overcounts)
        init_num_bound_domains = self.trial_system.num_bound_domains

        # Set growth point domain
        try:
            delta_e = self._set_staple_growth_point(staple_index, staple_domain,
                    scaffold_domain):
        except MoveRejection:
            accepted = False
            return accepted
        else:
            self._bias *= math.exp(delta_e / self.accepted_system.temp)

        # Grow staple
        try:
            self._grow_staple(staple_length, staple_index, staple_domain)
        except MoveRejection:
            accepted = False
            return accepted

        # If the configuration is such that the staple can bind with the other
        # domain, then there are two ways this can happen, so the ratio should
        # be halved to prevent overcounting. If staple ends in multiply bound
        # state, save resulting overcounts.
        cur_num_bound_domains = self.trial_system.num_bound_domains
        D_bind_state = init_num_bound_domains - cur_num_bound_domains
        overcounts = D_bind_state

        # Test acceptance
        if self._staple_insertion_accepted(staple_identity, overcounts):
            self.accepted_system = self._trial_system
            accepted = True
        else:
            accepted = False

        return accepted

    def _delete_staple(self):
        """Delete random staple."""

        # Randomly select staple identity
        staple_identity, domain_identities = (
                self.accepted_system.get_random_staple_identity())

        # Randomly select staple
        try:
            staple_index = self.trial_system.get_random_staple_of_identity(
                    staple_identity)

        # No staples in system
        except IndexError:
            accepted = False
            return accepted

        # Unassign domains and collect bound domain indices
        # list of staple_domain_i and scaffold_domain_i
        bound_domains = []
        for domain_i in range(self.accepted_system.chain_lengths[staple_index]):
            occupancy = self.accepted_system.get_bound_domain(staple_index,
                    domain_i)
            if bound_domain != ():
                bound_domains.append((domain_i, bound_domain[1]))
            else:
                pass

            self._trial_system.unassign_domain(staple_index, domain_i)

        # Select domain to regrow from
        staple_domain_i, scaffold_domain_i = random.choice(bound_domains)

        # Set growth point and regrow
        delta_e += self._set_staple_growth_point(staple_index, staple_domain_i,
                scaffold_domain_i)
        self._bias *= math.exp(delta_e / self.trial_system.temp)
        self._grow_chain(staple_index, domain_i, regrow_old=True)

        # Delete chain to create correct trial system config
        self.trial_system.delete_chain(staple_index)

        # Test acceptance
        if self._staple_deletion_accepted(staple_identity):
            self.accepted_system = self._trial_system
            accepted = True
        else:
            accepted = False

        return accepted


class RegrowthCBMCMovetype(CBMCMovetype):

    def __init__(origami_system, *args):
        self.endpoints = {}
        self._ideal_random_walks = IdealRandomWalks()
        if origami_system.cyclic:
            self._select_scaffold_indices = self._select_scaffold_indices_linear
        else:
            self._select_scaffold_indices = self._select_scaffold_indices_cyclic

        super().__init__(origami_system, *args)

    def attempt_move(self):
        raise NotImplementedError

    def _calc_fixed_end_rosenbluth(self, weights, configs):
        """Return fixed endpoint Rosenbluth factor and modify weights."""
        for i, config in enumerate(configs):
            start = config[0]
            end = self.endpoints['position']
            steps = self.endpoints['steps']
            num_walks = self._ideal_random_walks(start, end, steps)
            weight[i] = weight[i] * num_walks

        return self._calc_rosenbluth_and_select(weights)

    def _select_scaffold_indices_linear(self):
        """Return scaffold indices between two random segments."""

        # Randomly select end points
        start_domain_i = end_domain_index = random.randrange(
                self.scaffold_length)
        while start_domain_i == end_domain_i:
            end_domain_i = random.randrange(self.scaffold_length)

        # Ensure start domain is 5'
        if start_domain_i > end_domain_i:
            start_domain_i, end_domain_i = end_domain_i, start_domain_i

        scaffold_indices = range(start_domain_i, end_domain_i)

        # Set endpoint (always 3')
        try:
            endpoint = self.trial_system.get_domain_position(SCAFFOLD_INDEX,
                    end_domain_i + 1)
        except KeyError:
            pass
        else:
            self._endpoints.append(endpoint)
        
        return scaffold_indices

    def _select_scaffold_indices_cyclic(self):
        """Return scaffold indices between two random segments."""

        # Randomly select end points
        start_domain_i = end_domain_index = random.randrange(
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
        scaffold_indices = range(start_domain_i, start_domain_i +
                end_domain_i)

        # Wrap indices
        for i, scaffold_domain_i in scaffold_indices:
            if scaffold_domain_i >= self.scaffold_length:
                scaffold_domain_i -= self.scaffold_length
                scaffold_indices[i] = scaffold_domain_i

        # Set endpoint
        endpoint_i = end_domain_i + 1
        if endpoint_i == self.scaffold_length:
            endpoint_i = 0
        else:
            pass

        endpoint.append(self.trial_system.get_domain_position(SCAFFOLD_INDEX,
                endpoint_i))

        return scaffold_indices

    def _find_bound_domains(self, scaffold_indices):
        """Find all bound staples and return with bound scaffold domains.

        Includes only bound scaffold domains in given indices.
        """
        staples = {}
        for domain_index in scaffold_indices:

            # Check if scaffold domain bound
            staple_domain = self.accepted_system.get_bound_domain(
                    SCAFFOLD_INDEX, domain_index)
            if staple_domain == ():
                continue
            else:
                staple_index, staple_domain_i = staple_domain
                if not staple_index in staples:
                    staples[staple_index] = []
                else:
                    pass

                staples[staple_index] = (staple_domain_i, domain_index)

        return staples


class StapleRegrowthCBMCMovetype(RegrowthCBMCMovetype):
    """CB staple regrowth movetype."""

    def attempt_move(self):
        """Regrow random staple."""

        # Randomly select staple
        try:
            staple_index = random.randrange(1,
                    len(self.trial_system.chain_lengths))

        # No staples in system
        except ValueError:
            accepted = False
            return accepted

        staple_length = self.accepted_system.chain_lengths[staple_index]

        # Unassign domains
        for domain_index in range(staple_length):
            delta_e = self.trial_system.unassign_domain(staple_index,
                    domain_index)
            self._bias *= math.exp(delta_e / self.temp)

        # Find all complimentary domains and randomly select growth point
        staple_ident = self._trial_system._chain_identities[staple_index]
        comp_domains = self._trial_system.staple_to_scaffold[staple_ident]

        staple_domain_i = random.randrange(staple_length)
        scaffold_domain_i = comp_domains[staple_domain_i]

        # Set growth point
        try:
            delta_e += self._set_staple_growth_point(staple_index,
                    staple_domain_index, scaffold_domain_i)
        except MoveRejection:
            accepted = False
            return accepted
        else:
            self._bias *= math.exp(delta_e / self.trial_system.temp)

        # Grow staple
        try:
            self._grow_staple(staple_length, staple_index, staple_domain_i)
        except MoveRejection:
            accepted = False
            return accepted

        # Regrow staple in old conformation
        trial_system = copy.deepcopy(self._trial_system)
        new_bias = self._bias
        self._bias = 1

        # Unassign domains and find all bound domains on staple
        bound_domains = []
        for staple_domain_i in range(staple_length):
            bound_domain = self._accepted_system.get_bound_domain(staple_index,
                    staple_domain_i)
            if occupancy != ():
                bound_domains.append((staple_domain_i, bound_domain[1]))

            delta_e = self._trial_system.unassign_domain(staple_index,
                    staple_domain_i)
            self._bias *= math.exp(delta_e / self.temp)

        # Pick a starting point at random (will average over simulation)
        staple_domain_i, scaffold_domain_i = random.choice(bound_domains)

        # Grow staple
        delta_e = self._set_staple_growth_point(staple_index, staple_domain_i,
                scaffold_domain_i)
        self._bias *= math.exp(delta_e / self.trial_system.temp)
        self._grow_staple(staple_length, staple_index, staple_domain_i,
                regrow_old=True)

        # Test acceptance
        ratio = bias_new / self._bias
        if self._test_acceptence(ratio):
            self._accepted_system = trial_system
            accepted = True
        else:
            accepted = False

        return accepted


class ScaffoldRegrowthCBMCMovetype(RegrowthCBMCMovetype):
    """CB scaffold regrowth movetype."""

    def attempt_move():
        """Randomly regrow terminal section of scaffold and bound staples.

        From a randomly selected scaffold domain in a random direction to the
        end.
        """

        # Pick section of scaffold to regrow
        scaffold_indices = self._select_scaffold_indices()

        # Find bound staples and all complimentary domains
        staples = self._find_bound_staples_with_compliments(scaffold_indices)

        # Unassign scaffold domains
        delta_e = 0
        for domain_index in scaffold_indices[1:]:
            delta_e += self.trial_system.unassign_domain(SCAFFOLD_INDEX,
                    domain_index)

        # Unassign staples
        for staple_index in staples.keys():
            for domain_index in range(self._trial_system.chain_lengths[staple_index]):
                delta_e += self._trial_system.unassign_domain(staple_index,
                    domain_index)

        self._bias *= math.exp(delta_e / self.trial_system.temp)

        # Regrow scaffold
        if self._endpoints == []:
            endpoint = None
        else:
            endpoint = self._endpoints[0]

        try:
            self._grow_chain(SCAFFOLD_INDEX, scaffold_indices,
                    endpoint=endpoint)
        except MoveRejection:
            accepted = False
            return accepted

        # Regrow staples
        for staple_index, comp_domains in staples.items():

            # Pick domain on scaffold and staple to grow from
            scaffold_domain_index, staple_domain_index = random.choice(
                    comp_domains)
            try:
                delta_e = self._set_staple_growth_point(staple_index,
                        staple_domain_index, scaffold_domain_index):
            except MoveRejection:
                accepted = False
                return accepted
            else:
                self._bias *= math.exp(delta_e / self.trial_system.temp)

            # Grow remainder of staple
            staple_length = self._trial_system.chain_lengths[staple_index]
            try:
                self._grow_staple(staple_length, staple_index,
                        staple_domain_index)
            except MoveRejection:
                accepted = False
                return accepted

        # Regrow in old conformation
        trial_system = copy.deepcopy(self._trial_system)
        new_bias = self._bias
        self._bias = 1

        # Find all bound staples
        staples = self._find_bound_domains(scaffold_indices)

        # Unassign scaffold
        delta_e = 0
        for domain_index in scaffold_indices[1:]:
            delta_e += self.trial_system.unassign_domain(SCAFFOLD_INDEX,
                    domain_index)

        # Unassign staples
        for staple_index in staples.keys():
            for domain_index in range(self._trial_system.chain_lengths[staple_index]):
                delta_e += self._trial_system.unassign_domain(staple_index,
                    domain_index)

        self._bias *= math.exp(delta_e / self.trial_system.temp)

        # Regrow scaffold (same endpoint)
            self._grow_chain(SCAFFOLD_INDEX, scaffold_indices,
                    endpoint=endpoint)

        # Regrow staples
        for staple_index, bound_domains in staples.items():

            # Pick a starting point at random (will average over simulation)
            scaffold_domain_index, staple_domain_index = random.choice(
                    bound_domains)
            delta_e = self._set_staple_growth_point(staple_index,
                    staple_domain_index, scaffold_domain_index):
            self._bias *= math.exp(delta_e / self.trial_system.temp)

            # Grow remainder of staple
            staple_length = self._trial_system.chain_lengths[staple_index]
            try:
                self._grow_staple(staple_length, staple_index,
                        staple_domain_index, regrow_old=True)
            except MoveRejection:
                accepted = False
                return accepted

        # Test acceptance
        ratio = bias_new / self._bias
        if self._test_acceptence(ratio):
            self._accepted_system = trial_system
            accepted = True
        else:
            accepted = False

        return accepted


class ConservedTopologyCBMCMovetype(RegrowthCBMCMovetype):
    """CB constant topology scaffold/staple regrowth movetype."""

    def attempt_move(self):
        """Regrow scaffold and staples with fixed topology."""

        # Pick scaffold segment to regrow
        scaffold_indices = self._select_scaffold_indices()

        # Find and classify all bound staples
        staples = self._find_bound_domains(scaffold_indices)
        staple_types = self._classify_staples(staples, scaffold_indices)
        initial_endpoints = self.endpoints

        # Unassign scaffold and non-externaly bound staples
        self._unassign_domains(scaffold_indices, staples)

        # Regrow scaffold and staples
        self._grow_scaffold_and_staples(scaffold_indices, staples, staple_types)

        # Regrow in old conformation
        trial_system = copy.deepcopy(self._trial_system)
        new_bias = self._bias
        self._bias = 1
        self.endpoints = initial_endpoints

        # Unassign scaffold and non-externaly bound staples
        self._unassign_domains(scaffold_indices, staples)

        # Regrow scaffold and staples
        self._grow_scaffold_and_staples(scaffold_indices, staples, staple_types)

        # Test acceptance
        ratio = bias_new / self._bias
        if self._test_acceptence(ratio):
            self._accepted_system = trial_system
            accepted = True
        else:
            accepted = False

        return accepted

    def _classify_staples(self, staples, scaffold_indices):
        """Find and classify all staples bound to give indices.

        Returns a dictionary with single, multiple, and external keys to bound
        staples.
        """
        self.endpoints['singly_bound'] = []
        self.endpoints['multiply_bound'] = []
        for staple_index, staple in staples.values():
            staple_length = self.trial_system.chain_lengths[staple_index]

            # Check unbound domains if bound to another part of scaffold
            for staple_domain_i in range(staple_length):
                if filter(lambda x: x[0] == staple_domain_i, staple) != []:
                    domain = (staple_index, staple_domain_i)
                    bound_domain == self.trial_system.get_bound_domain(*domain)
                    if bound_domain != ():
                        for domain in staple:
                            self.endpoints['indices'].append(domain[1])
                            position = self.trial_system.get_domain_position(
                                    *domain)
                            self.endpoints['positions'].append(position)
                            Ni = domain[1] - scaffold_indices[0]

                            # Correct if cyclic
                            if Ni < 0:
                                Ni = (domains[1] + self.scaffold_length -
                                        scaffold_indices)

                            np.concatenate([self.endpoints['steps'], Ni])

                        del staples[staple_index]
                        break

            # If no externally bound domains, save possible growth points
            else:
                if len(staple) == 1:
                    staple_types['singly_bound'].append(staple[0][1])
                else:
                    scaffold_domains = [pair[1] for pair in staple]
                    staple_types['multiply_bound'].extend(scaffold_domains)

        return staple_types

    def _unassign_domains(self, scaffold_indices, staples):
        """Unassign all give scaffold and non-externaly bound staple domains."""

        # Unassign scaffold domains
        delta_e = 0
        for domain_index in scaffold_indices[1:]:
            delta_e += self.trial_system.unassign_domain(SCAFFOLD_INDEX,
                    domain_index)

        # Unassign staples
        for staple_index in staples.keys():
            for domain_index in range(self._trial_system.chain_lengths[staple_index]):
                delta_e += self._trial_system.unassign_domain(staple_index,
                    domain_index)

        self._bias *= math.exp(delta_e / self.trial_system.temp)

    def _calc_fixed_end_rosenbluth(self, weights, configs):
        """Return fixed endpoint Rosenbluth factor and modify weights."""
        for i, config in enumerate(configs):
            start_point = config[0]
            num_walks = 1
            for endpoint_i, in range(len(self.endpoints['indices'])):
                endpoint_p = endpoint['positions'][endpoint_i]
                endpoint_s = endpoint['steps'][endpoint_i]
                num_walks *= self._ideal_random_walks(start_point, endpoint_p,
                        endpoint_s)

            weight[i] = weight[i] * num_walks

        return self._calc_rosenbluth_and_select(weights)

    def _grow_staple_and_update_endpoints(self, domain_i, staple_types,
            staples):
        """Grow staples, update endpoints, and return modified staple_types."""

        # Grow singly bound staple if present
        if domain_i in staple_types['singly_bound'].keys():
            staple_domain = staple_types['singly_bound'][domain_i]
            self._grow_staple(staple_domain, domain_i)
            del staple_types['singly_bound'][domain_i]

        # Grow multiply bound staple and add endpoints if present
        elif domain_i in staple_types['multiply_bound'].keys():
            staple_domain = staple_types['multiply_bound'][domain_i]
            self._grow_staple(staple_domain, domain_i)
            del staple_types['multiply_bound'][domain_i]

            # Add remaining staple domains to endpoints
            staple_index = staple_domain[0]
            staples[staple_index].remove((staple_domain, domain_i))
            for staple_domain_i, scaffold_domain_i in staples[staple_index]:
                self.endpoints['indices'].append(scaffold_domain_i)
                position = self.trial_system.get_domain_position(
                        staple_index, staple_domain_i)
                self.endpoints['position'].append(position)
                Ni = scaffold_domain_i - domain_i

                # Correct Ni if cyclic
                if Ni < 0:
                    Ni = scaffold_domain_i + self.scaffold_length- domain_i

                np.concatenate(self.endpoints['steps'], [Ni])
                del staple_types['multiply_bound'][scaffold_domain_i]

        # Otherwise continue with scaffold
        else:
            pass

        return staple_types

    def _update_endpoints(self, domain_i):
        try:
            endpoint_i = self.endpoints['indices'].index(domain_i)
        except ValueError:
            pass
        else:
            del self.endpoints['indices'][endpoint_i]
            del self.endpoints['positions'][endpoint_i]
            del self.endpoints['steps'][endpoint_i]

        endpoints['steps'] -= 1

    def _grow_scaffold_and_staples(self, scaffold_indices, staples,
                staple_types):
        """Grow scaffold and staple chains."""

        # All position combinations for new domains
        dimensions = [XHAT, YHAT, ZHAT]
        directions = [-1, 1]
        positions = [np.array(j * i) for i in dimensions for j in directions]

        # Position vector of previous domain
        p_prev = self.trial_system.get_domain_position(chain_index,
                domain_indices[0])

        # Iterate through given indices, growing next domain from current index
        for i, domain_i in enumerate(domain_indices[:-1]):
            new_domain_i = domain_indices[i + 1]

            # Grow staples
            grow_staple_and_update_endpoints(domain_i, staple_types, staples)

            # List of tuples of position and orientations (if bound, orientation
            # vector, otherwise the number of possible orientation (6)
            configs = []

            # List of associated boltzmann factors
            bfactors = []

            # Iterate through all possible new positions
            for position in positions:

                # Trial position vector
                p_new = p_prev + position

                # Check energies of each configuration
                occupancy = self.trial_system.get_position_occupancy(p_new)

                # No contribution from blocked domain
                if occupancy == BOUND:
                    continue

                # Add energy and configuration if binding possible
                elif occupancy == UNBOUND:
                    unbound_domain = self.trial_system.get_unbound_domain(p_new)
                    orientation = -self.trial_sytem.get_domain_orientation(
                            unbound_domain)
                    try:
                        delta_e = self.trial_system.check_domain_configuration(
                                p_new)
                    except ConstraintViolation:
                        pass
                    else:
                        configs.append((p_new, orientation))
                        bfactor = math.exp(-delta_e / self.trial_system.temp)
                        bfactors.append(bfactor)

                # If unoccupied site, all 6 orientations possible
                else:
                    num_orientations = 6
                    possible_configs.append((p_new, num_orientations))
                    boltzmann_factors.append(6)

            # Check if dead end
            if possible_configs == []:
                raise MoveRejection

            # Calculate bias and select position
            weights = calc_bias(boltzmann_factors, possible_configs)
            domain = (chain_index, domain_i)
            selected_config = select_config(domain, weights, possible_configs)

            # If unnoccupied lattice site, randomly select orientation
            p_new = p_prev = selected_config[0]
            if type(selected_config[1]) == int:
                o_new = self._select_random_orientation()
                self.trial_system.set_domain_configuration(
                        chain_index, new_domain_i, p_new, o_new)

            # Otherwise use complimentary orientation
            else:
                o_new = selected_config[1]
                self.trial_system.set_checked_domain_configuration(
                        chain_index, new_domain_i, p_new, o_new)

            # Update endpoints
            self._update_endpoints(domain_i)

        return
