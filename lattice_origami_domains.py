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
    INSERT_STAPLE = 0
    DELETE_STAPLE = 1
    TRANSLATE_STAPLE = 2
    ROTATE_STAPLE = 3
    REGROW_STAPLE = 4
    REGROW_SCAFFOLD_AND_BOUND_STAPLES = 5
    ROTATE_ORIENTATION_VECTOR = 6
    EXCHANGE_STAPLE = 7


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


class OrigamiSystem:
    """Simple cubic lattice model of DNA origami at domain level resolution.

    The domains are 8 bp long. See reports/modelSpecs/domainResModelSpecs for
    exposition on the model.

    I've used get methods instead of properties as properties don't take
    indices. I would have to index the internal structure directly, which
    I don't want to do.
    """

    def __init__(self, input_file, step, temp, cation_M):
        self.temp = temp

        # Domain identities of each chain
        self.identities = input_file.identities

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

    def set_checked_domain_conguration(self, chain_index, domain_index,
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
                neighbour_domain = (chain_index, domain_index + direction)
                try:
                    occupancy = self.get_domain_occupancy(*neighbour_domain)
                except KeyError:
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
        self._output_file = output_file

        # Set seed for python's Mersenne Twister (64 bits from os)
        seed = os.urandom(8)
        random.seed(seed)
        output_file.write_seed(seed)

        # Create cumalative probability distribution for movetypes
        # List to associate movetype method with index in distribution
        self._movetype_methods = []
        self._movetypes = []
        self._movetype_probabilities = []
        cumulative_probability = 0
        for movetype, probability in move_settings.items():

            # I still wonder if there is a way to avoid all the if statements
            if movetype == MOVETYPE.EXCHANGE_STAPLE:
                movetype_method = self._exchange_staple
            elif movetype == MOVETYPE.REGROW_STAPLE:
                movetype_method = self._regrow_staple
            elif movetype == MOVETYPE.REGROW_SCAFFOLD_AND_BOUND_STAPLES:
                movetype_method = self._regrow_scaffold_and_bound_staples
            elif movetype == MOVETYPE.ROTATE_ORIENTATION_VECTOR:
                movetype_method = self._rotate_orientation_vector

            cumulative_probability += probability
            self._movetype_methods.append(movetype_method)
            self._movetypes.append(movetype)
            self._movetype_probabilities.append(cumulative_probability)

        # Check movetype probabilities are normalized
        # This could break from rounding errors
        # Might be better to normalize whatever numbers given
        if cumulative_probability != 1:
            print('Movetype probabilities not normalized')
            sys.exit()

        # Two copies of the system allows easy reversion upon rejection
        self._accepted_system = origami_system
        self._trial_system = copy.deepcopy(origami_system)

        # Change in energy for a proposed trial move
        self._delta_e = 0

        # Frequency for translating system back to origin
        self.center_freq = 1

        # Current movetype
        self._movetype = -1

    def run(self, num_steps, logging=True):
        """Run simulation for given number of steps."""

        for step in range(1, num_steps + 1):
            self._delta_e = 0
            self._trial_system = copy.deepcopy(self._accepted_system)
            movetype_method = self._select_movetype()
            outcome = movetype_method()
            if value_is_multiple(step, self.center_freq):
                self._accepted_system.center()

            self._output_file.check_and_write(self._accepted_system, step)

            # Loggging hack
            if logging:
                print(step, self._movetype, outcome,
                        len(self._accepted_system.chain_lengths), self._delta_e)
            else:
                pass

    def _select_movetype(self):
        """Return movetype method according to distribution."""
        random_number = random.random()
        lower_boundary = 0
        for movetype_index, upper_boundary in enumerate(
                self._movetype_probabilities):
            if lower_boundary <= random_number < upper_boundary:
                movetype_method = self._movetype_methods[movetype_index]
                self._movetype = self._movetypes[movetype_index]
                break
            else:
                lower_boundary = upper_boundary

        return movetype_method

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

    def _insert_staple(self):
        """Insert staple at random scaffold domain and grow."""

        # Randomly select staple identity and add chain
        staple_identity, domain_identities = (
                self._accepted_system.get_random_staple_identity())

        staple_index = self._trial_system.add_chain(staple_identity)
        staple_length = self._trial_system.chain_lengths[staple_index]

        # Select staple domain
        staple_domain = random.randrange(staple_length)
        domain_identity = domain_identities[staple_domain]

        # Select complimentary scaffold domain
        scaffold_index = 0
        scaffold_domain = self._trial_system.identities.index(-domain_identity)
        
        # Select correct position and orientation
        position = self._trial_system.get_domain_position(scaffold_index,
                scaffold_domain)
        orientation = -self._trial_system.get_domain_orientation(scaffold_index,
                scaffold_domain)

        # Number of bound domains in system (for calculating overcounts)
        init_num_bound_domains = self._trial_system.num_bound_domains

        # Attempt to set position (fails if bound or twist constraints violated)
        # First staple binding method?
        try:
            self._delta_e += self._trial_system.set_domain_configuration(
                    staple_index, staple_domain, position, orientation)
        except ConstraintViolation:
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
        cur_num_bound_domains = self._trial_system.num_bound_domains
        D_bind_state = init_num_bound_domains - cur_num_bound_domains
        overcounts = D_bind_state

        # Test acceptance
        if self._staple_insertion_accepted(staple_identity, overcounts):
            self._accepted_system = self._trial_system
            accepted = True
        else:
            accepted = False

        return accepted

    def _delete_staple(self):
        """Delete random staple."""

        # Randomly select staple identity
        staple_identity, domain_identities = (
                self._accepted_system.get_random_staple_identity())

        # Randomly select staple
        try:
            staple_index = self._trial_system.get_random_staple_of_identity(
                    staple_identity)

        # No staples in system
        except IndexError:
            accepted = False
            return accepted

        # deletion sub method?
        self._delta_e += self._trial_system.delete_chain(staple_index)

        # Test acceptance
        if self._staple_deletion_accepted(staple_identity):
            self._accepted_system = self._trial_system
            accepted = True
        else:
            accepted = False

        return accepted

    def _exchange_staple(self):
        if random.random() < 0.5:
            accepted = self._delete_staple()
        else:
            accepted = self._insert_staple()

        return accepted

    def _regrow_staple(self):
        """Regrow random staple."""

        # Randomly select staple
        try:
            staple_index = random.randrange(1,
                    len(self._trial_system.chain_lengths))

        # No staples in system
        except ValueError:
            accepted = False
            return accepted

        staple_length = self._accepted_system.chain_lengths[staple_index]

        # Find all bound domains and randomly select growth point
        bound_staple_domains = []
        for domain_index in range(staple_length):
            bound_domain = self._accepted_system.get_bound_domain(staple_index,
                    domain_index)
            if bound_domain != ():
                bound_staple_domains.append(domain_index)

        bound_domain_index = random.choice(bound_staple_domains)

        # Unassign domains
        for domain_index in range(staple_length):
            if domain_index == bound_domain_index:
                continue
            self._delta_e += self._trial_system.unassign_domain(staple_index,
                    domain_index)

        # Grow staple
        try:
            self._grow_staple(staple_length, staple_index, bound_domain_index)
        except MoveRejection:
            accepted = False
            return accepted
        
        # If staple started and ended in multiply bonded state, save resulting
        # overcounts
        cur_num_bound_domains = self._trial_system.num_bound_domains
        D_bind_state = init_num_bound_domains - cur_num_bound_domains
        if D_bind_state >= 0:
            self._overcount.append(len(bound_staple_domains))

        # Test acceptance
        if self._configuration_accepted():
            self._accepted_system = self._trial_system
            accepted = True
        else:
            accepted = False

        return accepted

    def _regrow_scaffold_and_bound_staples(self):
        """Randomly regrow terminal section of scaffold and bound staples.

        From a randomly selected scaffold domain in a random direction to the
        end.
        """

        # Pick section of scaffold to regrow
        scaffold_indices = self._select_scaffold_indices()

        # Find all staples bound to scaffold (includes repeats)
        staples = {}
        for domain_index in scaffold_indices:
            staple_domain = self._accepted_system.get_bound_domain(
                    scaffold_index, domain_index)
            if staple_domain == ():
                continue
            else:
                staple_index = staple_domain[0]
                staple_domain_i = staple_domain[1]
                try:
                    staples[staple_index].append((domain_index, staple_domain_i))
                except KeyError:
                    staples[staple_index] = [(domain_index, staple_domain_i)]

        # Unassign scaffold domains
        for domain_index in scaffold_indices[1:]:
            self._delta_e += self._trial_system.unassign_domain(scaffold_index,
                    domain_index)

        # Unassign staples
        for staple_index in staples.keys():
            for domain_index in range(self._trial_system.chain_lengths[staple_index]):
                self._delta_e += self._trial_system.unassign_domain(staple_index,
                    domain_index)

        # Regrow scaffold
        try:
            self._grow_chain(scaffold_index, scaffold_indices)
        except MoveRejection:
            accepted = False
            return accepted

        # Regrow staples
        for staple_index, bound_domains in staples.items():

            # Pick domain on scaffold and staple to grow from
            # Could do in method if want to include in config bias
            scaffold_domain_index, staple_domain_index = random.choice(
                    bound_domains)
            position = self._trial_system.get_domain_position(scaffold_index,
                    scaffold_domain_index)
            orientation = self._trial_system.get_domain_orientation(
                    scaffold_index, scaffod_domain_index)

            # Attempt to set growth domain
            # Needs to be in first staple method
            try:
                self._delta_e += self._trial_system.set_domain_configuration(
                        staple_index, staple_domain_index, position, o_new)
            except ConstraintViolation:
                accepted = False
                return accepted

            # Number of bound domains in system (for calculating overcounts)
            init_num_bound_domains = self._trial_system.num_bound_domains

            # Grow remainder of staple
            staple_length = self._trial_system.chain_lengths[staple_index]
            try:
                self._grow_staple(staple_length, staple_index,
                        staple_domain_index)
            except MoveRejection:
                accepted = False
                return accepted

            # If staple started and ended in multiply bound state, save
            # resulting overcounts
            cur_num_bound_domains = self._trial_system.num_bound_domains
            D_bind_state = init_num_bound_domains - cur_num_bound_domains
            if D_bind_state >= 0:
                self._overcount.append(len(bound_domains))

        # Test acceptance
        if self._configuration_accepted():
            self._accepted_system = self._trial_system
            accepted = True
        else:
            accepted = False

        return accepted

    def _rotate_orientation_vector(self):
        """Randomly rotate random domain."""

        # Select random chain and domain
        chain_lengths = self._accepted_system.chain_lengths
        chain_index = random.randrange(len(chain_lengths))
        domain_index = random.randrange(chain_lengths[chain_index])

        # Reject if in bound state
        occupancy = self._accepted_system.get_domain_occupancy(chain_index,
                domain_index)
        if occupancy == BOUND:
            accepted = False
            return accepted
        else:
            pass

        # Select random orientation and update (always accepted)
        dimension = random.choice([XHAT, YHAT, ZHAT])
        direction = random.randrange(-1, 2, 2)
        o_new = dimension * direction
        self._accepted_system.set_domain_orientation(chain_index, domain_index,
                o_new)

        accepted = True
        return accepted


class GCMCMetropolisSimulation(GCMCSimulation):
    """GCMC simulation with metropolis acceptance criteria."""

    def _configuration_accepted(self):
        """Metropolis acceptance test for configuration change."""
        T = self._accepted_system.temp
        boltz_factor = math.exp(-self._delta_e / T)
        return self._test_acceptance(boltz_factor)

    def _staple_insertion_accepted(self, identity, overcounts):
        """Metropolis acceptance test for particle insertion."""
        T = self._accepted_system.temp
        boltz_factor = math.exp(-self._delta_e / T)
        N = len(self._accepted_system._identity_to_index[identity])
        ratio = boltz_factor / (N + 1)

        # Correct for overcounts
        ratio = ratio / overcounts
        return self._test_acceptance(ratio)

    def _staple_deletion_accepted(self, identity):
        """Metropolis acceptance test for particle deletion."""
        T = self._accepted_system.temp
        boltz_factor = math.exp(-self._delta_e / T)
        N = len(self._accepted_system._identity_to_index[identity])
        return self._test_acceptance(N * boltz_factor)

    def _grow_chain(self, chain_index, domain_indices):
        """Randomly grow out chain from given domain indices.

        Updates changes in energy as binding events occur."""

        # Iterate through given indices, growing next domain from current index
        for i, domain_index in enumerate(domain_indices[:-1]):
            new_domain_index = domain_indices[i + 1]

            # Randomly select neighbour lattice site for new position
            dimension = random.choice([XHAT, YHAT, ZHAT])
            direction = random.randrange(-1, 2, 2)

            # Position vector of previous domain
            r_prev = self._trial_system.get_domain_position(chain_index, domain_index)

            # Trial position vector
            r_new = np.array(r_prev + direction * dimension)

            # Randomly select new orientation
            dimension = random.choice([XHAT, YHAT, ZHAT])
            direction = random.randrange(-1, 2, 2)

            # Trial position orientation
            o_new = dimension * direction

            # Attempt to set position
            try:
                self._delta_e += self._trial_system.set_domain_configuration(
                        chain_index, new_domain_index, r_new, o_new)
            except ConstraintViolation:
                raise MoveRejection

        return

    def _grow_staple(self, staple_length, staple_index, domain_index):
        """Randomly grow staple out from given domain in both directions."""

        # Grow in three-prime direction
        staple_indices = range(domain_index, staple_length)
        try:
            self._grow_chain(staple_index, staple_indices)
        except MoveRejection:
            raise

        # Grow in five-prime direction
        staple_indices = range(domain_index, -1, -1)
        try:
            self._grow_chain(staple_index, staple_indices)
        except MoveRejection:
            raise

    def _select_scaffold_indices(self):

        # Randomly select starting scaffold domain
        scaffold_index = 0
        scaffold_length = self._accepted_system.chain_lengths[scaffold_index]
        start_domain_index = random.randrange(scaffold_length)

        # Select direction to regrow, create index list
        direction = random.randrange(2)
        if direction == 1:
            scaffold_indices = range(start_domain_index, scaffold_length)
        else:
            scaffold_indices = range(start_domain_index, -1, -1)
        
        return scaffold_indices


class GCMCConfigurationBiasSimulation(GCMCSimulatoin):
    """GCMC configurational bias bound staple simulations."""

    def _regrow_chain(self):
        """Regrow scaffold between two points with configurational bias.
        
        All attached staple strands are regrown with configurational bias,
        although unlike the scaffold strand, they do not require the fixed
        end-point modification to the trial and Rosenbluth weights.
        """
