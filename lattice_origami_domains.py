#!/usr/env python

"""Run MC simulations of the lattice domain-resolution DNA origami model.

Data:

Functions:

Classes:

"""

import json
import math
import sys
import random
from enum import Enum

import h5py
import numpy as np
import scipy.constants


# Occupancy states
UNASSIGNED = 0
UNBOUND = 1
BOUND = 2

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


class MoveRejection(Exception):
    """Used for early move rejection."""
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


def rotate_vector_quarter(vector, rotation_axis, direction):
    """Rotate given vector pi/2 about given axis in given direction."""
    if all(np.abs(rotation_axis) == XHAT):
        y = vector[1]
        z = vector[2]
        vector[1] = direction * -z
        vector[2] = direction * y

    elif all(np.abs(rotation_axis) == YHAT):
        x = vector[0]
        z = vector[2]
        vector[2] = direction * -x
        vector[0] = direction * z

    elif all(np.abs(rotation_axis) == ZHAT):
        x = vector[0]
        y = vector[1]
        vector[0] = direction * -y
        vector[1] = direction * x

    return vector


def calc_hybridization_energy(sequence, T):
    """Calculate hybridization energy of domains with NN model.
    
    OUtputs energies in K (avoid multiplying by KB when calculating acceptances.
    """
    complimentary_sequence = calc_complimentary_sequence(sequence)

    # Initiation free energy
    DG_init = NN_ENTHALPY['INITIATION'] - T * NN_ENTROPY['INITIATION']

    # Symmetry penalty for palindromic sequences
    if sequence_is_palindromic(sequence):
        DG_sym = - T * NN_ENTROPY['SYMMETRY_CORRECTION']
    else:
        DG_sym = 0

    # NN pair energies
    DG_stack = 0
    for base_index in range(0, len(sequence), 2):
        first_pair = sequence[base_index : base_index + 2]
        second_pair = complimentary_sequence[base_index : base_index + 2]
        key = first_pair + '/' + second_pair

        # Not all permutations are included in dict as some reversals have
        # identical energies
        try:
            DG_stack += NN_ENTHALPY[key] - T * NN_ENTROPY[key]
        except KeyError:
            key = key[::-1]
            DG_stack += NN_ENTHALPY[key] - T * NN_ENTROPY[key]

    # Terminal AT penalties
    terminal_AT_pairs = 0
    for sequence_index in [0, -1]:
        if sequence[sequence_index] in ['A', 'T']:
            terminal_AT_pairs += 1

    if terminal_AT_pairs > 0:
        DG_at = (NN_ENTHALPY['TERMINAL_AT_PENALTY'] -
                T * NN_ENTROPY['TERMINAL_AT_PENALTY']) * terminal_AT_pairs
    else:
        DG_at = 0

    DG_hybridization = DG_init + DG_sym + DG_stack + DG_at

    # Convert from kcal/mol to K (so avoid KB later)
    DG_hybridization = DG_hybridization * J_PER_CAL * 1000 / R

    return DG_hybridization


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


# somehow use decorators to modify creation of positions so can select
# periodic or nonperiodic boundary conditions
class PeriodicPosition:
    """Periodic position vectors."""

    def __init__(self, max_dimension, position):
        self._position = np.array(position)
        self._max_dimension = max_dimension

        for component_index, component in enumerate(self._position):
            self._wrap_component(component_index, component)

    def __getitem__(self, index):
        return self._position[index]

    def __setitem__(self, index, coordinate):
        self._wrap_component(index, coordinate)

    def __add__(self, other_position):

        # This method does not unwrap the positions, which would be necessary
        # if trying to add position vectors of the same molecule (as they could
        # be split at the boundary.
        #
        # Also the other position must be a numpy array, which CURRENTLY
        # (16-03-30) is the only use case. Making it more general would
        # remove the speed advantage of using the numpy __add__ method.
        absolute_sum = self._position + other_position
        return PeriodicPosition(self._max_dimension, absolute_sum)

    def __sub__(self, other_periodic_position):

        # As with __add__, this method is dangerously flawed. The use case
        # is always between two contiguous domains, so only one component
        # should be non zero, and that component must then be +-1. Any use
        # outside of this is unsupported.
        same_cell_dif = self._position - other_periodic_position._position
        a
        return PeriodicPosition(self._max_dimension, absolute_dif)

    def __eq__(self, other_periodic_position):
        return all(self._position == other_periodic_position._position)

    def __iter__(self):
        return self._position.tolist().__iter__()

    def _wrap_component(self, component_index, component):
        if component > self._max_dimension:
            self._position[component_index] = (-self._max_dimension +
                    (component - self._max_dimension))
        elif component < self._max_dimension:
            self._position[component_index] = (self._max_dimension +
                    (component - self._max_dimension))
        else:
            pass


class OrigamiSystem:

    def __init__(self, input_file, step, max_dimension, temp, staple_p):

        # Set configuration from specified inputfile and step
        indices = []
        chain_identities = []
        positions = []
        occupancies = {}
        orientations = []

        for chain_index, chain in enumerate(input_file.chains(step)):
            indices.append(chain['index'])
            chain_identities.append(chain['identity'])
            chain_positions = []
            for position in chain['positions']:
                position = PeriodicPosition(max_dimension, position)
                chain_positions.append(position)

            positions.append(chain_positions)
            for domain_index, position in enumerate(chain_positions):
                position_key = tuple(position)
                self.add_occupancy(position_key, (chain_index, domain_index))

            orientations.append(chain['orientations'])

        # Calculate constants
        volume = (max_dimension * 2) ** 3
        chain_lengths = [len(positions[i]) for i in range(positions)]

        # Calculate and store hybridization energies
        hybridization_energies = []
        sequences = input_file.sequences
        for sequence in sequences:
            energy = calc_hybridization_energy(sequence, temp)
            hybridization_energies.append(energy)

        # Set instance variables
        self.identities = input_file.identities
        self.sequences = sequences
        self.max_dimension = max_dimension
        self.volume = volume
        self.temp = temp
        self.chain_lengths = chain_lengths
        self.staple_p = staple_p

        self._chain_identities = chain_identities
        self._indices = indices
        self._positions = positions
        self._occupancies = occupancies
        self._orientations = orientations
        self._current_chain_index = max(indices)
        self._hybridization_energies = hybridization_energies

    @property
    def chains(self):
        chains = []
        for working_index, unique_index in enumerate(self._indices):
            chain = {}
            chain['index'] = unique_index
            chain['identity'] = self.identities[working_index]
            positions = []
            for position in self._positions[working_index]:
                position = list(position)
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
        # Properties don't allow arguments
        return self._positions[chain_index][domain_index]

    def set_domain_position(self, chain_index, domain_index, position):
        previous_domain_position = self._positions[chain_index][domain_index]
        previous_domain_position = tuple(previous_domain_position)
        self.remove_occupancy(previous_domain_position, (chain_index, domain_index))

        self._positions[chain_index][domain_index] = position
        position = tuple(position)
        self.add_occupancy(position, (chain_index, domain_index))

    def get_domain_orientation(self, chain_index, domain_index):
        return self._orientations[chain_index][domain_index]

    def set_domain_orientation(self, chain_index, domain_index, orientation):
        self._orientations[chain_index][domain_index] = orientation

    def random_staple_identity(self):
        staple_identity = random.randrange(1, len(self.identities))
        domain_identities = self.identities[staple_identity]
        return staple_identity, domain_identities

    def add_chain(self, identity, positions, orientations):
        self.identities.append(identity)
        self._positions.append(positions)
        #chain_index = len(self._positions)
        #for domain_index, position in enumerate(positions):
        #    position = tuple(position)
        #    self._add_occupancy(position, (chain_index, domain_inde))

        self._orientations.append(orientations)
        self._current_chain_index += 1
        self._indices.append(self._current_chain_index)
        self.chain_lengths.append(len(positions))

    def delete_chain(self, chain_index):
        del self._indices[chain_index]
        del self.identities[chain_index]
        positions = self._positions[chain_index]
        for domain_index, position in enumerate(positions):
            position = tuple(position)
            self.remove_occupancy(position, (chain_index, domain_index))

        del self._positions[chain_index]
        del self._orientations[chain_index]
        del self.chain_lengths[chain_index]

    def position_occupied(self, position):
        overlap = False
        position = tuple(position)
        if position in self._occupancies:
            overlap = True
        else:
            pass

        return overlap

    def get_position_occupancy(self, position):
        try:
            occupancy = self._occupancies[position]['state']
        except KeyError:
            occupancy = UNASSIGNED
        return occupancy

    def get_domain_occupancy(self, chain_index, domain_index):

        # Consider having another data structure for direct access
        position = self.get_domain_position(chain_index, domain_index)
        occupancy = self.get_position_occupancy(position)
        return occupancy

    def add_occupancy(self, position, index):
        try:
            self._occupancies[position]['state'] = BOUND
            complimentary_index = self._occupancies[position]['identity']
            del self._occupancies[position]['identity']
            self._occupancies[position][complimentary_index] = index
            self._occupancies[position][index] = complimentary_index

        except KeyError:
            self._occupancies[position]['state'] = UNBOUND
            self._occupancies[position]['identity'] = complimentary_index

    def remove_occupancy(self, position, index):
        try:
            if self._occupancies[position]['state'] == BOUND:
                self._occupancies[position] = UNBOUND
                complimentary_index = self._occupancies[position][index]
                del self._occupancies[position][index]
                del self._occupancies[position][complimentary_index]
                self._occupancies[position]['identity'] = complimentary_index
            else:
                del self._occupancies[position]

        except KeyError:
            pass

    def get_domain_domain_at_position(self, position):
        return self._occupancies[position]['identity']

    def get_bound_domain(self, chain_index, domain_index):
        try:
            position = tuple(self._positions[chain_index][domain_index])
            bound_index = self._occupancies[position][(chain_index, domain_index)]
        except KeyError:
            bound_index = ()

        return bound_index

    def domains_match(self, chain_index_1, domain_index_1,
                chain_index_2, domain_index_2):

        # Determine domain identities
        chain_identity_1 = self._chain_identities[chain_index_1]
        domain_identity_1 = self.identities[chain_identity_1][domain_index_1]
        chain_identity_2 = self._chain_identities[chain_index_2]
        domain_identity_2 = self.identities[chain_identity_2][domain_index_2]

        # Complimentary if domain identities sum to 0
        complimentary = domain_identity_1 + domain_identity_2

        # Check if orientations are correct
        if complimentary == 0:
            orientation_1 = self._orientations[chain_index_1][domain_index_2]
            orientation_2 = self._orientations[chain_index_1][domain_index_2]

            # They should be opposite vectors, thus correct if sum to 0
            complimentary_orientations = orientation_1 + orientation_2
            if complimentary_orientations.sum() == 0:
                match = True
            else:
                match = False
        else:
            match = False

        return match

    def get_hybridization_energy(self, chain_index, domain_index):

        chain_identity = self._chain_identities[chain_index]
        domain_identity = self.identities[chain_identity][domain_index]

        # Because identites start at 1
        energy_index = abs(domain_identity) - 1
        return self._hybridization_energies[energy_index]

    # probably merge these two as they will always be called together
    def domains_part_of_same_helix(self, chain_index, domain_index_1,
            domain_index_2):

        # Ensure domain 1 is 3'
        if domain_index_2 < domain_index_1:
            domain_i_three_prime = domain_index_2
            domain_i_five_prime = domain_index_1

        else:
            domain_i_three_prime = domain_index_1
            domain_i_five_prime = domain_index_2

        position_three_prime = (
                self._positions[chain_index][domain_i_three_prime])
        position_five_prime = (
                self._positions[chain_index][domain_i_five_prime])

        orientation_three_prime = (
                self._orientations[chain_index][domain_i_three_prime])
        #orientation_five_prime = (
        #        self._orientations[chain_index][domain_i_five_prime])

        if (self._occupancies[position_three_prime]['state'] == BOUND and
                self._occupancies[position_five_prime]['state'] == BOUND):
            next_domain_vector = position_five_prime - position_three_prime
            # this includes the possibility that the five prime vector is
            # opposite to the next domain vector, which I'm not sure is possible
            # given the other rules. for now this is safe
            if all(next_domain_vector == np.abs(orientation_three_prime)):
                same_helix = False
            else:
                same_helix = True

        else:
            same_helix = False

        return same_helix

    def domains_have_correct_twist(self, chain_index, domain_index_1,
            domain_index_2):
        # this should only be applied to domains in the same helix

        # Ensure domain 1 is 3'
        if domain_index_2 < domain_index_1:
            domain_i_three_prime = domain_index_2
            domain_i_five_prime = domain_index_1

        else:
            domain_i_three_prime = domain_index_1
            domain_i_five_prime = domain_index_2

        position_three_prime = (
                self._positions[chain_index][domain_i_three_prime])
        position_five_prime = (
                self._positions[chain_index][domain_i_five_prime])

        orientation_three_prime = (
                self._orientations[chain_index][domain_i_three_prime])
        orientation_five_prime = (
                self._orientations[chain_index][domain_i_five_prime])

        # The PeriodicPosition object doesn't work with np.abs yet
        next_domain_vector = list(position_five_prime - position_three_prime)
        orientation_three_prime_rotated = (
                rotate_vector_quarter(orientation_three_prime,
                        next_domain_vector, -1))
        if orientation_three_prime_rotated == orientation_five_prime:
            twist_obeyed = True
        else:
            twist_obeyed = False

        return twist_obeyed


class OutputFile:

    def check_and_write(self, origami_system, step):
        if value_is_multiple(step, self._config_write_freq):
            self._write_configuration(self, origami_system, step)
        else:
            pass


class JSONOutputFile(OutputFile):

    def __init__(self, filename, origami_system, config_write_freq):
        json_origami = {'origami':{'identities':{}, 'configurations':[]}}
        json_origami['origami']['identities'] = origami_system.identities
        json_origami['origami']['sequences'] = origami_system.identities

        self._filename = filename
        self._config_write_freq = config_write_freq
        self.json_origami = json_origami

    def _write_configuration(self, origami_system, step):
        self.json_origami['origami']['configurations'].append({})
        self.json_origami['origami']['configurations'][-1]['step'] = step
        for chain in origami_system.chains:
            self.json_origami['origami']['configurations'][-1]['chains'] = (
                    [])
            self.json_origami['origami']['configurations'][-1]['chains'][-1]['identity'] = (
                    chain['identity'])
            self.json_origami['origami']['configurations'][-1]['chains'][-1]['index'] = (
                    chain['index'])
            self.json_origami['origami']['configurations'][-1]['chains'][-1]['positions'] = (
                    chain['positions'])
            self.json_origami['origami']['configurations'][-1]['chains'][-1]['orientations'] = (
                    chain['orientations'])

            json.dump(self.json_origami, open(self._filename, 'w'), indent=4,
                    seperators=(',', ': '))

    def close(self):
        pass


class HDF5OutputFile(OutputFile):

    def __init__(self, filename, origami_system, config_write_freq):
        self.hdf5_origami = h5py.File(filename, 'w')
        self.hdf5_origami.create_group('origami')
        self.hdf5_origami.attrs['identities'] = origami_system.identities
        self.hdf5_origami.attrs['sequences'] = origami_system.sequences

        self.hdf5_origami.create_group('origami/configurations')
        for chain in origami_system.chains:
            self._create_chain(chain)

        self.filename = filename
        self._config_write_freq = config_write_freq
        self._writes = 0

    def _write_config(self, origami_system, step):
        write_index = self._writes
        self._writes += 1
        for chain in origami_system.chains:
            chain_index = chain['index']
            database_key = 'origami/configurations/{}'.format(chain_index)
            try:
                self.hdf5_origami[database_key]
            except KeyError:
                self._create_chain(chain)

            self.hdf5_origami[database_key + '/step'].resize(self._writes, axis=0)
            self.hdf5_origami[database_key + '/step'][write_index] = step
            self.hdf5_origami[database_key + '/positions'].resize(self._writes, axis=0)
            self.hdf5_origami[database_key + '/positions'][write_index] = chain['positions']
            self.hdf5_origami[database_key + '/orientations'].resize(self._writes, axis=0)
            self.hdf5_origami[database_key + '/orientations'][write_index] = chain['orientations']

    def _create_chain(self, chain):
        chain_length = len(chain['positions'])
        chain_index = chain['index']
        database_key = 'origami/configurations/{}'.format(chain_index)
        self.hdf5_origami.create_group(database_key)
        self.hdf5_origami[database_key].attrs['index'] = (
                chain_index)
        self.hdf5_origami[database_key].attrs['identity'] = (
                chain['identity'])
        self.hdf5_origami.create_dataset(database_key + '/step',
                (1, 1),
                chunks=(1, 1),
                maxshape=(None, 1),
                dtype='i')
        self.hdf5_origami.create_dataset(database_key + '/positions',
                (1, chain_length, 3),
                chunks=(1, chain_length, 3),
                maxshape=(None, chain_length, 1),
                compresion='gzip',
                dtype='i')
        self.hdf5_origami.create_dataset(database_key + '/orientations',
                chain_length=len(chain['positions'])
                (1, chain_length, 3),
                chunks=(1, chain_length, 3),
                maxshape=(None, chain_length, 1),
                compresion='gzip',
                dtype='i')

    def close(self):
        pass


class JSONInputFile:

    def __init__(self, filename):
        json_origami = json.load(open(filename))

        self._filename = filename
        self._json_origami = json_origami

    @property
    def identities(self):
        return self._json_origami['origami']['identities']

    @property
    def sequences(self):
        return self._json_origami['origami']['sequences']

    def chains(self, step):
        chains = []
        for chain in self._json_origami['origami']['configurations'][step]:
            chain.append(chain)

        return chains


class HDF5InputFile:

    def __init__(self, filename):
        hdf5_origami = h5py.File(filename, 'r')

        self._filename = filename
        self._hdf5_origami = hdf5_origami

    @property
    def identities(self):
        return self._hdf5_origami['origami'].attrs['identities'].tolist()

    @property
    def sequences(self):
        return self._hdf5_origami['origami'].attrs['sequences'].tolist()

    def chains(self, step):
        chains = []
        for chain_database in self._hdf5_origami['origami/configurations']:
            if step in chain_database['steps']:
                chain = {}
                chain['index'] = chain_database.attrs['index']
                chain['identity'] = chain_database.attrs['identity']
                chain['positions'] = chain_database['positions'].tolist()
                chain['orientations'] = chain_database['orientations'].tolist()
                chains.append(chain)
            else:
                pass

        return chains


class GCMCBoundStaplesSimulation:

    def __init__(self, origami_system, move_settings, output_file):

        # Create cumalative probability distribution for movetypes
        # List to associate movetype with index in distrubution
        movetypes = []
        movetype_probabilities = []
        cumulative_probability = 0
        for movetype, probability in move_settings.items():
            movetypes.append(movetype)
            cumulative_probability += probability
            movetype_probabilities.append(cumulative_probability)

        # Check movetype probabilities are normalized
        # This could break from rounding errors
        if cumulative_probability != 1:
            print('Movetype probabilities not normalized')
            sys.exit()

        self._accepted_system = origami_system
        self._trial_system = origami_system
        self._max_dimension = origami_system.max_dimension
        self._output_file = output_file
        self._movetypes = movetypes
        self._movetype_probabilities = movetype_probabilities
        self._delta_e = 0

    def run(self, num_steps):
        self._delta_e = 0
        self._trial_system = self._accepted_system
        for step in range(num_steps):
            movetype = self._select_movetype()
            if movetype == MOVETYPE.INSERT_STAPLE:
                self._insert_staple()
            elif movetype == MOVETYPE.DELETE_STAPLE:
                self._delete_staple()
            #elif movetype == MOVETYPE.TRANSLATE_STAPLE:
            #    self._translate_staple()
            #elif movetype == MOVETYPE.ROTATE_STAPLE:
            #    self._rotate_staple()
            elif movetype == MOVETYPE.REGROW_STAPLE:
                self._regrow_staple()
            elif movetype == MOVETYPE.REGROW_SCAFFOLD_AND_BOUND_STAPLES:
                self._regrow_scaffold_and_bound_staples()
            elif movetype == MOVETYPE.ROTATE_ORIENTATION_VECTOR:
                self._rotate_orientation_vector()

            self._output_file.check_and_write(self._accepted_system, step)

    def _select_movetype(self):
        random_number = random.random()
        lower_boundary = 0
        for movetype_index, upper_boundary in enumerate(
                self._movetype_probabilities):
            if lower_boundary <= random_number < upper_boundary:
                movetype = self._movetypes[movetype_index]
                break
            else:
                lower_boundary = upper_boundary

        return movetype

    def _test_acceptance(self, ratio):
        p_accept = min(1, ratio)
        if p_accept == 1:
            accept = True
        else:
            if p_accept >= random.random():
                accept = True
            else:
                accept = False

        return accept

    def _configuration_accepted(self):
        T = self._accepted_system.temp
        boltz_factor = math.exp(self._delta_e / T)
        return self._test_acceptance(boltz_factor)

    def _staple_insertion_accepted(self):
        T = self._accepted_system.temp
        boltz_factor = math.exp(self._delta_e / T)
        number_density = self._accepted_system.staple_p
        return self._test_acceptance(number_density * boltz_factor)

    def _staple_deletion_accepted(self):
        T = self._accepted_system.temp
        boltz_factor = math.exp(self._delta_e / T)
        inverse_number_density = 1 / self._accepted_system.staple_p
        return self._test_acceptance(inverse_number_density * boltz_factor)

    def _grow_staple(self, staple_length, staple_index, domain_index):

        # Grow in five-prime direction
        staple_indices = range(domain_index, staple_length)
        try:
            self._regrow_chain(staple_index, staple_indices)
        except MoveRejection:
            raise

        # Grow in three-prime direction
        staple_indices = range(domain_index, -1, -1)
        try:
            self._regrow_chain(staple_index, staple_indices)
        except MoveRejection:
            raise

    def _insert_staple(self):

        # Randomly select staple identity
        staple_identity, domain_identities = self._accepted_system.random_staple_identity()
        staple_length = len(domain_identities)

        # Create filler positions and orientations and add chain
        positions = []
        for i in range(staple_length):
            positions.append(PeriodicPosition(self._max_dimension, [0, 0, 0]))

        orientations = [[0, 0, 0]] * staple_length
        self._trial_system.add_chain(staple_identity, positions, orientations)
        staple_index = len(self._trial_system.chain_lengths) - 1

        # Randomly select staple and scaffold domains to bind
        scaffold_index = 0
        scaffold_length = self._accepted_system.chain_lengths[scaffold_index]
        staple_domain = random.randrange(staple_length)
        scaffold_domain = random.randrange(scaffold_length)
        scaffold_domain_occupancy = self._trial_system.get_domain_occupancy(
                scaffold_index, scaffold_domain)

        # Reject if scaffold domain already bound
        if scaffold_domain_occupancy == BOUND:
            return

        # Attempt binding
        try:
            self._attempt_binding(staple_index, staple_domain)
        except MoveRejection:
            return

        # Grow staple
        try:
            self._grow_staple(staple_length, staple_index, staple_domain)
        except MoveRejection:
            return

        # Test acceptance
        if self._staple_insertion_accepted():
            self._accepted_system = self._trial_system
        else:
            pass

    def _delete_staple(self):

        # Randomly select staple
        staple_index = random.randrange(len(self._accepted_system.indices))
        staple_length = self._accepted_system.chain_lengths[staple_index]

        # Find all bound domains and subtract energies
        for domain_index in range(staple_length):
            occupancy_state = self._accepted_system.get_domain_occupancy(
                    staple_index, domain_index)
            if occupancy_state == BOUND:
                energy = self._accepted_system.get_hybridization_energy(
                        staple_index, domain_index)
                self._delta_e -= energy
            else:
                pass

        # Test acceptance
        if self._staple_deletion_accepted():
            self._trial_system.delete_chain(staple_index)
            self._accepted_system = self._trial_system
        else:
            pass

    def _regrow_staple(self):

        # Randomly select staple
        staple_index = random.randrange(1, len(self._accepted_system.indices))
        staple_length = self._accepted_system.chain_lengths[staple_index]

        # Find all bound domains and randomly select growth point
        bound_staple_domains = []
        for staple_domain_index in range(staple_length):
            occupancy_state = self._accepted_system.get_domain_occupancy(
                    staple_index, staple_domain_index)
            if occupancy_state == BOUND:
                bound_staple_domains.append(staple_domain_index)
            else:
                pass

        random_index = random.randrange(len(bound_staple_domains))
        starting_domain_index = bound_staple_domains[random_index]

        # Subtract unbound domain energies
        del bound_staple_domains[random_index]
        for staple_domain_index in bound_staple_domains:
            energy = self._accepted_system.get_hybridization_energy(
                staple_index, starting_domain_index)
            self._delta_e -= energy

        # Grow staple
        try:
            self._grow_staple(staple_length, staple_index, staple_domain_index)
        except MoveRejection:
            return

        # Test acceptance
        if self._configuration_accepted():
            self._accepted_system = self._trial_system
        else:
            self._trial_system = self._accepted_system

    def _attempt_binding(self, chain_index, domain_index):
        """Attempt to bind domain in trial position to domain in accepted position"""
        # Consider making an origami method called by set_position

        # Test if complimentary (and has correct orientation for binding)
        position = self._trial_system.get_domain_position(chain_index, domain_index)
        occupying_domain = self._trial_system.get_domain_at_position(position)
        occupying_chain_index = occupying_domain[0]
        occupying_domain_index = occupying_domain[1]
        complimentary = self._trial_system.domains_match(chain_index,
                domain_index, occupying_chain_index, occupying_domain_index)
        if not complimentary:
            raise MoveRejection
        else:
            pass

        # Create list of contiguous domains to both domains involved in binding
        contiguous_domains = []

        # Redundant list for binding domains; makes iterating over contiguous
        # more convienient
        binding_domains = []

        # If 3' domain exists on trial chain, add
        three_prime_occupancy = self._trial_system.get_domain_occupancy(
                chain_index, domain_index - 1)

        if three_prime_occupancy == UNASSIGNED:
            contiguous_domains.append((chain_index, domain_index - 1))
            binding_domains.append((chain_index, domain_index))

        # If 5' domain exists on trial chain, add
        five_prime_occupancy = self._trial_system.get_domain_occupancy(
                chain_index, domain_index + 1)

        if five_prime_occupancy == UNASSIGNED:
            contiguous_domains.append((chain_index, domain_index + 1))
            binding_domains.append((chain_index, domain_index))

        # If 3' domain exists on occupying chain, add
        if occupying_domain_index != 0:
            contiguous_domains.append((occupying_chain_index,
                    occupying_domain_index - 1))
            binding_domains.append(occupying_domain)

        # If 5' domain exists on occupying chain, add
        if occupying_domain_index != self._trial_system.chain_lengths[occupying_chain_index]:
            contiguous_domains.append((occupying_chain_index,
                    occupying_domain_index + 1))
            binding_domains.append(occupying_domain)

        # For all contiguous domains, if they are part of the same helix, check
        # if the twist constraints are obeyed
        for i in enumerate(contiguous_domains):
            contiguous_chain_index = contiguous_domains[i][0]
            contiguous_domain_index = contiguous_domains[i][1]
            binding_domain_index = binding_domains[i][1]

            if self._trial_system.domains_part_of_same_helix(
                    contiguous_chain_index, binding_domain_index,
                    contiguous_domain_index):
                if self._trial_system.domains_have_correct_twist(
                        contiguous_chain_index, binding_domain_index,
                        contiguous_domain_index):
                    binding_successful = True

                    # Add new binding energies
                    energy = self._trial_system.get_hybridization_energy(
                        contiguous_chain_index, binding_domain_index)
                    self._delta_e += energy

                else:
                    raise MoveRejection
            else:
                raise MoveRejection

        return binding_successful

    def _regrow_chain(self, chain_index, domain_indices, binding_allowed=False):

        # Iterate through given indices, growing next domain from current index
        for domain_index in domain_indices[:-1]:

            # Randomly select neighbour lattice site for new position
            dimension = random.choice([XHAT, YHAT, ZHAT])
            direction = random.randrange(-1, 2, 2)

            # Position vector of previous domain
            r_prev = self._trial_system.get_domain_position(chain_index, domain_index)

            # Trial position vector
            r_new = PeriodicPosition(r_prev + direction * dimension,
                    self._max_dimension)

            # Randomly select new orientation
            dimension = random.choice([XHAT, YHAT, ZHAT])
            direction = random.randrange(-1, 2, 2)

            # Trial position orientation
            o_new = dimension * direction

            # Reject if position in bound state or binding not allowed
            occupancy = self._trial_system.get_position_occupancy(r_new)
            if occupancy == BOUND or not binding_allowed:
                raise MoveRejection
            else:
                pass

            # Store positions and orientations
            self._trial_system.set_domain_position(chain_index, domain_index,
                    r_new)
            self._trial_system.set_domain_orientation(chain_index,
                    domain_index, o_new)

            # Attempt binding if position occupied in unbound state
            if occupancy == UNBOUND:
                current_domain_index = domain_index + 1
                try:
                    self._attempt_binding(current_domain_index, chain_index)
                except MoveRejection:
                    raise

            # Continue if site unoccupied
            else:
                pass

        return

    def _regrow_scaffold_and_bound_staples(self):

        # Randomly select starting scaffold domain
        scaffold_index = 0
        scaffold_length = self._accepted_system.chain_lengths[scaffold_index]
        start_domain_index = random.randrange(scaffold_length)

        # Select direction to regrow, create index list
        direction = random.randrange(2)
        if direction == 1:
            scaffold_indices = range(start_domain_index, scaffold_length)
        else:
            scaffold_indices = range(scaffold_length, -1, -1)

        # Update trial system occupancies
        for scaffold_domain_index in scaffold_indices:
            self._trial_system.remove_occupancy(scaffold_index,
                    scaffold_domain_index)

        # Regrow scaffold
        try:
            self._regrow_chain(scaffold_index, scaffold_indices)
        except MoveRejection:
            return

        # Find all staples bound to scaffold (includes repeats)
        staples = []
        unique_staple_indices = set()
        for domain_index in scaffold_indices:
            bound_domain = self._trial_system.get_bound_domain(scaffold_index, domain_index)
            if bound_domain == ():
                continue
            else:
                staples.append(bound_domain)
                unique_staple_indices.add(bound_domain[0])

        # Find all repeated staples and pick domain on scaffold to grow from
        for staple_index in unique_staple_indices:
            repeated_staple_indices = []
            for staple, i in enumerate(staples):
                if staple[0] == staple_index:
                    repeated_staple_indices.append(i)
                else:
                    pass

            selected_staple_i = random.choice(repeated_staple_indices)

            # Test if selected scaffold domain blocked
            staple = staples[selected_staple_i]
            occupancy = self._trial_system.get_domain_occupancy(*staple)
            if occupancy == BOUND:
                return
            else:
                pass

            for repeated_staple_index in repeated_staple_indices:
                del staples[repeated_staple_index]

        # Subtract all staple binding energies to other parts of the scaffold
        for staple_index in unique_staple_indices:
            for staple_domain_index in range(self._trial_system.chain_lengths[staple_index]):
                occupancy = self._accepted_system.get_domain_occupancy(
                        staple_index, staple_domain_index)
                staple = (staple_index, staple_domain_index)
                if occupancy == BOUND and staple not in staples:
                    energy = self._accepted_system.get_hybridization_energy(
                        staple_index, staple_domain_index)
                    self._delta_e -= energy
                else:
                    pass

        # Grow staples
        for staple_index in unique_staple_indices:
            staple_chain_index = staple[0]
            staple_domain_index = staple[1]
            staple_length = self._trial_system.chain_lengths[staple_chain_index]

            try:
                self._grow_staple(staple_length, staple_index,
                        staple_domain_index)
            except MoveRejection:
                return

        # Test acceptance
        if self._configuration_accepted():
            self._accepted_system = self._trial_system
        else:
            pass

    def _rotate_orientation_vector(self):

        # Select random chain and domain
        chain_lengths = self._accepted_system
        chain_index = random.randrange(len(chain_lengths))
        domain_index = random.randrange(chain_lengths[chain_index])

        # Reject if in bound state
        occupancy = self._accepted_system.get_domain_occupancy(chain_index,
                domain_index)
        if occupancy == BOUND:
            return
        else:
            pass

        # Select random orientation and update
        dimension = random.choice([XHAT, YHAT, ZHAT])
        direction = random.randrange(-1, 2, 2)
        o_new = dimension * direction
        self._accepted_system.set_domain_orientation(chain_index, domain_index,
                o_new)

#    def _rotate_contiguous_helix_orientation_vectors(self):
#        pass

        # Select random chain and domain

        #

#    def _center(self):

#    def swap_staples:
#        # Select start domains for each staple
#        staple_starting_domains = []
#        staple_starting_scaffold_domains = []
#        for staple_index in staple_indices:
#
#            # Select domain on staple
#            staple_domain_index = random.randrange(chain_lengths[staple_index])
#            staple_starting_domains.append(staple_domain_index)
#
#            # Find all complimentary domains
#            complimentary_domains = []
#            for domain_index in scaffold_indices:
#                if self._origami_system.domain_match((staple_index, staple_domain_index),
#                        (scaffold_index, domain_index)):
#                    complimentary_domains.append(domain_index)
#
#            # Randomly select complimentary domain
#            complimentary_domain = random.choice(complimentary_domains)
#            staple_starting_scaffold_domains.append(complimentary_domain)
#

class GCMCSimulation:
    pass


#class GCMCBoundStaplesSimulation:
#    pass
    #def _insert_bound_staple(self, chain_index, domain_index, identity):
    #    check if domains match and have correct orientations
    #    build staple orientation with randomly chosen positions and orientations
    #    check for overlaps
    #    check if overlaps matching and in correct orientation
    #    if so, check if prexisiting helix, if so check constraints
    #    if overlap or twist constraints violated, reject
    #    check if adding to prexisting helix, if so check twist constraints
    #    reject if twist constraints violated
    #    test acceptance
    #    if accepted, add chain
#
    #def _delete_bound_staple

class GCMCFreeStaplesSimulation:
    def _translate_staple(self):
        pass

    def _rotate_staple(self):
        pass
