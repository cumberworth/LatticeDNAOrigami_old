#!/usr/env python

"""Run MC simulations of the lattice domain-resolution DNA origami model.

Data:

Functions:

Classes:

"""

import json
import hdf5
import numpy as np
from enum import Enum
import random


UNBOUND = 1
BOUND = 2
XHAT = np.array([1, 0, 0])
YHAT = np.array([0, 1, 0])
ZHAT = np.array([0, 0, 1])


class MOVETYPE(Enum):
    INSERT_STAPLE = 0
    DELETE_STAPLE = 1
    TRANSLATE_STAPLE = 2
    ROTATE_STAPLE = 3
    REGROW_STAPLE = 4
    REGROW_SCAFFOLD_AND_BOUND_STAPLES = 5


def value_is_multiple(value, multiple):
    value_is_multiple = False
    try:
        if value % multiple == 0:
            value_is_multiple = True
        else:
            pass

    except ZeroDivisionError:
        value_is_multiple = False

    return value_is_multiple


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

# Kcal/mol/K
NN_ENTROPY
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


def calc_hybridization_energy(sequence, T):
    complimentary_sequence = complimentary_sequence(sequence)

    # Intiation free energy
    DG_init = NN_ENTHALPY['INITIATION'] - T * NN_ENTROPY['INITIATION']

    # Symmetry penalty for palindromic sequences
    if sequence_is_palindromic(sequence):
        DG_sym = - T * NN_ENTROPY['SYMMETRY_CORRECTION']
    else:
        DG_sym = 0

    # NN pair energies
    DG_stack = 0
    for base_index in range(len(sequence), 2):
        first_pair = sequence[base_index: base_index + 1]
        second_pair = complimentary_sequence[base_index: base_index + 1]
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
        if sequence[0] in ['A', 'T']:
            terminal_AT_pairs += 1

    if terminal_AT_pairs > 0:
        DG_at = (NN_ENTHALPY['TERMINAL_AT_PENALTY'] -
            T * NN_ENTROPY['TERMINAL_AT_PENALTY']) * terminal_AT_pairs
    else:
        DG_at = 0

    DG_hybridization = DG_init + DG_sym + DG_stack + DG_at

    return DG_hybridization


COMPLIMENTARY_BASE_PAIRS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


def complimentary_sequence(sequence):
    complimentary_sequence = []
    for base in sequence:
        complimentary_sequence.append(COMPLIMENTARY_BASE_PAIRS[base])

    complimentary_sequence = ''.join(complimentary_sequence)
    return complimentary_sequence


def sequence_is_palindromic(sequence):
    complimentary_sequence = complimentary_sequence(sequence)
    reverse_complimentary_sequence = complimentary_sequence[::-1]
    if reverse_complimentary_sequence == sequence:
        palindromic = True
    else:
        palindromic = False

    return palindromic


# somehow use decorators to modify creation of positions so can select
# periodic or nonperiodic boundary conditions
class PeriodicPositions:

    def __init__(self, max_dimension, position):
        self._position = np.array(position)
        for component_index, component in enumerate(self._position):
            self._wrap_component(component_index, component)

    def __getitem__(self, index):
        return self._position[index]

    def __setitem__(self, index, coordinate):
        self._wrap_component(index, coordinate)

    def __add__(self, other_perdiodic_position):
        absolute_sum = self._position + other_perdiodic_position._position
        return PeriodicPosition(asbolute_sum)

    def __sub__(self, other_periodic_position):
        absolute_dif = self._position - other_perdiodic_position._position
        return PeriodicPosition(asbolute_dif)

    def __eq__(self, other_periodic_position):
        return all(self._position == other_periodic_position._position)

    def __iter__(self):
        return self._position.tolist().__iter__

    def _wrap_component(self, component):
        if component > max_dimension:
            self._position[component_index] = (-max_dimension +
                (component - max_dimension))
        elif component < max_dimension:
            self._position[component_index] = (max_dimension +
                (component - max_dimension))
        else:
            pass


class OrigamiSystem:

    def __init__(self, input_file, step, max_dimension):

        # Set configuration from specified inputfile and step
        indices = []
        chain_identities = []
        positions = []
        occupancies = {}
        orientations = []

        for chain in input_file.chains(step):
            indices.append(chain['index'])
            chain_identities.append(chain['identity'])
            positions = []
            for position in chain['positions']:
                position = PeriodicPosition(max_dimension, position)
                positions.append(position)

            positions.append(positions)
            for domain, position in enumerate(positions[-1]):
                occupancies[tuple(position)] = UNBOUND

            orientations.append(chain['orientations'])

        # Calculate constants
        volume = (max_dimension * 2) ** 3
        scaffold_length = len(positions[0])

        # Calculate and store hybridization energies
        sequences = input_file.sequences
        e
        for sequence in sequences:
            calc_hybridization_energy(sequence)

        # Set instance variables
        self.identities = input_file.identities
        self.sequences = sequences
        self.max_dimension = max_dimension
        self.volume = volume
        self.scaffold_length = scaffold_length

        self._chain_identity = identity
        self._indices = indices
        self._identities = identities
        self._positions = positions
        self._occupancies = occupancies
        self._orientations = orientations
        self._current_chain_index = max(indices)

    @property
    def chains():
	chains = []
        for working_index, unique_index in enumerate(self._indices):
            chain = {}
            chain['index'] = unique_index
            chain['identity'] = self._identities[working_index]
            positions = []
            for position in self._positions[working_index]:
                position = list(position)
                positions.append(position)

            chain['positions'] = positions
            orientations = []
            for orientation in self._orientations[working_index]:
                orienation = orientation.tolist()
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
        if self._occupancies[previous_domain_position] == BOUND:
            self._occupancies[previous_domain_position] = UNBOUND
        else:
            del self._occupancies[tuple(previous_domain_position)]

        self._positions[chain_index][domain_index] = position
        position = tuple(position)
        # assumes that the position is not already in a bound state
        try:
            self._occupancies[position] = BOUND

        except KeyError:
            self._occupancies[position] = UNBOUND

    def get_domain_orientation(self, chain_index, domain_index):
        return self._orientations[chain_index, domain_index]

    def set_domain_orientation(self, chain_index, domain_index, orientation)
        self._orientations[chain_index, domain_index] = orientation

    def add_chain(self, identity, positions, orientations):
        chain_index = len(self._positions)
        self._identities.append(identity)
        self._positions.append(positions)
        for domain_index, position in enumerate(positions):
            try:
                self._occupancies[position] = BOUND

            except KeyError:
                self._occupancies[position] = UNBOUND

        self._positions.append(orientations)
        self._current_chain_index += 1
        self._indices.append(self._current_chain_index)

    def delete_chain(self, chain_index):
        del self._indices[chain_index]
        del self._identities[chain_index]
        positions = self._positions[chain_index]
        for position in positions:
            position = tuple(position)
            if self._occupancies[position] == BOUND:
                self._occupancies[position] == UNBOUND
            else:
                del self._occupancies[tuple(position)]

        del self._positions[chain_index]
        del self._orientations[chain_index]

    def position_occupied(self, position):
        overlap = False
        if tuple(position) in self._domain_hash:
            overlap = True
        else:
            pass
        
        return overlap

    def domains_match(chain_index_1, domain_index_1,
            chain_index_2, domain_index_2):
        chain_identity_1 = self._identities[chain_index_1]
        domain_identity_1 = self._identity_dict[chain_identity_1][domain_index_1]
        chain_identity_2 = self._identities[chain_index_2]
        domain_identity_2 = self._identity_dict[chain_identity_1][domain_index_2]
        complimentary = domain_identity_1 + domain_identity_2
        if complimentary == 0:
            orientation_1 = self._orientations[chain_index_1][domain_index_1]
            orientation_2 = self._orientations[chain_index_2][domain_index_2]
            complimentary_orientations = orientation_1 + orientation_2
            if complimentary_orientations.sum() == 0:
                match = True
            else:
                match = False
        else:
            match = False

        return match

    def domains_part_of_same_helix(chain_index, domain_index_1,
            domain_index_2, new_position_1=(), new_position_2=(),
            new_orientation_1=()):
        if new_position_1 == ():
            position_1 = self._positions[chain_index][domain_index]
        else:
            position_1 = new_position_1

        if new_position_2 == ():
            position_2 = self._positions[chain_index][domain_index]
        else:
            position_2 = new_position_2
        
        if (self._occupancies[position_1] == BOUND and
                self._occupancies[position_2] == BOUND):
            if new_orientation_1 == ():
                five_prime_vector = self._orientations[chain_index][domain_index_1]
            else:
                five_prime_vector = new_orientation_1
        
            next_domain_vector = position_2 - position_1
            # this includes the possibility that the five prime vector is
            # opposite to the next domain vector, which I'm not sure is possible
            # given the other rules. for now this is safe
            if all(next_domain_vector == np.abs(five_prime_vector)):
                same_helix = False
            else:
                same_helix = True

        else:
            same_helix = False

        return same_helix

    def domains_have_correct_twist(chain_index, domain_index_1,
            domain_index_2, new_position_1=(), new_position_2=(),
            new_orientation_1=(), new_orientation_2=()):
        # this should only be applied to domains in the same helix
        if new_position_1 == ():
            position_1 = self._positions[chain_index][domain_index]
        else:
            position_1 = new_position_1

        if new_position_2 == ():
            position_2 = self._positions[chain_index][domain_index]
        else:
            position_2 = new_position_2
        
        if new_orientation_1 == ():
            orientation_1 = self._orientations[chain_index][domain_index]
        else:
            orientation_1 = new_orientation_1

        if new_orientation_2 == ():
            position_2 = self._orientations[chain_index][domain_index]
        else:
            position_2 = new_orientation_2
        
        next_domain_vector = position_2 - position_1
	orientation_1_rotated = (
		rotate_vector(orientation_1, next_domain_vector, -1))
	if orientation_1_rotated == orientation_2:
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

        self.filename = filename
        self._config_write_freq = config_write_freq
        self.json_origami = json_origami

    def _write_configuration(self, origami_system, step):
        self.json_origami['origami']['configurations'].append({})
        self.json_origami['origami']['configurations'][-1]['step'] = step
        for chain in origami.chains:
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
        for chain_index, chain in enumerate(origami_system.chains):
            self._create_chain(chain)

        self.filename = filename
        self.write_config_freq = write_config_freq
        self.writes = 0

    def _write_config(self, origami_system, step):
        write_index = self.writes
        self.writes += 1
        for chain in origami_system.chains:
            chain_index = chain['index']
            database_key = 'origami/configurations/{}'.format(chain_index)
            try:
                self.hdf5_origami[database_key]
            except KeyError:
                self._create_chain(chain)

            self.hdf5_origami[database_key + '/step'].resize(self.writes, axis=0)
            self.hdf5_origami[database_key + '/step'][write_index] = step
            self.hdf5_origami[database_key + '/positions'].resize(self.writes, axis=0)
            self.hdf5_origami[database_key + '/positions'][write_index] = chain['positions']
            self.hdf5_origami[database_key + '/orientations'].resize(self.writes, axis=0)
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
                chain_length = len(chain['positions'])
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
        return self.hdf5_origami['origami'].attrs['identities'].tolist()

    @property
    def sequences(self):
        return self.hdf5_origami['origami'].attrs['sequences'].tolist()

    def chains(self, step):
        chains = []
        for chain_database in self._hdf5_origami['origami/configurations']:
            if step in chain_database['steps']:
                chain = {}
                chain['index'] = chain_database.attrs['index']
                chain['identity'] = chain_database.attrs['identity']
                chain['positions'] = chain_database['positions'].tolist()
                chain['orientations'] = chain_databse['orientations'].tolist()
                chains.append(chain)
            else:
                pass

        return chains


class GCMCFreeStaplesSimulation()

    def __init__(self, origami_system, movetype_settings, output_file):

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

        self._origami_system = origami_system
        self._max_dimension = origami_system.max_dimension
        self._output_file = output_file
        self._movetypes = movetypes
        self._movetype_probabilities = movetype_probabilities

    def run(self, num_steps):
        for step in range(num_steps):
            movetype = self._select_movetype()
            if movetype == MOVETYPE.INSERT_STAPLE:
                self._insert_staple
            elif movetype == MOVETYPE.DELETE_STAPLE:
                self._delete_staple
            elif movetype == MOVETYPE.TRANSLATE_STAPLE:
                self._translate_staple
            elif movetype == MOVETYPE.ROTATE_STAPLE:
                self._rotate_staple
            elif movetype == MOVETYPE.REGROW_STAPLE:
                self._regrow_staple
            elif movetype == MOVETYPE.REGROW_SCAFFOLD_AND_BOUND_STAPLES:
                self._regrow_scaffold_and_bound_staples
            elif movetype == MOVETYPE.ROTATE_ORIENTATION_VECTOR:
                self._rotate_orientation_vector

            self._output_file.check_and_write(self._origami_system, step)

    def _select_movetype(self):
        random_number = random.uniform(0, 1)
        lower_boundary = 0
        for movetype_index, upper_boundary in enumerate(
                self._movetype_probabilities):
            if lower_boundary <= random_number < upper_boundary:
                movetype = self._movetypes[movetype_index]
                break
            else:
                lower_boundary = upper_boundary

        return movetype


    def _configuration_accepted

    def _staple_insertion_accepted

    def _staple_deletion_accepted

    def _insert_staple

    def _delete_staple

    def _translate_staple

    def _rotate_staple

    def _regrow_staple

    def _regrow_chain_and_bound_staples(self):

        # Randomly select scaffold domain
        scaffold_length = self_origami_system.scaffold_length
        scaffold_index = random.randrange(scaffold_length)

        # Select shortest direction to end of scaffold
        if scaffold_index <= (scaffold_length / 2 - 1):
            scaffold_indices = range(scaffold_index - 1, -1, -1)
        else:
            scaffold_indices = range(scaffold_index + 1, scaffold_length)

        # Select starting domains for each staple

        # Grow scaffold
        positions = []
        for scaffold_index in scaffold_indices:

            # Randomly select neighbour lattice site
            dimension = random.choice([XHAT, YHAT, ZHAT])
            direction = random.randrange(-1, 2, 2)

            # Position vector of previous domain
            r_prev = self._origami_system.get_domain_position(0, scaffold_index)

            # Trial position vector
            r_new = PeriodicVector(r_prev + direction * dimension,
                    self._max_dimension)
            
            # Test if position occupied
            if self._origami_system.position_occupied(r_new):
                accept = False
                break

        # Grow staples

    def _rotate_orientation_vector

    def _center

class GCMCSimulation:
    pass


class GCMCBoundStaplesSimulation:
    pass
    #check center interval and center
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
    pass
