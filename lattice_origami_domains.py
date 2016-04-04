#!/usr/env python

"""Run MC simulations of the lattice domain-resolution DNA origami model."""

import json
import math
import sys
import random
import pdb
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


class ConstraintViolation(Exception):
    """Used for constraint violations in OrigamiSystem."""
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
    vector = np.copy(vector)
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
    Sequences are assumed to be 5' to 3'.
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


class OrigamiSystem:
    """Simple cubic lattice model of DNA origami at domain level resolution.

    The domains are 8 bp long. See reports/modelSpecs/domainResModelSpecs for
    exposition on the model.

    I've used get methods instead of properties as properties don't take
    indices. I would have to index the internal structure directly, which
    I don't want to do.
    """

    def __init__(self, input_file, step, temp, staple_p):
        self.temp = temp
        self.staple_p = staple_p

        # Unique indices; important for writing trajectories to file
        self._indices = []

        # Indices to identities list for current chains
        self._chain_identities = []
        self._positions = []
        self._orientations = []
        self.chain_lengths = []

        # Dictionary with position keys and state values
        self._position_occupancies = {}

        # Dictionary with domain keys and state values
        self._domain_occupancies = {}

        # Dictionary with bound domain keys and values
        self._bound_domains = {}

        # Dictionary with position keys and unbound domain values
        self._unbound_domains = {}
        self.identities = input_file.identities
        self.sequences = input_file.sequences

        # Calculate and store hybridization energies
        self._hybridization_energies = []
        for sequence in self.sequences:
            energy = calc_hybridization_energy(sequence, temp)
            self._hybridization_energies.append(energy)

        # Set configuration to specified input file step
        for chain_index, chain in enumerate(input_file.chains(step)):
            self._indices.append(chain['index'])
            identity = chain['identity']
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
                if (position - previous_position).sum() > 1:
                    raise ConstraintViolation

                self.set_domain_configuration(chain_index, domain_index,
                        position, orientation)
            
                previous_position = np.array(chain['positions'][domain_index])

        # Keep track of unique chain index
        self._current_chain_index = max(self._indices)

    @property
    def chains(self):
        """Standard format for passing chain configuration."""
        chains = []
        for working_index, unique_index in enumerate(self._indices):
            chain = {}
            chain['index'] = unique_index
            chain['identity'] = self._chain_identities[working_index]
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

        try:
            occupancy = self._domain_occupancies[(chain_index, domain_index)]
        except KeyError:
            occupancy = UNASSIGNED

        return occupancy

    def get_bound_domain(self, chain_index, domain_index):
        """Return domain bound to given domain, otherwise return empty tuple.

        Consider failing instead to be consistent with get_unbound_domain.
        """
        try:
            domain = self._bound_domains[(chain_index, domain_index)]
        except KeyError:
            domain = ()

        return domain

    def get_random_staple_identity(self):
        """Return random staple identity."""
        staple_identity = random.randrange(1, len(self.identities))
        domain_identities = self.identities[staple_identity]
        return staple_identity, domain_identities

    def get_hybridization_energy(self, chain_index, domain_index):
        """Return precalculated hybridization energy."""

        chain_identity = self._chain_identities[chain_index]
        domain_identity = self.identities[chain_identity][domain_index]

        # Because identites start at 1
        energy_index = abs(domain_identity) - 1
        return self._hybridization_energies[energy_index]

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
        if occupancy == BOUND:

            # Note that exception raised before setting position. If the
            # exceptions are to be handled, either set positions here or
            # save and return previous positions for later exceptions
            raise ConstraintViolation
        else:
            pass

        # Update position now for ease of access (considering allowing reversion
        # if position rejected).
        self._positions[chain_index][domain_index] = position
        self._orientations[chain_index][domain_index] = orientation

        # Attempt binding if position occupied in unbound state
        if occupancy == UNBOUND:
            try:
                delta_e = self._bind_domain(*domain)
            except ConstraintViolation:
                raise
            else:

                # Update occupancies
                position = tuple(position)
                occupying_domain = self._unbound_domains[position]
                del self._unbound_domains[position]
                self._domain_occupancies[domain] = BOUND
                self._domain_occupancies[occupying_domain] = BOUND
                self._position_occupancies[position] = BOUND
                self._bound_domains[occupying_domain] = domain
                self._bound_domains[domain] = domain

        # Move to empty site and update occupancies
        else:
            position = tuple(position)
            self._domain_occupancies[domain] = UNBOUND
            self._position_occupancies[position] = UNBOUND
            self._unbound_domains[position] = domain

        return delta_e

    def set_domain_orientation(self, chain_index, domain_index, orientation):
        """Set domain orientation.

        There is a orientation setter and not an indepenent public position
        setter because a new orientation is always generated when a new position
        is generated (easier than somehow trying to keep the same relative
        orientation), while the orientation is set indepentnly, so not all the
        checks of set_domain_configuration are necessary anymore.
        """
        if self._domain_occupancies[(chain_index, domain_index)] == BOUND:
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
        domain = (chain_index, domain_index)
        occupancy = self._domain_occupancies
        if occupancy == BOUND:

            # Collect energy
            delta_e = -self.get_hybridization_energy(*domain)
            bound_domain = self._bound_domains[domain]
            position = tuple(self._positions[chain_index][domain_index])
            self._positions[chain_index][domain_index] = []
            del self._bound_domains[domain]
            del self._bound_domains[bound_domain]
            del self._domain_occupancies[domain]
            self._unbound_domains[position] = bound_domain
            self._position_occupancies[position] = UNBOUND
            self._domain_occupancies[bound_domain] = UNBOUND
        elif occupancy == UNBOUND:
            position = tuple(self._positions[chain_index][domain_index])
            self._positions[chain_index][domain_index] = []
            del self._unbound_domains[position]
            del self._position_occupancies[position]
            del self._domain_occupancies[domain]
        else:
            pass

        return delta_e

    def add_chain(self, identity):
        """Add chain with domains in unassigned state and return chain index."""
        self._current_chain_index += 1
        self._indices.append(self._current_chain_index)
        self._chain_identities.append(identity)
        chain_length = len(self.identities[identity])
        self._positions.append([[]] * chain_length)
        self._orientations.append([[]] * chain_length)
        self.chain_lengths.append(chain_length)
        chain_index = len(self.chain_lengths) - 1
        return chain_index

    def delete_chain(self, chain_index):
        """Delete chain."""

        # Change in energy
        delta_e = 0
        del self._indices[chain_index]
        del self._chain_identities[chain_index]

        for domain_index in range(self.chain_lengths[chain_index]):
            delta_e += self.unassign_domain(chain_index, domain_index)

        del self._positions[chain_index]
        del self._orientations[chain_index]
        del self.chain_lengths[chain_index]

        return delta_e

    def center(self):
        """Translates system such that first domain of scaffold on origin."""

        # Translation vector
        r_t = self._positions[0][0]
        for chain_index, chain_positions in enumerate(self._positions):
            for domain_index, chain_positions in enumerate(chain_positions):
                r_o = self._positions[chain_index][domain_index]
                r_n = r_o - r_t
                self._positions[chain_index][domain_index] = r_n

                # Update occupancies
                domain = (chain_index, domain_index)
                r_o = tuple(r_o)
                r_n = tuple(r_n)
                occupancy = self._domain_occupancies[domain]
                del self._position_occupancies[r_o]
                if occupancy == BOUND:
                    self._position_occupancies[r_n] = BOUND
                elif occupancy == UNBOUND:
                    self._position_occupancies[r_n] = UNBOUND
                    del self._unbound_domains[r_o]
                    self._unbound_domains[r_n] = domain

    def _bind_domain(self, trial_chain_index, trial_domain_index):
        """Bind given domain in preset trial config and return change in energy.
        """
        position = tuple(self._positions[trial_chain_index][trial_domain_index])

        # Test if complimentary (and has correct orientation for binding)
        try:
            occupying_domain = self._unbound_domains[position]
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
                    occupancy = self._domain_occupancies[neighbour_domain]
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
        delta_e = self.get_hybridization_energy(*neighbour_domain)

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
            if complimentary_orientations.sum() == 0:
                match = True
            else:
                match = False
        else:
            match = False

        #pdb.set_trace()
        return match

    def _helical_pair_constraints_obeyed(self, chain_index, domain_index_1,
            domain_index_2):
        """Return True if domains not in same helix or twist constraints obeyed.

        Does not check if they are in bound states.
        """

        # Ensure domain 1 is 5'
        if ((chain_index == 0 and domain_index_2 < domain_index_1) or
                (chain_index > 0 and domain_index_2 > domain_index_1)):
            five_p_index = domain_index_2
            three_p_index = domain_index_1

        else:
            five_p_index = domain_index_1
            three_p_index = domain_index_2

        position_five_p = (self._positions[chain_index][five_p_index])
        position_three_p = (self._positions[chain_index][three_p_index])

        orientation_five_p = (self._orientations[chain_index][five_p_index])
        orientation_three_p = (self._orientations[chain_index][three_p_index])

        next_domain_vector = position_three_p - position_five_p

        # Only one allowed configuration not in the same helix
        if all(next_domain_vector == np.abs(orientation_five_p)):
            constraints_obeyed = True

        # Check twist constraint if same helix
        else:
            orientation_five_p_r = (rotate_vector_quarter(orientation_five_p,
                    next_domain_vector, -1))
            if all(orientation_five_p_r == orientation_three_p):
                constraints_obeyed = True
            else:
                constraints_obeyed = False

        if (chain_index, domain_index_1) == (2, 0):
            pdb.set_trace()
        return constraints_obeyed


class OutputFile:
    """Base output file class to allow check_and_write to be shared."""

    def check_and_write(self, origami_system, step):
        """Check property write frequencies and write accordingly."""
        if value_is_multiple(step, self._config_write_freq):
            self._write_configuration(self, origami_system, step)
        else:
            pass


class JSONOutputFile(OutputFile):
    """JSON output file class."""

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
        """Perform cleanup."""
        pass


class HDF5OutputFile(OutputFile):
    """HDF5 output file class.

    Custom format; not compatable with VMD (not H5MD).
    """

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
        """Perform any cleanup."""
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


class HDF5InputFile:
    """Input file taking hdf5 formatted origami system files in constructor."""

    def __init__(self, filename):
        hdf5_origami = h5py.File(filename, 'r')

        self._filename = filename
        self._hdf5_origami = hdf5_origami

    @property
    def identities(self):
        """Standard format for passing origami domain identities."""
        return self._hdf5_origami['origami'].attrs['identities'].tolist()

    @property
    def sequences(self):
        """Standard format for passing origami system sequences.

        Only includes sequences of scaffold or strand, as energy calculation
        only requires one each of the complimentary strands for input.

        Is this 5' or 3'? For scaffold or staple?
        """
        return self._hdf5_origami['origami'].attrs['sequences'].tolist()

    def chains(self, step):
        """Standard format for passing chain configuration."""
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
    """GCMC sim for domain-res origami model with bound staples only.

    Grand cannonical Monte Carlo simulations on a simple cubic lattice of the
    origami model defined by the origami class and associated documentation. The
    simulations run with this class do not include any moves that can lead to
    free staples.

    No safety checks to ensure starting config has no free staples.
    """

    def __init__(self, origami_system, move_settings, output_file):
        self._output_file = output_file

        # Create cumalative probability distribution for movetypes
        # List to associate movetype method with index in distribution
        self._movetype_methods = []
        self._movetype_probabilities = []
        cumulative_probability = 0
        for movetype, probability in move_settings.items():

            # I still wonder if there is a way to avoid all the if statements
            if movetype == MOVETYPE.INSERT_STAPLE:
                movetype_method = self._insert_staple
            elif movetype == MOVETYPE.DELETE_STAPLE:
                movetype_method = self._delete_staple
            elif movetype == MOVETYPE.REGROW_STAPLE:
                movetype_method = self._regrow_staple
            elif movetype == MOVETYPE.REGROW_SCAFFOLD_AND_BOUND_STAPLES:
                movetype_method = self._regrow_scaffold_and_bound_staples
            elif movetype == MOVETYPE.ROTATE_ORIENTATION_VECTOR:
                movetype_method = self._rotate_orientation_vector

            cumulative_probability += probability
            self._movetype_methods.append(movetype_method)
            self._movetype_probabilities.append(cumulative_probability)

        # Check movetype probabilities are normalized
        # This could break from rounding errors
        if cumulative_probability != 1:
            print('Movetype probabilities not normalized')
            sys.exit()

        # Two copies of the system allows easy reversion upon rejection
        self._accepted_system = origami_system
        self._trial_system = origami_system

        # Change in energy for a proposed trial move
        self._delta_e = 0

        # Frequency for translating system back to origin
        self.center_freq = 1000

    def run(self, num_steps):
        """Run simulation for given number of steps."""
        self._delta_e = 0
        self._trial_system = self._accepted_system

        for step in range(num_steps):
            movetype_method = self._select_movetype()
            movetype_method()
            self._output_file.check_and_write(self._accepted_system, step)
            if step == self.center_freq:
                self._accepted_system.center()

    def _select_movetype(self):
        """Return movetype method according to distribution."""
        random_number = random.random()
        lower_boundary = 0
        for movetype_index, upper_boundary in enumerate(
                self._movetype_probabilities):
            if lower_boundary <= random_number < upper_boundary:
                movetype_method = self._movetype_methods[movetype_index]
                break
            else:
                lower_boundary = upper_boundary

        return movetype_method

    def _test_acceptance(self, ratio):
        """Metropolis acceptance test for given ratio."""
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
        """Metropolis acceptance test for configuration change."""
        T = self._accepted_system.temp
        boltz_factor = math.exp(self._delta_e / T)
        return self._test_acceptance(boltz_factor)

    def _staple_insertion_accepted(self):
        """Metropolis acceptance test for particle insertion."""
        T = self._accepted_system.temp
        boltz_factor = math.exp(self._delta_e / T)
        number_density = self._accepted_system.staple_p
        return self._test_acceptance(number_density * boltz_factor)

    def _staple_deletion_accepted(self):
        """Metropolis acceptance test for particle deletion."""
        T = self._accepted_system.temp
        boltz_factor = math.exp(self._delta_e / T)
        inverse_number_density = 1 / self._accepted_system.staple_p
        return self._test_acceptance(inverse_number_density * boltz_factor)

    # Following are helper methods to the top level moves
    def _grow_chain(self, chain_index, domain_indices):
        """Randomly grow out chain from given domain indices.

        Updates changes in energy as binding events occur."""

        # Iterate through given indices, growing next domain from current index
        for domain_index in domain_indices[:-1]:

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
                self._delta_e += self._trial_system.set_configuration(
                        chain_index, domain_index, r_new, o_new)
            except ConstraintViolation:
                raise MoveRejection

        return

    def _grow_staple(self, staple_length, staple_index, domain_index):
        """Randomly grow staple out from given domain in both directions."""

        # Grow in five-prime direction
        staple_indices = range(domain_index, staple_length)
        try:
            self._grow_chain(staple_index, staple_indices)
        except MoveRejection:
            raise

        # Grow in three-prime direction
        staple_indices = range(domain_index, -1, -1)
        try:
            self._grow_chain(staple_index, staple_indices)
        except MoveRejection:
            raise

    # Following are top level moves
    def _insert_staple(self):
        """Insert staple at random scaffold domain and grow."""

        # Randomly select staple identity
        staple_identity, domain_identities = (
                self._accepted_system.random_staple_identity())

        staple_index = self._trial_system.add_chain(staple_identity)
        staple_length = self._accepted_system.chain_length[staple_index]

        # Randomly select staple and scaffold domains to bind
        staple_domain = random.randrange(staple_length)

        scaffold_index = 0
        scaffold_length = self._accepted_system.chain_lengths[scaffold_index]
        scaffold_domain = random.randrange(scaffold_length)
        position = self._trial_system.get_domain_position(scaffold_index,
                scaffold_domain)

        # Attempt to set position
        try:
            self._delta_e += self._trial_system.set_domain_configuration(
                    staple_index, staple_domain, position)
        except ConstraintViolation:
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
        """Delete random staple."""

        # Randomly select staple
        staple_index = random.randrange(1, len(self._accepted_system.indices))

        self._delta_e += self._trial_system.delete_chain(staple_index)

        # Test acceptance
        if self._staple_deletion_accepted():
            self._accepted_system = self._trial_system
        else:
            pass

    def _regrow_staple(self):
        """Regrow random staple."""

        # Randomly select staple
        staple_index = random.randrange(1, len(self._accepted_system.indices))
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
            return

        # Test acceptance
        if self._configuration_accepted():
            self._accepted_system = self._trial_system
        else:
            self._trial_system = self._accepted_system

    def _regrow_scaffold_and_bound_staples(self):
        """Randomly regrow terminal section of scaffold and bound staples.

        From a randomly selected scaffold domain in a random direction to the
        end.
        """

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
                    staples[staple_index].append(domain_index, staple_domain_i)
                except KeyError:
                    staples[staple_index] = [(domain_index, staple_domain_i)]

        # Unassign scaffold domains
        for domain_index in scaffold_indices[1:]:
            self._delta_e += self._trial_system.unassign_domain(scaffold_index,
                    domain_index)

        # Unassign staples
        for staple_index in staples.keys():
            for domain_index in self._trial_system.chain_lengths[staple_index]:
                self._delta_e += self._trial_system.unassign_domain(staple_index,
                    domain_index)

        # Regrow scaffold
        try:
            self._grow_chain(scaffold_index, scaffold_indices)
        except MoveRejection:
            return

        # Regrow staples
        for staple_index, bound_domains in staples.values():

            # Pick domain on scaffold and staple to grow from
            scaffold_domain_index, staple_domain_index = random.choice(
                    bound_domains)
            position = self._trial_system.get_domain_position(scaffold_index,
                    scaffold_domain_index)

            # Randomly select first orientation
            dimension = random.choice([XHAT, YHAT, ZHAT])
            direction = random.randrange(-1, 2, 2)
            o_new = dimension * direction

            # Attempt to set growth domain
            try:
                self._delta_e += self._trial_system.set_configuration(
                        staple_index, staple_domain_index, position, o_new)
            except ConstraintViolation:
                return

            # Grow remainder of staple
            staple_length = self._trial_system.chain_lengths[staple_index]
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
        """Randomly rotate random domain."""

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
