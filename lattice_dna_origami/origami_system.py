#!/usr/env python

"""Run MC simulations of the lattice domain-resolution DNA origami model."""

import random

import numpy as np
import scipy.constants

from lattice_dna_origami.nearest_neighbour import *
from lattice_dna_origami.origami_io import *
from lattice_dna_origami.utility import *


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

            p_i = self.get_domain_position(chain_i, d_i)

            d_j = d_i + 1
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
