#!/usr/bin/env python

"""Simulation and movetype classes."""

import copy
import math
import os
import sys
from enum import Enum

import numpy as np

from lattice_dna_origami.origami_io import *
from lattice_dna_origami.utility import *
from lattice_dna_origami.origami_system import *

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


IDEAL_RANDOM_WALKS = IdealRandomWalks()


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
