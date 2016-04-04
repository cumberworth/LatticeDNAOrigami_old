#!/usr/env python

"""Unit tests for the lattice_origami_domains module.

Run tests with py.test.
"""

import json
import sys
import pdb
import pytest
import scipy.constants
from lattice_origami_domains import *


class random():
    """Mock functions of the random module."""

    def randrange(self):
        pass

    def random(self):
        pass

    def choice(self):
        pass


@pytest.mark.parametrize('value, multiple, expected', [
    (10, 5, True),
    (11, 5, False),
    (0, 5, True),
    (10, 0, False)])
def test_value_is_multiple(value, multiple, expected):
    test_is_multiple = value_is_multiple(value, multiple)
    assert test_is_multiple == expected


@pytest.mark.parametrize('vector, rotation_axis, direction, expected', [
    ([1, 0, 0], XHAT, 1, [1, 0, 0]),
    (np.array([1, 0, 0]), XHAT, 1, np.array([1, 0, 0])),
    (np.array([3, 7, 2]), XHAT, 1, np.array([3, -2, 7])),
    (np.array([3, 7, 2]), XHAT, -1, np.array([3, 2, -7])),
    (np.array([3, 7, 2]), YHAT, 1, np.array([2, 7, -3])),
    (np.array([3, 7, 2]), ZHAT, 1, np.array([-7, 3, 2]))])
def test_rotate_vector_quarter(vector, rotation_axis, direction, expected):
    test_vector = rotate_vector_quarter(vector, rotation_axis, direction)
    try:
        assert all(test_vector == expected)
    except TypeError:
        assert test_vector == expected


@pytest.mark.parametrize('sequence, T, expected', [

    # Two terminal AT
    ('TTATAACT', 300, -1424.1117624983883),
# (0.2 - 300 * -0.0057 + -7.6 - 300 * -0.0213 + -7.2 - 300 * -0.0204 + -7.6 - 300 * -0.0213 + -7.8 - 300 * -0.021 + 2 * (2.2 - 300 * 0.0069)) * 4.184 * 1000 / scipy.constants.gas_constant

    # Complimentary to previous (and reversed to be 5' to 3')
    ('AGTTATAA', 300, -1424.1117624983883),

    # Palindrome
    ('TGCATGCA', 320, -1252.0106237088303),
# (0.2 - 320 * -0.0057 + 4 * (-8.5 - 320 * -0.0227) + 2 * (2.2 - 320 * 0.0069) - 320 * -0.0014) * 4.184 * 1000 / scipy.constants.gas_constant

    # One terminal AT
    ('CTATAACT', 300, -1635.4640382048626),
# (0.2 - 300 * -0.0057 + -7.8 - 300 * -0.021 + -7.2 - 300 * -0.0204 + -7.6 - 300 * -0.0213 + -7.8 - 300 * -0.021 + 2.2 - 300 * 0.0069) * 4.184 * 1000 / scipy.constants.gas_constant

    # No terminal AT
    ('CTATAACG', 300, -2173.909121552311),
# (0.2 - 300 * -0.0057 + -7.8 - 300 * -0.021 + -7.2 - 300 * -0.0204 + -7.6 - 300 * -0.0213 + -10.6 - 300 * -0.0272) * 4.184 * 1000 / scipy.constants.gas_constant

    # Sequence with every pair in the table
    ('AAATTACAGTCTGACGGCGG', 300, -7256.4281325889615)])
# (0.2 - 300 * -0.0057 -7.6 - 300 * -0.0213 + -7.2 - 300 * -0.0204 + -7.2 - 300 * -0.0213 + -8.5 - 300 * -0.0227 + -8.4 - 300 * -0.0224 + -7.8 - 300 * -0.021 + -8.2 - 300 * -0.0222 + -10.6 - 300 * -0.0272 + -9.8 - 300 * -0.0244 + -8 - 300 * -0.0199 + 2.2 - 300 * 0.0069) * 4.184 * 1000 / scipy.constants.gas_constant
def test_calc_hybridization_energy(sequence, T, expected):
    energy = calc_hybridization_energy(sequence, T)
    assert math.isclose(energy, expected)


# calc_complimentary_sequence was implicitly tested with test_calc_hybridization


@pytest.mark.parametrize('sequence, expected', [

    # Regular palindromes should fail
    ('TGAAGT', False),

    # Palindromic sequences are those which, when complimented and reversed, are
    # equal (i.e., they fold back on themselves and hybridize (hairpin))
    ('TGATCA', True)])
def test_sequence_is_palidromic(sequence, expected):
    assert sequence_is_palindromic(sequence) == expected


@pytest.fixture
def example_origami_json():
    return JSONInputFile('example_origami.json')


class TestJSONInputFile:

    def test_identities(self, example_origami_json):
        expected = [[-1, -2, -3, -4, -5, -6], [1, 2], [3, 4]]
        assert example_origami_json.identities == expected

    def test_sequences(self, example_origami_json):
        expected = ['TCCCTAGA', 'GGGTGGGA', 'CTCAAAGG', 'TTGTTGAA', 'GGAATAAG', 'GCTAGCGG']
        assert expected == example_origami_json.sequences

    def test_chains(self, example_origami_json):
        expected = [
            {
                'index': 0,
                'identity': 0,
                'positions': [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0]],
                'orientations': [[0, 1, 0], [0, 0, 1], [0, 1, 0], [0, 0, -1], [0, 0, -1], [0, 0, 1]]
            }, {
                'index': 1,
                'identity': 1,
                'positions': [[1, 0, -1], [1, 0, 0]],
                'orientations': [[0, -1, 0], [0, 0, -1]]
            }, {
                'index': 2,
                'identity': 2,
                'positions': [[2, 0, 0], [3, 0, 0]],
                'orientations': [[0, -1, 0], [0, 0, 1]]
            }
        ]
        chains = example_origami_json.chains(0)
        assert chains == expected


@pytest.fixture
def example_origami_system(example_origami_json):
    return OrigamiSystem(example_origami_json, 0, 300, 1)


class TestOrigamiSystem:

    def test_chains(self, example_origami_system, example_origami_json):
        assert example_origami_system.chains == example_origami_json.chains(0)

    @pytest.mark.parametrize('chain_index, domain_index, expected', [
        (0, 0, np.array([0, 0, 0])),
        (2, 1, np.array([3, 0, 0]))])
    def test_get_domain_position(self, chain_index, domain_index, expected,
            example_origami_system):
        position = example_origami_system.get_domain_position(chain_index,
                domain_index)
        assert all(position == expected)

    @pytest.mark.parametrize('chain_index, domain_index, expected', [
        (0, 0, np.array([0, 1, 0])),
        (2, 1, np.array([0, 0, -1]))])
    def test_get_domain_orientation(self, chain_index, domain_index, expected,
            example_origami_system):
        orientation = example_origami_system.get_domain_orientation(chain_index,
                domain_index)
        assert all(orientation == expected)

    @pytest.mark.parametrize('position, expected', [
        (np.array([0, 0, 0]), UNBOUND),
        (np.array([0, 1, 0]), BOUND),
        (np.array([4, 0, 0]), UNBOUND)])
    def test_get_position_occupancy(self, position, expected,
            example_origami_system):
        test_occupancy = example_origami_system.get_position_occupancy(position)
        assert test_occupancy == expected

    @pytest.mark.parametrize('chain_index, domain_index, expected', [
        (0, 0, UNBOUND),
        (1, 0, UNBOUND),
        (1, 1, BOUND),
        (0, 1, BOUND)])
    def test_get_domain_occupancy(self, chain_index, domain_index, expected,
            example_origami_system):
        test_occupancy = example_origami_system.get_domain_occupancy(chain_index,
                domain_index)
        assert test_occupancy == expected

    @pytest.mark.parametrize('domain, expected', [
        ((0, 1), (1, 1)),
        ((0, 4), ())])
    def test_get_bound_domain(self, domain, expected, example_origami_system):
        test_domain = example_origami_system.get_bound_domain(*domain)
        assert test_domain == expected

    @pytest.mark.parametrize('chain_index, domain_index, expected', [

        # Change unbound scaffold strand domain
        (0, 0, np.array([1, 0, 0])),

        # Change staple strand domain
        (1, 0, np.array([0, 1, 0]))])
    def test_set_domain_orientation_correct(self, chain_index, domain_index, expected,
            example_origami_system):
        example_origami_system.set_domain_orientation(chain_index, domain_index,
                expected)
        test_orientation = example_origami_system.get_domain_orientation(chain_index,
                domain_index)
        assert all(test_orientation == expected)

    @pytest.mark.parametrize('chain_index, domain_index, expected', [

        # Change bound scaffold strand domain
        (0, 1, np.array([1, 0, 0])),

        # Multiply by scalar
        (0, 1, np.array([0, 0, 99]))])
    def test_set_domain_orientation_wrong(self, chain_index, domain_index, expected,
            example_origami_system):
        with pytest.raises(ConstraintViolation):
            example_origami_system.set_domain_orientation(chain_index,
                    domain_index, expected)

    @pytest.mark.parametrize('position, domain, expected', [
        (np.array([4, 0, 0]), (0, 0), BOUND),
        (np.array([5, 5, 5]), (0, 5), UNBOUND)])
    def test_add_occupancy(self, position, domain, expected,
                example_origami_system):
        example_origami_system.add_occupancy(position, domain)
        test_occupancy = example_origami_system.get_position_occupancy(position)
        assert test_occupancy == expected

    @pytest.mark.parametrize('position, domain, expected', [
        (np.array([0, 0, 0]), (0, 0), UNBOUND),
        (np.array([4, 0, 0]), (0, 4), UNASSIGNED)])
    def test_remove_occupancy(self, position, domain, expected,
            example_origami_system):
        example_origami_system.remove_occupancy(position, domain)
        test_occupancy = example_origami_system.get_position_occupancy(position)
        assert test_occupancy == expected
        domain_occupancy = example_origami_system.get_domain_occupancy(*domain)
        #assert domain_occupancy == UNASSIGNED

    def test_add_chain(self, example_origami_system):
        identity = 2
        chain_length = len(example_origami_system.identities[2])
        positions = orientations = [np.zeros(3)] * chain_length
        example_origami_system.add_chain(identity, positions, orientations)
        expected_unique_indices = [0, 1, 2, 3]
        assert example_origami_system._indices  == expected_unique_indices
        test_position = example_origami_system.get_domain_position(3, 0)
        assert all(test_position == np.array([0, 0, 0]))
        #test_occupancy = example_origami_system.get_domain_occupancy(3, 0)
        #assert test_occupancy == UNASSIGNED

    #def test_delete_chain

    #def test_random_staple_identity

    def domains_match(self, example_origami_system):

        # Complimentary domains in correct orientation
        domain_1 = (0, 0)
        domain_2 = (1, 0)
        test_match = example_origami_system.domains_match(*domain_1, *domain_2)
        assert test_match == True

        # Complimentary domains in incorrect orientation
        domain_2_orientation = example_origami_system.get_domain_orientation(
                *domain_2)
        domain_2_orientation = rotate_vector_quarter(domain_2_orientation, XHAT, 1)
        example_origami_sytem.set_domain_orientation(*domain_2, domain_2_orientation)
        test_match = example_origami_system.domains_match(*domain_1, *domain_2)
        assert test_match == False

        # Non-complimentary domains in correct orientation

        # First double check revert orientation goes back to a match
        domain_2_orientation = rotate_vector_quarter(domain_2_orientation, XHAT, -1)
        test_match = example_origami_system.domains_match(*domain_1, *domain_2)
        assert test_match == True

        # Delete chain and replace with new chain with differetn identity
        example_origami_system.delete_chain(1)
        example_origami_system.add_chain(3, [np.array([0, 0, 0]),
                np.array([1, 0, 0])], [domain_2_orientation,
                        np.array([0, 1, 0])])
        domain_2 = (2, 0)
        test_match = example_origami_system.domains_match(*domain_1, *domain_2)
        assert test_match == False

        # Double check that directly reverting chain identity will return match
        example_origami_system._chain_identities[2] = 1
        test_match = example_origami_system.domains_match(*domain_1, *domain_2)
        assert test_match == True

    @pytest.mark.parametrize('domain, sequence', [

        ((0, 0), 'TCCCTAGA'),
        ((1, 1), 'GGGTGGGA')])
    def test_get_hybridization_energy(self, domain, sequence,
                example_origami_system):
        expected = calc_hybridization_energy(sequence, 300)
        test_energy = example_origami_system.get_hybridization_energy(*domain)
        assert test_energy == expected

    def test_domains_part_of_same_helix(self, example_origami_system):

        # Contiguous helical domains
        chain_index = 0
        domain_index_1 = 0
        domain_index_2 = 1
        same_helix = example_origami_system.domains_part_of_same_helix(
                chain_index, domain_index_1, domain_index_2)
        assert same_helix == True

        # Iterate through all other perpindicular orientations
        for i in range(3):
            domain_orientation = example_origami_system.get_domain_orientation(
                    chain_index, domain_index_1)
            domain_orientation = rotate_vector_quarter(domain_orientation,
                    XHAT, 1)
            example_origami_system.set_domain_orientation(chain_index,
                    domain_index_1, domain_orientation)
            same_helix = example_origami_system.domains_part_of_same_helix(
                    chain_index, domain_index_1, domain_index_2)
            assert same_helix == True


        # Antiparallel orientation
        example_origami_system.set_domain_orientation(chain_index,
                domain_index_1, np.array([-1, 0, 0]))
        same_helix = example_origami_system.domains_part_of_same_helix(
                chain_index, domain_index_1, domain_index_2)
        assert same_helix == False

        # Parallel orientation
        example_origami_system.set_domain_orientation(chain_index,
                domain_index_1, np.array([1, 0, 0]))
        same_helix = example_origami_system.domains_part_of_same_helix(
                chain_index, domain_index_1, domain_index_2)
        assert same_helix == False

        # Bound/unbound domain pair with correct orientation
        domain_index_1 = 3
        domain_index_2 = 4
        same_helix = example_origami_system.domains_part_of_same_helix(
                chain_index, domain_index_1, domain_index_2)
        assert same_helix == False

        # Bound staple domains


    def test_domains_have_correct_twist(self, example_origami_system):

        # Contiguous helices with correct twist
        chain_index = 0
        domain_index_1 = 0
        domain_index_2 = 1
        twist_obeyed = example_origami_system.domains_have_correct_twist(
                chain_index, domain_index_1, domain_index_2)
        assert twist_obeyed == True

        # Contiguous helices with wrong twist
        for i in range(3):
            domain_orientation = example_origami_system.get_domain_orientation(
                    chain_index, domain_index_1)
            domain_orientation = rotate_vector_quarter(domain_orientation,
                    XHAT, 1)
            example_origami_system.set_domain_orientation(chain_index,
                    domain_index_1, domain_orientation)
            twist_obeyed = example_origami_system.domains_have_correct_twist(
                    chain_index, domain_index_1, domain_index_2)
            assert twist_obeyed == False
