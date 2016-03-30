#!/usr/env python

import json
import pytest
import scipy.constants
from lattice_origami_domains import *

"""Tests for python implementation of lattice domain-res origami model."""


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
    ('TTATAACT', 300, -1424.1117624983883),
# (0.2 - 300 * -0.0057 + -7.6 - 300 * -0.0213 + -7.2 - 300 * -0.0204 + -7.6 - 300 * -0.0213 + -7.8 - 300 * -0.021 + 2 * (2.2 - 300 * 0.0069)) * 4.184 * 1000 / scipy.constants.gas_constant
    ('TGCATGCA', 320, -1252.0106237088303),
# (0.2 - 320 * -0.0057 + 4 * (-8.5 - 320 * -0.0227) + 2 * (2.2 - 320 * 0.0069) - 320 * -0.0014) * 4.184 * 1000 / scipy.constants.gas_constant
    ('CTATAACT', 300, -1635.4640382048626),
# (0.2 - 300 * -0.0057 + -7.8 - 300 * -0.021 + -7.2 - 300 * -0.0204 + -7.6 - 300 * -0.0213 + -7.8 - 300 * -0.021 + 2.2 - 300 * 0.0069) * 4.184 * 1000 / scipy.constants.gas_constant
    ('CTATAACG', 300, -2173.909121552311),
# (0.2 - 300 * -0.0057 + -7.8 - 300 * -0.021 + -7.2 - 300 * -0.0204 + -7.6 - 300 * -0.0213 + -10.6 - 300 * -0.0272) * 4.184 * 1000 / scipy.constants.gas_constant
    ('AAATTACAGTCTGACGGCGG', 300, -7256.4281325889615)])
# (0.2 - 300 * -0.0057 -7.6 - 300 * -0.0213 + -7.2 - 300 * -0.0204 + -7.2 - 300 * -0.0213 + -8.5 - 300 * -0.0227 + -8.4 - 300 * -0.0224 + -7.8 - 300 * -0.021 + -8.2 - 300 * -0.0222 + -10.6 - 300 * -0.0272 + -9.8 - 300 * -0.0244 + -8 - 300 * -0.0199 + 2.2 - 300 * 0.0069) * 4.184 * 1000 / scipy.constants.gas_constant
def test_calc_hybridization_energy(sequence, T, expected):
    energy = calc_hybridization_energy(sequence, T)
    assert math.isclose(energy, expected)


# calc_complimentary_sequence was implicitly tested with test_calc_hybridization


@pytest.mark.parametrize('sequence, expected', [

    # Regular palidromes should fail
    ('TGAAGT', False),

    # Palindromic sequences are those which, when complimented and reversed, are
    # equal (i.e., they fold back on themselves and hybridize (hairpin))
    ('TGATCA', True)])
def test_sequence_is_palidromic(sequence, expected):
    assert sequence_is_palindromic(sequence) == expected


#class TestOrigamiSystem(
