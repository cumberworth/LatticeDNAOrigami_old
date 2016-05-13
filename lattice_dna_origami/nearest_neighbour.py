#!/usr/env python

"""Utilities for calculating NN model quantities."""

import math
import scipy.constants

# Molar gas constant (J/K/mol)
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


def calc_hybridization_energy(sequence, T, cation_M):
    """Outputs energies in K to avoid multiplying by KB when calculating acceptances.

    Sequences are assumed to be 5' to 3'.

    cation_M -- Total cation molarity.
    """
    DH, DS = calc_hybridization_enthalpy_and_entropy(sequence, cation_M)
    DG = DH - T * DS

    # Convert from kcal/mol to K (so avoid KB later)
    DG = DG * J_PER_CAL * 1000 / R
    
    return DG


def calc_melting_point(sequence, strand_M, cation_M):
    """Calculate melting point assuming two state behaviour."""
    DH, DS = calc_hybridization_enthalpy_and_entropy(sequence, cation_M)

    # Convert to J/mol
    DH = DH * J_PER_CAL * 1000

    # Convert to J/mol/K
    DS = DS * J_PER_CAL * 1000

    # Factor dependent on whether sequence is palindrome (non-selfcomplimentary)
    if sequence_is_palindromic(sequence):
        x = 1
    else:
        x = 4

    melting_T = DH / (DS + R * math.log(strand_M / x))

    return melting_T


def calc_hybridization_enthalpy_and_entropy(sequence, cation_M):
    """Calculate hybridization enthalpy and entropy of domains with NN model.

    Outputs in kcal/mol.

    """
    complimentary_sequence = calc_complimentary_sequence(sequence)

    # Initiation free energy
    DH_init = NN_ENTHALPY['INITIATION']
    DS_init = NN_ENTROPY['INITIATION']

    # Symmetry penalty for palindromic sequences
    if sequence_is_palindromic(sequence):
        DS_sym = NN_ENTROPY['SYMMETRY_CORRECTION']
    else:
        DS_sym = 0.

    # NN pair energies
    DH_stack = 0
    DS_stack = 0
    for base_index in range(0, len(sequence) - 1):
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

    return DH_hybrid, DS_hybrid


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
