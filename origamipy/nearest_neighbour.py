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

COMPLEMENTARY_BASE_PAIRS = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}


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


def find_longest_contig_complement(seq_1, seq_2):
    """Find longest contiguous complementary sequences between two strands.
    
    The sequences are assumed to be 3'-5'.
    """

    # Find smallest sequence
    if len(seq_2) < len(seq_1):
        seq_3 = seq_2[::-1]
        seq_5 = seq_1
    else:
        seq_3 = seq_1[::-1]
        seq_5 = seq_2

    seq_3 = calc_complementary_sequence(seq_3)

    # Iterate through all lengths and starting points
    comp_seqs = []
    for subseq_len in range(len(seq_3), 0, -1):
        for start_i in range(0, len(seq_3) - subseq_len + 1):
            subseq = seq_3[start_i:subseq_len + start_i]
            seq_i = seq_5.find(subseq)
            if seq_i == -1:
                continue
            else:
                comp_seq = seq_5[seq_i:seq_i + subseq_len]
                comp_seqs.append(comp_seq)
        if comp_seqs == []:
            continue
        else:
            return comp_seqs

    # WRONG JUST FOR BENCHMARKING
    print('WARNING WRONG WRONG WRONG')
    if comp_seqs == []:
        comp_seqs = ['A']

    return comp_seqs

def calc_melting_point(sequence, strand_M, cation_M):
    """Calculate melting point assuming two state behaviour."""
    DH, DS = calc_hybridization_enthalpy_and_entropy(sequence, cation_M)

    # Convert to J/mol
    DH = DH * J_PER_CAL * 1000

    # Convert to J/mol/K
    DS = DS * J_PER_CAL * 1000

    # Factor dependent on whether sequence is palindrome (non-selfcomplementary)
    if sequence_is_palindromic(sequence):
        x = 1
    else:
        x = 4

    melting_T = DH / (DS + R * math.log(strand_M / x))

    return melting_T

def calc_internal_melting_point(sequence, cation_M):
    """Calculate temperature at which enthalpy and entropy are equal."""
    DH, DS = calc_hybridization_enthalpy_and_entropy(sequence, cation_M)

    # Convert to J/mol
    DH = DH * J_PER_CAL * 1000

    # Convert to J/mol/K
    DS = DS * J_PER_CAL * 1000

    melting_T = DH / DS

    return melting_T


def calc_hybridization_enthalpy_and_entropy(sequence, cation_M):
    """Calculate hybridization enthalpy and entropy of domains with NN model.

    Outputs in kcal/mol.

    """
    complementary_sequence = calc_complementary_sequence(sequence)

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
        second_pair = complementary_sequence[base_index : base_index + 2]
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


def calc_complementary_sequence(sequence):
    """Return the complementary DNA sequence."""
    complementary_seq_list = []
    for base in sequence:
        complementary_seq_list.append(COMPLEMENTARY_BASE_PAIRS[base])

    complementary_sequence = ''.join(complementary_seq_list)
    return complementary_sequence


def sequence_is_palindromic(sequence):
    """True if reverse complemenet is equal to given sequence."""
    complementary_sequence = calc_complementary_sequence(sequence)
    reverse_complementary_sequence = complementary_sequence[::-1]
    palindromic = reverse_complementary_sequence == sequence
    return palindromic
