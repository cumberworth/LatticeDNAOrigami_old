#!/usr/bin/env python

"""Calculate average NN enthalpies and entropies of bound and misbound domains"""

import argparse

import numpy as np

import origamipy.nearest_neighbour as nn
import origamipy.files as files


def main():
    args = parse_cl()
    origami = files.JSONStructInpFile(args.system_file)
    binding_h = []
    binding_s = []
    misbinding_h = []
    misbinding_s = []
    for chain_i in origami.sequences:
        for seq_i in chain_i:
            seq_i_revcomp = nn.calc_complementary_sequence(seq_i)[::-1]
            for chain_j in origami.sequences:
                for seq_j in chain_j:
                    if seq_i_revcomp == seq_j:
                        h, s = nn.calc_hybridization_enthalpy_and_entropy(
                                seq_i, args.cation_M)
                        binding_h.append(nn.remove_energy_units(h))
                        binding_s.append(nn.remove_energy_units(s))
                    else:
                        seqs = nn.find_longest_contig_complement(seq_i, seq_j)
                        seqs_h = []
                        seqs_s = []
                        for seq in seqs:
                            h, s = nn.calc_hybridization_enthalpy_and_entropy(
                                    seq, args.cation_M)
                            seqs_h.append(h)
                            seqs_s.append(s)

                        if len(seqs) == 0:
                            misbinding_h.append(0)
                            misbinding_s.append(0)
                        else:
                            misbinding_h.append(nn.remove_energy_units(
                                    np.mean(seqs_h)))
                            misbinding_s.append(nn.remove_energy_units(
                                    np.mean(seqs_s)))

    sbinding_h = []
    sbinding_s = []
    for staple in origami.sequences[1:]:
        seq = ''
        for s in staple:
            seq += s

        h, s = nn.calc_hybridization_enthalpy_and_entropy(seq, args.cation_M)
        sbinding_h.append(nn.remove_energy_units(h))
        sbinding_s.append(nn.remove_energy_units(s))


    print('Average bound domain hybridization enthalpy: ', np.mean(binding_h))
    print('Average bound domain hybridization entropy: ', np.mean(binding_s))
    print('Average misbound domain hybridization enthalpy: ', np.mean(misbinding_h))
    print('Average misbound domain hybridization entropy: ', np.mean(misbinding_s))
    print('Average staple hybridization enthalpy: ', np.mean(sbinding_h))
    print('Average staple domain hybridization entropy: ', np.mean(sbinding_s))


def parse_cl():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'system_file',
        type=str,
        help='System file')
    parser.add_argument(
        'cation_M',
        type=float,
        help='Cation concentration')

    return parser.parse_args()


if __name__ == '__main__':
    main()
