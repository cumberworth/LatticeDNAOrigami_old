#!/usr/env python

"""Generates tikz scripts for configurations of the origami model.

Takes template tikz scripts for geneting configuration diagrams and outputs a
script for configurations at specified steps in the specified intput file.
"""

from lattice_origami_domains import JSONInputFile, HDF5InputFile
import sys
import string


INPUT_FILENAME = sys.argv[1]
OUTPUT_FILENAME = sys.argv[2]
STEP = int(sys.argv[3])

TEMPLATE_FILENAME = 'tikz_template.tex'


def open_inputfile(filename):
    filename_extension = filename.split('.')[1]
    if filename_extension == 'json':
        input_file = JSONInputFile(filename)
    elif filename_extension == 'hdf5':
        input_file = HDF5InputFile(filename)
    else:
        print('Filetype {} not supported.'.format(filename_extension))
        sys.exit()

    return input_file


def make_tikz_position_bond_orientation_list(chains):
    tikz_list = ''
    for chain in chains:
        for domain_index in range(len(chain['positions'])):
            rix = chain['positions'][domain_index][0]
            riy = chain['positions'][domain_index][1]
            riz = chain['positions'][domain_index][2]
            try:
                rjx = chain['positions'][domain_index + 1][0]
                rjy = chain['positions'][domain_index + 1][1]
                rjz = chain['positions'][domain_index + 1][2]
                aix = rjx - rix
                aiy = rjy - riy
                aiz = rjz - riz
            except IndexError:
                aix = 0
                aiy = 0
                aiz = 0

            bix = chain['orientations'][domain_index][0]
            biy = chain['orientations'][domain_index][1]
            biz = chain['orientations'][domain_index][2]

            tikz_list = tikz_list + '{} / {} / {} / {} / {} / {} / {} / {} / {}, '.format(
                    rix, riy, riz, aix, aiy, aiz, bix, biy, biz)

    # Remove last comma and space
    tikz_list = tikz_list[:-2]

    return tikz_list


def insert_list_and_write(tikz_list, output_filename):
    with open(TEMPLATE_FILENAME) as input:
        template = string.Template(input.read())

    template = template.substitute(tikz_list=tikz_list)
    with open(output_filename, 'w') as output:
        output.write(template)


def main():
    input_file = open_inputfile(INPUT_FILENAME)
    chains = input_file.chains(STEP)
    tikz_list = make_tikz_position_bond_orientation_list(chains)
    insert_list_and_write(tikz_list, OUTPUT_FILENAME)

if __name__ == '__main__':
    main()
