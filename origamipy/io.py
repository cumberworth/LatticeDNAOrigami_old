#!/usr/env python

"""IO classes for origami system information, topologies, and configurations"""

import numpy as np
import json


class JSONStructInpFile:
    """JSON input file for system information, topology, and configuration

    Can contain multiple configurations.
    """

    def __init__(self, filename):
        json_origami = json.load(open(filename))

        self._filename = filename
        self._json_origami = json_origami

    @property
    def cyclic(self):
        return self._json_origami['origami']['cyclic']

    @property
    def identities(self):
        """Standard format for passing origami domain identities"""
        return self._json_origami['origami']['identities']

    @property
    def sequences(self):
        """Standard format for passing origami domain sequences"""
        return self._json_origami['origami']['sequences']

    def chains(self, step):
        """Standard format for passing chain configuration."""
        return self._json_origami['origami']['configurations'][step]['chains']

    def close(self):
        pass


class JSONStructOutFile:
    """JSON output file for system information, topology, and configuration"""

    def __init__(self, filename, origami_system):
        self._filename = filename

        self.json_origami = {'origami': {'identities': {},
                             'configurations': []}}
        self.json_origami['origami']['identities'] = origami_system.identities
        self.json_origami['origami']['sequences'] = origami_system.sequences
        self.json_origami['origami']['cyclic'] = origami_system.cyclic

    def write(self, origami_system):
        self.json_origami['origami']['configurations'].append({})
        current_config = self.json_origami['origami']['configurations'][-1]

        # Step should probably be changed as this is no longer being used by a
        # a simulation class
        current_config['step'] = 0
        current_config['chains'] = []
        for chain in origami_system.chains:
            current_config['chains'].append({})
            json_chain = current_config['chains'][-1]
            json_chain['identity'] = chain['identity']
            json_chain['index'] = chain['index']
            json_chain['positions'] = chain['positions']
            json_chain['orientations'] = chain['orientations']

        json.dump(self.json_origami, open(self._filename, 'w'), indent=4,
                  separators=(',', ': '))


class TxtTrajInpFile:
    """Plain text trajectory input file

    Requires system information be provided through an input file.
    """

    def __init__(self, traj_filename, struct_file):
        self._traj_filename = traj_filename
        self._struct_file = struct_file
        self._line = ''
        self._eof = False
        self._step = -1
        self._chains = []

        self._traj_file = open(traj_filename)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._eof:
            try:
                self._parse_chains()
            except StopIteration:
                self._eof = True
            finally:
                return self._chains
        else:
            self._eof = False
            # TODO: use to be implemented method for get_chains
            self._traj_file.seek(0)
            raise StopIteration

    @property
    def step(self):
        return len(self._step)

    def get_chains(self, step):
        pass
        # find starting lines
        # parse chains

    def close(self):
        self._traj_file.close()

    def _parse_chains(self):
        self._next_line()
        self._step = self._get_step()
        self._chains = []
        chains_remain = True
        self._next_line()
        while chains_remain:
            self._parse_chain()
            if self._line == '':
                chains_remain = False

    def _next_line(self):
        self._line = next(self._traj_file).rstrip()

    def _get_step(self):
        return int(self._line)

    def _parse_chain(self):
        chain = {}
        chain['index'], chain['identity'] = self._get_index_and_identity()
        self._next_line()
        chain['positions'] = self._get_domainsx3_matrix_from_line()
        self._next_line()
        chain['orientations'] = self._get_domainsx3_matrix_from_line()
        self._chains.append(chain)
        self._next_line()

    def _get_index_and_identity(self):
        split_line = self._line.split()
        return split_line[0], split_line[1]

    def _get_domainsx3_matrix_from_line(self):
        row_major_vector = np.array(self._line.split(), dtype=int)
        return row_major_vector.reshape(len(row_major_vector) // 3, 3).tolist()


class TxtTrajOutFile:
    """Plain text trajectory output file"""

    def __init__(self, filename):
        self.file = open(filename, 'w')

    def write_config(self, chains, step):
        self.file.write('{}\n'.format(step))
        for chain in chains:
            self.file.write('{} '.format(chain['index']))
            self.file.write('{}\n'.format(chain['identity']))
            for pos in chain['positions']:
                for comp in pos:
                    self.file.write('{} '.format(comp))
            self.file.write('\n')
            for ore in chain['orientations']:
                for comp in ore:
                    self.file.write('{} '.format(comp))
            self.file.write('\n')
        self.file.write('\n')
