"""IO classes for origami system information, topologies, and configurations"""

import collections
import json

import numpy as np

NEWLINE = '\n'


FileInfo = collections.namedtuple('FileInfo', ['inputdir', 'outputdir',
                                               'filebase'])


class JSONStructInpFile:
    """JSON input file for system information, topology, and configuration.

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
    """JSON output file for system information, topology, and configuration."""

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


class StepsInpFile:
    def __init__(self, filename):
        self._filename = filename
        self._line = ''
        self._eof = False
        self._step = -1

        self._file = open(filename)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._eof:
            try:
                self._parse_step()
            except StopIteration:
                self._eof = True
            finally:
                return self._return_step()
        else:
            self._eof = False
            # TODO: use to be implemented method for get_chains
            self._file.seek(0)
            self._parse_header()
            raise StopIteration

    @property
    def step(self):
        return len(self._step)

    def close(self):
        self._file.close()

    def _parse_step(self):
        raise NotImplementedError

    def _return_step(self):
        raise NotImplementedError

    def _parse_header(self):
        raise NotImplementedError


class TxtTrajInpFile(StepsInpFile):
    """Plain text trajectory input file.

    Requires system information be provided through an input file.
    """

    def __init__(self, filename, struct_file):
        super().__init__(filename)
        self._struct_file = struct_file
        self._chains = []
        self._next_line()

    def get_chains(self, step):
        for i, chains in enumerate(self):
            if i == step:
                return chains

        else:
            raise IndexError

    def _parse_step(self):
        self._step = self._get_step()
        self._chains = []
        chains_remain = True
        self._next_line()
        while chains_remain:
            self._parse_chain()
            if self._line == '':
                chains_remain = False

        self._next_line()

    def _return_step(self):
        return self._chains

    def _next_line(self):
        self._line = next(self._file).rstrip()

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

    def _parse_header(self):
        pass


class TxtTrajOutFile:
    """Plain text trajectory output file."""

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


class SwapInpFile(StepsInpFile):
    def __init__(self, inputdir, filebase):
        filename = self._create_filename(inputdir, filebase)
        super().__init__(filename)
        self._header = ''
        self._threads_to_replicas = []
        # TODO: check if starting at step 0 or 1

        self._parse_header()

    def _create_filename(self, inputdir, filebase):
        return '{}/{}.{}'.format(inputdir, filebase, 'swp')

    def _parse_header(self):
        # TODO: extract the replica parameters
        self._next_line()
        self._header = self._line
        self._next_line()

    def _next_line(self):
        self._line = next(self._file).rstrip()

    def _parse_step(self):
        # TODO: take into account step size
        self._step += 1
        self._threads_to_replicas = [int(i) for i in self._line.split()]
        self._next_line()

    def _return_step(self):
        return self._threads_to_replicas


class UnparsedStepInpFile(StepsInpFile):
    """Read steps in single string chunks."""
    def __init__(self, filename, headerlines=0):
        super().__init__(filename)
        self._header = ''
        self._step_chunk = ''
        self._headerlines = headerlines

        self._parse_header()

    @property
    def header(self):
        return self._header

    def get_step(self, step):
        i = 0
        while i <= step:
            next(self)
            i += 1

        return self._step_chunk

    def get_last_step(self):
        for step in self:
            pass

        return step

    def _parse_header(self):
        self._next_line()
        for i in range(self._headerlines):
            self._header = self._header + self._line
            self._next_line()

    def _return_step(self):
        return self._step_chunk

    def _next_line(self):
        self._line = next(self._file)

    def _parse_step(self):
        raise NotImplementedError


class UnparsedMultiLineStepInpFile(UnparsedStepInpFile):
    """Read multi-line steps in single string chunks.

    The program's standard delimiter for steps is an empy new line (or a double
    new line.
    """
    def _parse_step(self):
        self._step_chunk = self._line
        while self._line != NEWLINE:
            self._next_line()
            self._step_chunk = self._step_chunk + self._line

        self._next_line()


class UnparsedSingleLineStepInpFile(UnparsedStepInpFile):
    """Read single line steps in single string chunks.

    The program's standard delimiter for steps is an empy new line (or a double
    new line.
    """
    def _parse_step(self):
        self._step_chunk = self._line
        self._next_line()


class TagOutFile:
    def __init__(self, filename):
        self._filename = filename

    def write(self, tags, data):
        np.savetxt(self._filename, data, header=' '.join(tags), comments='',
                   fmt='%.6f')
