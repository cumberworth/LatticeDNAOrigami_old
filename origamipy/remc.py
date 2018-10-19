"""Analysis of REMC simulations"""

import collections

from origamipy import io


def deconvolute_remc_outputs(all_exchange_params, fileinfo, filetypes):
    swapfile = io.SwapInpFile(fileinfo.inputdir, fileinfo.filebase)
    all_file_collections = create_file_collections(all_exchange_params,
                                                   fileinfo, filetypes)
    for threads_to_replicas in swapfile:
        for file_collection in all_file_collections:
            file_collection.deconvolute_and_write_step(threads_to_replicas)


def create_file_collections(all_exchange_params, fileinfo, filetypes):
    all_file_collections = []
    for filetype in filetypes:
        f_collection = FileCollection(all_exchange_params, fileinfo, filetype)
        all_file_collections.append(f_collection)


def create_exchange_params(temps, stack_mults):

    # This is the order the exchange parameters are output in the exchange file
    # TODO: Have this be read from the exchange file
    all_params = []
    for temp in temps:
        for stackm in stack_mults:
            all_params.append(Params(temp, stackm))

    return all_params


def exchange_params_to_subfile_string(params):
    return '-'.join(params)


Params = collections.namedtuple('Params', ['temp', 'stack_mult'])


class FileCollection:
    """Read from thread files and write to replica files"""

    def __init__(self, fileinfo, ext, all_exchange_params):
        self._fileinfo = fileinfo
        self._ext = ext
        self._all_exchange_params = all_exchange_params
        self._num_threads = len(all_exchange_params)
        self._thread_files = []
        self._replica_files = []
        self._filetype = io.UnparsedStepInpFile
        self._num_header_lines = 0

        self._set_filetype()
        self._open_thread_files()
        self._open_replica_files()
        self._write_headers()

    def deconvolute_and_write_step(self, threads_to_replicas):
        for thread, replica in enumerate(threads_to_replicas):
            step = next(self._thread_files[thread])
            self._replica_files[replica].write(step)

    def _set_filetype(self):
        if self._ext in ['trj', 'vcf']:
            self._num_header_lines = 0
            self._filetype = io.UnparsedMultiLineStepInpFile
        elif self._ext in ['times', 'ores', 'states', 'staples',
                           'staplestates']:
            self._num_header_lines = 0
            self._filetype = io.UnparsedSingleLineStepInpFile
        elif self._ext in ['enes', 'ops']:
            self._num_header_lines = 1
            self._filetype = io.UnparsedSingleLineStepInpFile
        else:
            raise NotImplementedError

    def _open_thread_files(self):
        for thread in range(self._num_threads):
            filename = '{}/{}-{}.{}'.format(self._fileinfo.inputdir,
                                            self._fileinfo.filebase, thread,
                                            self._ext)
            thread_file = self.__filetype(filename, self._num_header_lines)
            self._thread_files.append(thread_file)

    def _open_replica_files(self):
        for params in self._all_exchange_params:
            params = exchange_params_to_subfile_string(params)
            filename = '{}/{}-{}.{}'.format(self._fileinfo.outputdir,
                                            self._fileinfo.filebase, params,
                                            self._ext)
            self._replica_files.append(open(filename, 'w'))

    def _write_headers(self):
        for thread_f, replica_f in zip(self._thread_files,
                                       self._replica_files):
            replica_f.write(thread_f.header)
