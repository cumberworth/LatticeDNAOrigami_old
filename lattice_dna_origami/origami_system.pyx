# distutils: language = c++
# distutils: sources = origami_system.cpp

from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from 'origami_system.h' namespace 'Origami':

    cdef cppclass Chain:
        int index
        int identity
        vector[?] positions
        vector[?] orientations

    cdef cppclass OrigamiSystem:

        # Data
        vector[vector[int]] m_identities
        vector[vector[string]] m_sequences
        double m_temp
        double m_volume
        double m_cation_M
        double m_strand_M
        int c_scaffold

        # Constructor
        OrigamiSystem(
                vector[vector[int]] identities,
                vector[vector][string] sequences,
                Chains chains,
                double temp,
                double volume,
                double cation_M,
                double strand_M) except +

        # Configuration properties
        vector[int] chain_lengths()
        int num_staples()
        int num_bound_domains()
        double energy()

        # Staple properties
        int numb_staples_of_ident(int staple_ident)
        vector[int] complimentary_scaffold_domains(int staple_ident)

        # Configuration accessors
        Chains chains()
        ? domain_position(? cd_i)
        ? domain_orientation(? cd_i)
        ? position_occupancy(? pos)
        ? domain_occupancy(? cd_i)
        ? domain_bound_to(? cd_i)
        ? domain_bound_at(? pos)

        # Constraint checkers
        void check_all_constraints()
        double check_domain_constraints(? cd_i, ? pos, ? ore)
        double unassign_domain(? cd_i)
        int add_chain(int c_i_ident)
        int add_chain(int c_i_ident, int uc_i)
        void delete_chain(int c_i)
        void set_check_domain_config(? cd_i, ? pos, ? ore)
        double set_domain_config(? cd_i, ? pos, ? ore)
        void set_domain_orientation(? cd_i, ? ore)
        void centre()

cdef class OrigamiSystem:

    def __cinit__(self, ?):
        self._thisptr = new origami_system(?)
        if self._thisptr == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr
