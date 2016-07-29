# distutils: language = c++
# distutils: sources = origami_system.cpp

cdef extern from 'origami_system.h' from '?':
    cdef cppclass origami_system:
        OrigamiSystem(?)

        ? get_chains()
        ? get_num_staples()
        ? get_num_bound_domains()
        ? get_energy()
        ? get_complimentary_domains()
        ? get_domain_position()
        ? get_domain_orientation()
        ? get_position_occupancy()
        ? get_domain_occupancy()
        ? get_bound_domain()
        ? get_unbound_domain()
        # maybe get rid of these?
        ? get_random_staple_identity()
        ? get_random_staple_of_identity
        #
        ? get_hybridization_energy() # should this be private?
        ? check_all_constraints
        ? check_domain_configuration
        ? set_checked_domain_configuration
        ? set_domain_configuration
        ? set_domain_orientation
        ? unassign_domain
        ? add_chain
        ? readd_chain
        ? delete_chain
        ? center

cdef class OrigamiSystem:

    def __cinit__(self, ?):
        self._thisptr = new origami_system(?)
        if self._thisptr == NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._thisptr != NULL:
            del self._thisptr
