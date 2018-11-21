#ifndef DOMAIN_H
#define DOMAIN_H

#include "utility.hpp"

namespace domain {

using utility::Occupancy;
using utility::VectorThree;

// DNA origami binding domain
class Domain {
  public:
    Domain(int c, int c_ident, int d, int d_ident, int c_length):
            m_c {c},
            m_c_ident {c_ident},
            m_d {d},
            m_d_ident {d_ident},
            m_c_length {c_length} {};
    virtual ~Domain() = default;

    const int m_c; // Unique index of the associated chain
    const int m_c_ident; // Identity of the chain the associated chain
    const int m_d; // Domain index
    const int m_d_ident; // Domain identity
    const int m_c_length; // Associated chain length

    VectorThree m_pos {}; // Position vector
    VectorThree m_ore {}; // Orientation vector
    Occupancy m_state {Occupancy::unassigned}; // Binding state

    bool next_domain_exists() const;
    bool prev_domain_exists() const;
    bool bound_domain_exists() const;
    bool contig_domain_exists(const int i) const;
    Domain& get_next_domain() const;
    Domain& get_prev_domain() const;
    Domain& get_bound_domain() const;
    Domain& get_contig_domain(const int i);
    const Domain& get_contig_domain(const int i) const;
    bool check_twist_constraint(const VectorThree ndr, const Domain& cd_2)
            const;
    bool check_kink_constraint(const VectorThree ndr, const Domain& cd_2) const;
    Domain* m_next_domain {nullptr}; // Domain in ?' direction
    Domain* m_prev_domain {nullptr}; // Domain in ?' direction
    Domain* m_bound_domain {nullptr}; // Pointer to bound domain
};
} // namespace domain

#endif // DOMAIN_H
