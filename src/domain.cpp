#include <iostream>

#include "domain.hpp"

namespace domain {

bool Domain::next_domain_exists() const { return m_next_domain != nullptr; }

bool Domain::prev_domain_exists() const { return m_prev_domain != nullptr; }

bool Domain::bound_domain_exists() const { return m_bound_domain != nullptr; }

bool Domain::contig_domain_exists(int i) const {
    if (i == 1) {
        return next_domain_exists();
    }
    if (i == -1) {
        return prev_domain_exists();
    }
    if (i == 0) {
        return true;
    }

    throw utility::NoElement {};
}

Domain& Domain::get_next_domain() const { return *m_next_domain; }

Domain& Domain::get_prev_domain() const { return *m_prev_domain; }

Domain& Domain::get_bound_domain() const { return *m_bound_domain; }

const Domain& Domain::get_contig_domain(int i) const {
    if (i == 1) {
        return get_next_domain();
    }
    if (i == -1) {
        return get_prev_domain();
    }
    if (i == 0) {
        return *this;
    }

    throw utility::NoElement {};
}

Domain& Domain::get_contig_domain(int i) {
    if (i == 1) {
        return get_next_domain();
    }
    if (i == -1) {
        return get_prev_domain();
    }
    if (i == 0) {
        return *this;
    }

    throw utility::NoElement {};
}

bool Domain::check_twist_constraint(const VectorThree ndr, const Domain& cd_2) const {
    bool twist_constraint_obeyed {true};
    VectorThree ore_1_rotated {m_ore.rotate_half(ndr)};
    if (not(ore_1_rotated == cd_2.m_ore)) {
        twist_constraint_obeyed = false;
    }
    return twist_constraint_obeyed;
}

bool Domain::check_kink_constraint(const VectorThree ndr, const Domain& cd_2) const {
    bool kink_constraint_obeyed {true};
    if (ndr == -m_ore) {
        kink_constraint_obeyed = false;
    }
    else if (ndr == m_ore) {
        if (cd_2.m_ore != m_ore) {
            kink_constraint_obeyed = false;
        }
    }
    else if (cd_2.m_ore == ndr or cd_2.m_ore == -ndr) {
        kink_constraint_obeyed = false;
    }

    return kink_constraint_obeyed;
}
} // namespace domain
