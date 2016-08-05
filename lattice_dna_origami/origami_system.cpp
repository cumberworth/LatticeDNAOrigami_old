// origami_system.cpp

#include <algorithm>
#include <cmath>

#include "origami_system.h"
#include "nearest_neighbour.h"
#include "utility.h"

using std::abs;
using std::max;

using namespace NearestNeighbour;
using namespace Utility;
using namespace Origami;

// Public methods

OrigamiSystem::OrigamiSystem(
        const vector<vector<int>>& identities,
        const vector<vector<string>>& sequences,
        const Chains& chains,
        double temp,
        double volume,
        double cation_M,
        double strand_M) :

        m_identities {identities},
        m_sequences {sequences},
        m_temp {temp},
        m_volume {volume},
        m_cation_M {cation_M},
        m_strand_M {strand_M} {

    initialize_complementary_associations();
    initialize_energies();
    initialize_config(chains);
}

Chains OrigamiSystem::chains() const {
    // Return chains data structure for current config
    Chains chains;
    for (int c_i {0}; c_i != m_working_to_unique.size(); c_i++) {
        int uc_i = m_working_to_unique[c_i];
        int c_i_ident = m_chain_identities[c_i];
        Chain chain {uc_i, c_i_ident, m_positions[c_i], m_orientations[c_i]};
    }
    return chains;
}

double OrigamiSystem::unassign_domain(CDPair cd_i) {
    // Deletes positions, orientations, and removes/unassigns occupancies.
    CDPair ucd_i {m_working_to_unique[cd_i.c], cd_i.d};
    Occupancy occupancy {domain_occupancy(ucd_i)};
    double delta_e {0};
    switch (occupancy) {
        case Occupancy::bound:
            m_num_fully_bound_domains -= 1;
            delta_e += unassign_bound_domain(cd_i);
            break;
        case Occupancy::misbound:
            delta_e += unassign_bound_domain(cd_i);
            break;
        case Occupancy::unbound:
            unassign_unbound_domain(cd_i);
            break;
         case Occupancy::unassigned:
            break;
    }
    return delta_e;
}

void OrigamiSystem::set_domain_orientation(CDPair cd_i, VectorThree ore) {
    Occupancy occupancy {domain_occupancy(cd_i)};
    if (occupancy == Occupancy::bound or occupancy == Occupancy::misbound) {
        throw ConstraintViolation {};
    }
    else {
        m_orientations[cd_i.c][cd_i.d] = ore;
    }
}

double OrigamiSystem::set_domain_config(
        CDPair cd_i,
        VectorThree pos,
        VectorThree ore) {
    // Check constraints and update if obeyed, otherwise throw
    double delta_e {check_domain_constraints(cd_i, pos, ore)};
    update_occupancies(cd_i, pos);
    return delta_e;
}

void OrigamiSystem::set_checked_domain_config(
        CDPair cd_i,
        VectorThree pos,
        VectorThree ore) {
    update_domain(cd_i, pos, ore);
    update_occupancies(cd_i, pos);
}

double OrigamiSystem::check_domain_constraints(
        CDPair cd_i,
        VectorThree pos,
        VectorThree ore) {
    // Updates positions and orientations and returns without reverting if no 
    // constraint violation. But states remained unassigned.
    Occupancy occupancy {position_occupancy(pos)};
    double delta_e {0};
    switch (occupancy) {
        case Occupancy::bound: case Occupancy::misbound:
            throw ConstraintViolation {};
        case Occupancy::unbound:
            update_domain(cd_i, pos, ore);
            try {
                delta_e += bind_domain(cd_i);
            }
            catch (ConstraintViolation) {
                throw;
            }
        case Occupancy::unassigned:
            update_domain(cd_i, pos, ore);
    }
    return delta_e;
}

int OrigamiSystem::add_chain(int c_i_ident) {
    // Add chain with domains in unassigned state and return chain index.
    m_current_uc_i += 1;
    return add_chain(c_i_ident, m_current_uc_i);
}

int OrigamiSystem::add_chain(int c_i_ident, int uc_i) {
    // Add chain with given unique index
    auto c_i {m_chain_lengths.size()};
    m_identity_to_index[c_i_ident].push_back(uc_i);
    m_working_to_unique.push_back(uc_i);
    m_unique_to_working[uc_i] = c_i;
    m_chain_identities.push_back(c_i_ident);

    auto chain_length {m_identities[c_i_ident].size()};
    m_chain_lengths.push_back(chain_length);
    for (int i {0}; i != chain_length; i++) {
        m_positions[c_i].push_back(VectorThree {});
        m_orientations[c_i].push_back(VectorThree {});
    }
    return c_i;
}

void OrigamiSystem::delete_chain(int c_i) {
    // Delete chain c_i
    int uc_i {m_working_to_unique[c_i]};
    int c_i_ident {m_chain_identities[c_i]};
    m_identity_to_index[c_i_ident].erase(m_identity_to_index[c_i_ident].begin()
            + uc_i);
    m_working_to_unique.erase(m_working_to_unique.begin() + c_i);
    m_unique_to_working.erase(uc_i);

    // Update map from unique to working indices
    int uc_j;
    for (int c_j {0}; c_j != m_working_to_unique.size(); c_i++) {
        if (c_j > c_i) {
            int uc_j {m_working_to_unique[c_j]};
            m_unique_to_working[uc_j] = c_j - 1;
        }
        else {
        }
    }
    m_chain_identities.erase(m_chain_identities.begin() + c_i);
    m_positions.erase(m_positions.begin() + c_i);
    m_orientations.erase(m_orientations.begin() + c_i);
    m_chain_lengths.erase(m_chain_lengths.begin() + c_i);
}

// Private methods

void OrigamiSystem::initialize_complementary_associations() {
    // Intialize staple identity to complementary scaffold domains container
    // Staple identities are 1 indexed
    m_staple_ident_to_scaffold_ds[0] = {};
    for (int i {0}; i != m_identities.size(); ++i) {
        m_identity_to_index.push_back({});
        vector<int> staple {m_identities[i]};
        vector<int> scaffold_d_is {};
        for (int j {0}; j != staple.size(); ++j) {
            int staple_d_i {staple[j]};

            // Staple d_is are negatives of scaffold d_is
            int scaffold_d_i {index(m_identities[scaffold_d_i], -staple_d_i)};
            scaffold_d_is.push_back(scaffold_d_i);
        }
        m_staple_ident_to_scaffold_ds.push_back(scaffold_d_is);
    }
}

void OrigamiSystem::initialize_energies() {
    // Calculate and store all energies
    for (int c_i {0}; c_i != m_sequences.size(); c_i++) {
        for (int c_j {0}; c_j != m_sequences.size(); c_j++) {
            for (int d_i {0}; d_i != m_sequences[c_i].size(); d_i++) {
                for (int d_j {0}; d_j != m_sequences[c_j].size(); d_j++) {
                    string seq_i {m_sequences[c_i][d_i]};
                    string seq_j {m_sequences[c_j][d_j]};
                    CDPair cd_i {c_i, d_i};
                    CDPair cd_j {c_j, d_j};
                    pair<CDPair, CDPair> key {cd_i, cd_j};

                    // Hybridization energies
                    vector<string> comp_seqs {find_longest_contig_complement(
                            seq_i, seq_j)};
                    double energy {0};
                    int N {0};
                    for (auto comp_seq: comp_seqs) {
                        energy += calc_hybridization_energy(
                               comp_seq, m_temp, m_cation_M);
                        N++;
                    }
                    energy /= N;
                    m_hybridization_energies[key] = energy;

                    // Stacking energies
                    double s_energy {0};
                    s_energy += calc_stacking_energy(seq_i, seq_j, m_temp,
                            m_cation_M);
                    m_stacking_energies[key] = s_energy;
                }
            }
        }
    }
}

void OrigamiSystem::initialize_config(Chains chains) {
    // Extract configuration from chains
    for (int c_i {0}; c_i != chains.size(); c_i++) {
        Chain chain {chains[c_i]};
        int uc_i {chain.index};
        m_unique_to_working[uc_i] = c_i;
        m_working_to_unique.push_back(uc_i);
        int c_i_ident {chain.identity};
        m_identity_to_index[c_i_ident].push_back(uc_i);
        m_chain_identities.push_back(c_i_ident);
        auto num_domains {m_identities[c_i_ident].size()};
        m_chain_lengths.push_back(num_domains);
        for (int d_i {0}; d_i != num_domains; d_i++) {
            VectorThree pos = chain.positions[d_i];
            VectorThree ore = chain.orientations[d_i];
            set_domain_config(CDPair {c_i, d_i}, pos, ore);
        }
    }
    m_current_uc_i = *max(m_working_to_unique.begin(),
            m_working_to_unique.end());
}

double OrigamiSystem::hybridization_energy(CDPair cd_i, CDPair cd_j) const {
    pair<CDPair, CDPair> key {cd_i, cd_j};
    return m_hybridization_energies.at(key);
}

double OrigamiSystem::stacking_energy(CDPair cd_i, CDPair cd_j) const {
    pair<CDPair, CDPair> key {cd_i, cd_j};
    return m_stacking_energies.at(key);
}

double OrigamiSystem::unassign_bound_domain(CDPair cd_i) {
    CDPair ucd_i {m_working_to_unique[cd_i.c], cd_i.d};
    CDPair ucd_j {domain_bound_to(ucd_i)};
    CDPair cd_j {m_unique_to_working[ucd_j.c], ucd_j.d};

    double delta_e {-hybridization_energy(cd_i, cd_j)};

    m_bound_d_to_bound_d.erase(ucd_i);
    m_bound_d_to_bound_d.erase(ucd_j);
    m_domain_occupancies.erase(ucd_i);

    VectorThree pos {m_positions[cd_i.c][cd_i.d]};
    m_pos_to_unbound_d[pos] = ucd_j;
    m_position_occupancies[pos] = Occupancy::unbound;
    m_domain_occupancies[ucd_j] = Occupancy::unbound;
    return delta_e;
}

void OrigamiSystem::unassign_unbound_domain(CDPair cd_i) {
    VectorThree pos {m_positions[cd_i.c][cd_i.d]};
    m_pos_to_unbound_d.erase(pos);
    m_position_occupancies.erase(pos);
    m_domain_occupancies[cd_i] = Occupancy::unassigned;
}

void OrigamiSystem::update_domain(
        CDPair cd_i,
        VectorThree pos,
        VectorThree ore) {
    m_positions[cd_i.c][cd_i.d] = pos;
    m_orientations[cd_i.c][cd_i.d] = ore;
}    

void OrigamiSystem::update_occupancies(CDPair cd_i, VectorThree pos) {
    Occupancy occupancy {position_occupancy(pos)};
    CDPair ucd_i {m_working_to_unique[cd_i.c], cd_i.d};
    Occupancy new_state;
    switch (occupancy) {
        case Occupancy::unbound: {
            CDPair ucd_j {unbound_domain_at(pos)};
            CDPair cd_j {m_unique_to_working[ucd_j.c], ucd_j.d};
            int c_i_ident {m_chain_identities[cd_i.c]};
            int d_i_ident {m_identities[c_i_ident][cd_i.d]};
            int c_j_ident {m_chain_identities[cd_j.c]};
            int d_j_ident {m_identities[c_i_ident][cd_j.d]};
            if (d_i_ident == -d_j_ident) {
                new_state = Occupancy::bound;
                m_num_fully_bound_domains += 1;
            }
            else {
                new_state = Occupancy::misbound;
            }

            m_pos_to_unbound_d.erase(pos);
            m_domain_occupancies[ucd_i] = new_state;
            m_domain_occupancies[ucd_j] = new_state;
            m_position_occupancies[pos] = new_state;
            m_bound_d_to_bound_d[ucd_j] = ucd_i;
            m_bound_d_to_bound_d[ucd_i] = ucd_j;
            break;
        }
        case Occupancy::unassigned:
            new_state = Occupancy::unbound;
            m_domain_occupancies[ucd_i] = new_state;
            m_position_occupancies[pos] = new_state;
            m_pos_to_unbound_d[pos] = ucd_i;
            break;
    }
}

double OrigamiSystem::bind_domain(CDPair cd_i) {
    // Check constraints for an unbound to (mis)bound transition
    CDPair cd_j {domain_bound_to(cd_i)};
    double delta_e {0};
    bool comp {check_domains_complementary(cd_i, cd_j)};
    if (comp) {
        delta_e += bind_complementary_domains(cd_i, cd_j);
    }
    else {
        delta_e += bind_noncomplementary_domains(cd_i, cd_j);
    }
    return delta_e;
}

double OrigamiSystem::bind_complementary_domains(CDPair cd_i, CDPair cd_j) {
    // cd_i is new
    check_domain_orientations_opposing(cd_i, cd_j);
    check_domain_pair_constraints(cd_i);
    check_domain_pair_constraints(cd_j);

    // Missed one linear helix check per chain
    check_linear_helix_rear(cd_i);
    check_linear_helix_rear(cd_j);

    // Missed two contiguous junction checks per chain
    check_junction_front(cd_i);
    check_junction_front(cd_j);
    check_junction_rear(cd_i);
    check_junction_rear(cd_j);

    // Collect energies
    double delta_e {0};
    delta_e += hybridization_energy(cd_i, cd_j);
    delta_e += check_stacking(cd_i, cd_j);

    return delta_e;
}

bool OrigamiSystem::check_domains_complementary(CDPair cd_i, CDPair cd_j) {
    int c_i_ident {m_chain_identities[cd_i.c]};
    int d_i_ident {m_identities[c_i_ident][cd_i.d]};
    int c_j_ident {m_chain_identities[cd_j.c]};
    int d_j_ident {m_identities[c_j_ident][cd_j.d]};

    bool comp;
    if (d_i_ident == -d_j_ident) {
        comp = true;
    }
    else {
        comp = false;
    }
    return comp;
}

double OrigamiSystem::check_stacking(CDPair cd_new, CDPair cd_old) {
    // Check both sides
    double delta_e {0};
    for (int i: {-1, 0}) {
        CDPair cd_i;
        CDPair cd_j;
        try {
            CDPair cd_i = increment_index(cd_old, i);
            CDPair cd_j = increment_index(cd_i, 1);
        }
        catch (IndexOutOfRange) {
            continue;
        }
        bool cd_j_bound {m_domain_occupancies[cd_j] == Occupancy::bound};
        if (not cd_j_bound) {
            continue;
        }
        CDPair cd_adjacent {domain_bound_to(cd_j)};
        if (cd_adjacent.c == cd_new.c) {
            continue;
        }
        VectorThree pos_i = {m_positions[cd_i.c][cd_i.d]};
        VectorThree pos_j = {m_positions[cd_j.c][cd_j.d]};
        VectorThree ndr {pos_j - pos_i};
        VectorThree ore_i = {m_orientations[cd_i.c][cd_i.d]};
        if (ndr == ore_i) {
            continue;
        }
        delta_e += stacking_energy(cd_new, cd_adjacent);
    }
    return delta_e;
}

void OrigamiSystem::check_domain_pair_constraints(CDPair cd_i) {
    // Check both pairs
    for (int i: {-1, 0}) {
        CDPair cd_1;
        CDPair cd_2;
        try {
            CDPair cd_1 = increment_index(cd_i, i);
            CDPair cd_2 = increment_index(cd_1, 1);
        }
        catch (IndexOutOfRange) {
            continue;
        }
        Occupancy occ_1 {m_domain_occupancies[cd_1]};
        Occupancy occ_2 {m_domain_occupancies[cd_2]};
        if (occ_1 == Occupancy::bound and occ_2 == Occupancy::bound) {
            check_helical_constraints(cd_1, cd_2);
        }
        else {
        }
    }
}

void OrigamiSystem::check_helical_constraints(CDPair cd_1, CDPair cd_2) {
    // Given that d1 and d2 exist and are bound

    // Calculate next domain vector
    VectorThree pos_1 {m_positions[cd_1.c][cd_1.d]};
    VectorThree pos_2 {m_positions[cd_2.c][cd_2.d]};
    VectorThree ndr {pos_2 - pos_1};

    CDPair cd_bound_1 {domain_bound_to(cd_1)};
    CDPair cd_bound_2 {domain_bound_to(cd_2)};
    bool bound_same_chain {cd_bound_1.c == cd_bound_2.c};

    VectorThree ore_1 {m_orientations[cd_1.c][cd_1.d]};
    VectorThree ore_2 {m_orientations[cd_1.c][cd_1.d]};

    // New helix case
    if (ore_1 == ndr) {

        // Check doubly contiguous constraint
        if (bound_same_chain) {
            if (cd_bound_1.d == cd_bound_2.d + 1) {
                throw ConstraintViolation {};
            }
            else {
                check_doubly_contiguous_junction(cd_1, cd_2);
            }
        }
    }

    // Non-physical case
    else if (ore_1 == -ndr) {
        throw ConstraintViolation {};
    }

    // Same helix case
    else {

        // Check double contiguous constraint
        if (bound_same_chain) {
            if (cd_bound_1.d == cd_bound_2.d - 1) {
                throw ConstraintViolation {};
            }
            else {
                check_twist_constraint(ndr, ore_1, ore_2);
                check_linear_helix(ndr, pos_2, ore_2, cd_2);
            }
        }
    }
}

void OrigamiSystem::check_linear_helix_rear(CDPair cd_3) {
    // Check linear helix constraints given rear domain exists and bound
    CDPair cd_1;
    CDPair cd_2;
    try {
        CDPair cd_2 = increment_index(cd_3, -1);
        CDPair cd_1 = increment_index(cd_2, -1);
    }
    catch (IndexOutOfRange) {
        return;
    }
    
    // Domains 1 and 2 not bound
    bool cd_1_bound {m_domain_occupancies[cd_1] == Occupancy::bound};
    bool cd_2_bound {m_domain_occupancies[cd_2] == Occupancy::bound};
    if (not (cd_1_bound and cd_2_bound)) {
        return;
    }
    VectorThree pos_1 {m_positions[cd_1.c][cd_1.d]};
    VectorThree pos_2 {m_positions[cd_2.c][cd_2.d]};
    VectorThree pos_3 {m_positions[cd_3.c][cd_3.d]};
    VectorThree ndr_1 {pos_2 - pos_1};
    VectorThree ndr_2 {pos_3 - pos_2};
    VectorThree ore_1 {m_orientations[cd_1.c][cd_1.d]};
    VectorThree ore_2 {m_orientations[cd_2.c][cd_2.d]};
    // Domains 1 and 2 are junction
    if (ndr_1 == ore_1) {
        return;
    }

    // Domains 2 and 3 are junction
    if (ndr_2 == ore_2) {
        return;
    }

    // Domains are linear
    if (ndr_1 == ndr_2) {
        return;
    }

    throw ConstraintViolation {};
}

void OrigamiSystem::check_linear_helix(
        VectorThree ndr_1,
        VectorThree pos_2,
        VectorThree ore_2,
        CDPair cd_2) {
    CDPair cd_3;
    try {
        CDPair cd_3 = increment_index(cd_2, 1);
    }
    catch (IndexOutOfRange) {
        return;
    }

    // Third domain not bound or unasssigned
    if (m_domain_occupancies[cd_3] != Occupancy::bound) {
        return;
    }

    // Third domain part of new helix
    VectorThree pos_3 {m_positions[cd_3.c][cd_3.d]};
    VectorThree ndr_2 {pos_3 - pos_2};
    if (ndr_2 == ore_2) {
        return;
    }

    // Third domain linear
    if (ndr_1 == ndr_2) {
        return;
    }

    throw ConstraintViolation {};
}

void OrigamiSystem::check_junction_front(CDPair cd_1) {
    // Check junction given d1 exists
    CDPair cd_2;
    CDPair cd_3;
    CDPair cd_4;
    try {
        CDPair cd_2 = increment_index(cd_1, 1);
        CDPair cd_3 = increment_index(cd_1, 2);
        CDPair cd_4 = increment_index(cd_1, 3);
    }
    catch (IndexOutOfRange) {
        return;
    }

    // Check that all domains in bound state
    bool cd_2_bound {m_domain_occupancies[cd_2] == Occupancy::bound};
    bool cd_3_bound {m_domain_occupancies[cd_3] == Occupancy::bound};
    bool cd_4_bound {m_domain_occupancies[cd_4] == Occupancy::bound};
    if (not (cd_2_bound and cd_3_bound and cd_4_bound)) {
        return;
    }

    if (not doubly_contiguous_junction(cd_2, cd_3)) {
        return;
    }

    return check_doubly_contiguous_junction(cd_1, cd_2, cd_3, cd_4);
}

void OrigamiSystem::check_junction_rear(CDPair cd_4) {
    // Check junction given d4 exists
    CDPair cd_1;
    CDPair cd_2;
    CDPair cd_3;
    try {
        CDPair cd_3 = increment_index(cd_4, -1);
        CDPair cd_2 = increment_index(cd_4, -2);
        CDPair cd_1 = increment_index(cd_4, -3);
    }
    catch (IndexOutOfRange) {
        return;
    }

    // Check that all domains in bound state
    bool cd_1_bound {m_domain_occupancies[cd_1] == Occupancy::bound};
    bool cd_2_bound {m_domain_occupancies[cd_2] == Occupancy::bound};
    bool cd_3_bound {m_domain_occupancies[cd_3] == Occupancy::bound};
    if (not (cd_1_bound and cd_2_bound and cd_3_bound)) {
        return;
    }

    if (not doubly_contiguous_junction(cd_2, cd_3)) {
        return;
    }

    return check_doubly_contiguous_junction(cd_1, cd_2, cd_3, cd_4);
}

bool OrigamiSystem::doubly_contiguous_junction(CDPair cd_1, CDPair cd_2) {
    // Given that d1 and d2 exist and are bound

    // Check doubly contiguous
    CDPair cd_bound_1 {domain_bound_to(cd_1)};
    CDPair cd_bound_2 {domain_bound_to(cd_2)};
    if (not cd_bound_1.c == cd_bound_2.c) {
        return false;
    }

    if (not cd_bound_1.d == cd_bound_2.d - 1) {
        return false;
    }

    // Check junction
    VectorThree pos_1 {m_positions[cd_1.c][cd_1.d]};
    VectorThree pos_2 {m_positions[cd_1.c][cd_1.d]};
    VectorThree ndr {pos_2 - pos_1};
    VectorThree ore_1 {m_orientations[cd_1.c][cd_1.d]};
    if (ndr != ore_1) {
        return false;
    }

    return true;
}

void OrigamiSystem::check_doubly_contiguous_junction(CDPair cd_2, CDPair cd_3) {
    // Already know d_2 and d_3 are doubly contiguous junction
    CDPair cd_1;
    CDPair cd_4;
    try {
        CDPair cd_1 {increment_index(cd_2, -1)};
        CDPair cd_4 {increment_index(cd_3, 1)};
    }
    catch (IndexOutOfRange) {
        return;
    }

    bool cd_1_bound {m_domain_occupancies[cd_1] == Occupancy::bound};
    bool cd_4_bound {m_domain_occupancies[cd_4] == Occupancy::bound};
    if (not (cd_1_bound and cd_4_bound)) {
        return;
    }
    return check_doubly_contiguous_junction(cd_1, cd_2, cd_3, cd_4);
}

void OrigamiSystem::check_doubly_contiguous_junction(
        CDPair cd_1,
        CDPair cd_2,
        CDPair cd_3,
        CDPair cd_4) {
    // Calling functions already check for existance and boundeness of d1-4.
    VectorThree pos_1 {m_positions[cd_1.c][cd_1.d]};
    VectorThree pos_2 {m_positions[cd_2.c][cd_2.d]};
    VectorThree pos_3 {m_positions[cd_3.c][cd_3.d]};
    VectorThree pos_4 {m_positions[cd_4.c][cd_4.d]};
    VectorThree ndr_1 {pos_2 - pos_1};
    VectorThree ndr_3 {pos_4 - pos_3};
    if (ndr_1 == -ndr_3) {
        return;
    }
    throw ConstraintViolation {};
}

void OrigamiSystem::check_domain_orientations_opposing(CDPair cd_i, CDPair cd_j) {
    VectorThree ore_i {m_orientations[cd_i.c][cd_i.d]};
    VectorThree ore_j {m_orientations[cd_j.c][cd_j.d]};
    if (ore_i != -ore_j) {
        throw ConstraintViolation {};
    }
    else {
    }
}

template<typename Origami_T>
CDPair OrigamiSystemLinearScaffold<Origami_T>::increment_index(CDPair cd_i, int incr) {
    int d_j {cd_i.d + incr};
    if (d_j >= OrigamiSystem::chain_lengths()[cd_i.c] or d_j < 0) {
        throw IndexOutOfRange {};
    }
    return CDPair {cd_i.c, d_j};
}

template<typename Origami_T>
CDPair OrigamiSystemCyclicScaffold<Origami_T>::increment_index(CDPair cd_i, int incr) {
    int d_j {cd_i.d + incr};
    if (cd_i.c == OrigamiSystem::c_scaffold) {
        d_j %= OrigamiSystem::chain_lengths()[OrigamiSystem::c_scaffold];
    }
    else {
        if (d_j >= OrigamiSystem::chain_lengths()[cd_i.c] or d_j < 0) {
            throw IndexOutOfRange {};
        }
    }
    return CDPair {cd_i.c, d_j};
}

template<typename Origami_T>
double OrigamiSystemWithMisbinding<Origami_T>::bind_noncomplementary_domains(
        CDPair cd_i,
        CDPair cd_j) {
    return OrigamiSystem::hybridization_energy(cd_i, cd_j);
}

template<typename Origami_T>
double OrigamiSystemWithoutMisbinding<Origami_T>::bind_noncomplementary_domains(
        CDPair cd_i,
        CDPair cd_j) {
    throw ConstraintViolation {};
}

template<typename Origami_T>
void OrigamiSystemSixteen<Origami_T>::check_twist_constraint(
        VectorThree ndr,
        VectorThree ore_1,
        VectorThree ore_2) {
    VectorThree ore_1_rotated {ore_1.rotate_half(ndr)};
    if (ore_1 == ore_2) {
        return;
    }
    else {
        throw ConstraintViolation {};
    }
}
