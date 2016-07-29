// origami_system.cpp

#include <algorithm>

#include "origami_system.h"
#include "nearest_neighbour.h"
#include "utility.h"

using std::max;

using namespace NearestNeighbour;
using Utility::index;
using namespace Origami;

OrigamiSystem::OrigamiSystem(
        const vector<vector<int>>& identities,
        const vector<vector<string>>& sequences,
        const Chains& chains,
        bool cyclic,
        bool misbinding,
        bool stacking,
        double temp,
        double cation_M,
        double strand_M) :

        m_identities {identities},
        m_sequences {sequences},
        m_cyclic {cyclic},
        m_misbinding {misbinding},
        m_stacking {stacking},
        m_temp {temp},
        m_cation_M {cation_M},
        m_strand_M {strand_M} {

    initialize_complementary_associations();
    initialize_hybridization_energies();
    initialize_config(chains);
}

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

void OrigamiSystem::initialize_hybridization_energies() {
    // Calculate and store all hybridization energies
    for (int c_i {0}; c_i != m_sequences.size(); c_i++) {
        for (int c_j {0}; c_j != m_sequences.size(); c_j++) {
            for (int d_i {0}; d_i != m_sequences[c_i].size(); d_i++) {
                for (int d_j {0}; d_j != m_sequences[c_j].size(); d_j++) {
                    string seq_i {m_sequences[c_i][d_i]};
                    string seq_j {m_sequences[c_j][d_j]};
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
                    CDPair cd_i {c_i, d_i};
                    CDPair cd_j {c_j, d_j};
                    pair<CDPair, CDPair> key {cd_i, cd_j};
                    m_hybridization_energies[key] = energy;
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

double OrigamiSystem::set_domain_config(
        CDPair cd_i,
        VectorThree pos,
        VectorThree ore) {
    // Check constraints and update if obeyed, otherwise throw
    delta_e = check_domain_constraints(cd_i, pos, ore);
    update_occupancies(cd_i, pos);
    return delta_e;
}

double OrigamiSystem::check_domain_config(
        CDPair cd_i,
        VectorThree pos,
        VectorThree ore) {
    // Check constraints obeyed, otherwise throw
    delta_e = check_domain_constraints();
    revert_next_domains();
    return delta_e;

void OrigamiSystem::set_checked_domain(
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
    // constraint violation. This could be dangerous.
    Occupancy occupancy {position_occupancy(pos)};
    double delta_e {0};
    switch (occupancy) {
        case Occupancy::bound or Occupancy::misbound:
            throw ConstraintViolation {};
        case Occuapncy::unbound:
            update_domains(cd_i);
            try {
                delta_e += bind_domain(cd_i);
            }
            catch (ConstraintViolation) {
                revert_next_domains();
                throw;
            }
        case Occupancy::unassigned:
            update_domains(cd_i);
    }
    return delta_e;
}

void add_chain(int c_i_ident) {
    // Add chain with domains in unassigned state and return chain index.
    m_current_uc_i += 1;
    return add_chain(c_i_ident, m_current_uc_i);

void add_chain(int c_i_ident, int uc_i) {
    // Add chain with given unique index
    c_i = self.chain_lengths.size();
    m_identity_to_index[c_i_ident].push_back(uc_i);
    m_working_to_unique.push_back(uc_i);
    m_unique_to_working[uc_i] = c_i;
    m_chain_identities.push_back(c_i_ident);

    int chain_length {m_identities[c_i_ident].size()};
    m_chain_lengths.push_back(chain_length);
    for (int i {0}; i != chain_length; i++) {
        m_positions.push_back(VectorThree {});
        m_orientations.push_back(VectorThree {});
        m_next_domains.push_back(VectorThree {}):
    }
    return c_i
}

void delete_chain(c_i) {
	// Delete chain c_i
    int uc_i m_working_to_unique[c_i];
    int c_i_ident {m_chain_identities[c_i]};
    m_identity_to_index[identity].erase(uc_i);
    m_working_to_unique.erase(chain_index);
    m_unique_to_working.erase(unique_index);

    // Update map from unique to working indices
	int uc_j;
    for (int c_j {0}; i != m_working_to_unique.size(); i++) {
        if (c_j > c_i) {
            uc_j {m_working_to_unique[c_j]};
            m_unique_to_working[uc_j] = c_j - 1;
        }
        else {
        }
    }
    m_chain_identities.erase(c_i);
    m_positions.erase(c_i);
    m_orientations.erase(c_i);
    m_next_domains.erase(c_i);
    m_chain_lengths.erase(c_i);
}

void OrigamiSystem::update_domain(
        CDPair cd_i,
        VectorThree pos,
        VectorThree ore) {
    int c_i {cd_i.first};
    int d_i {cd_i.second};
    m_positions[c_i][d_i] = pos;
    m_orientations[c_i][d_i] = ore;
    self._prev_next_domains = []
    for (int i {-1}; i != 1; i++) {
        int d_j {d_i + i};
        if d_j < 0 or d_j >= m_chain_lengths[c_i]:
            if (self.cyclic and c_i == scaffold_c_i) {
                d_j = self.wrap_cyclic_scaffold(d_j);
            }
            else {
                continue;
            }

        p_j = domain_position[c_i][d_j]
        d_k = d_j + 1
        if d_k < 0 or d_k >= m_chain_lengths[c_i]:
            if (self.cyclic and c_i == scaffold_c_i) {
                d_k = self.wrap_cyclic_scaffold(d_k);
            }
            else {
                prev_ndr = m_next_domains[chain_i][d_j];
                m_prev_next_domains.(((chain_i, d_j), prev_ndr));
                m_next_domains[chain_i][d_j] = {VectorThree {}};
                continue;
            }

        p_j = self.get_domain_position(chain_i, d_j)
        if p_i == [] or p_j == []:
            prev_ndr = self._next_domains[chain_i][d_j]
            self._prev_next_domains.append(((chain_i, d_j), prev_ndr))
            ndr = np.zeros(3)
            self._next_domains[chain_i][d_j] = ndr
            #self._next_domains[chain_i][d_j] = []
            continue

        ndr = p_j - p_i

        prev_ndr = self._next_domains[chain_i][d_i]
        self._prev_next_domains.append(((chain_i, d_i), prev_ndr))
        self._next_domains[chain_i][d_i] = ndr
}    

void OrigamiSystem::update_occupancies(CDPair cd_i, VectorThree pos) {
    Occupancy occupancy {position_occupancy(pos)};
    uc_i = m_working_to_unique[c_i];
    CDPair ucd_i = (uc_i, d_i);
    Occupancy new_state;
    switch (occupancy) {
        case Occupancy::Unbound:
            CDPair ucd_j = m_unbound_domains[pos];
            uc_j = ucd_j.first;
            c_j = m_unique_to_working[c_uj];
            c_i_ident = m_chain_identities[c_i]
            d_i_ident = m_identities[tc_ident][d_i]
            c_j_ident = m_chain_identities[occuyping_c_i]
            d_j_ident = m_identities[oc_ident][occupying_d_i]
            if d_i_ident == -d_j_ident:
                new_state {Occupancy::bound};
                m_fully_bound_domains += 1;
            else:
                new_state {Occupancy::misbound};

            m_unbound_domains.erase(pos);
            m_domain_occupancies[ucd_i] = new_state;
            m_domain_occupancies[ucd_j] = new_state;
            m_position_occupancies[pos] = new_state
            m_bound_domains[ucd_j] = ucd_i;
            m_bound_domains[ucd_i] = ucd_j;

        case Occupancy::Unassigned:

            new_state {Occupancy::unbound};
            m_domain_occupancies[ucd_i] = new_state;
            m_position_occupancies[pos] = new_state;
            m_unbound_domains[pos] = ucd_i;
    }
}
