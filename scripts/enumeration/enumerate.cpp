// enumerate.cpp

#include "enumerate.h"
#include "parser.h"
#include "domain.h"
#include "origami_system.h"
#include "files.h"

using namespace Parser;
using namespace DomainContainer;
using namespace Origami;
using namespace Files;

int main(int argc, char* argv[]) {
    cout << "WARNING: Not for staples other than length 2.\n";
    InputParameters input_parameters {argc, argv};

    // Create origami object
    OrigamiInputFile origami_input {input_parameters.m_origami_input_filename};
    vector<vector<int>> identities {origami_input.m_identities};
    vector<vector<string>> sequences {origami_input.m_sequences};
    vector<Chain> configs {origami_input.m_chains};
    OrigamiSystem origami {
            identities,
            sequences,
            configs,
            input_parameters.m_temp,
            input_parameters.m_staple_M,
            input_parameters.m_cation_M,
            input_parameters.m_lattice_site_volume,
            input_parameters.m_cyclic};

	// Enumerate configurations
    ConformationalEnumerator conf_enumerator {origami};

    // Hack for staples
    vector<pair<int, int>> staples {{1, 1}, {1, 1}};
    GrowthpointEnumerator growthpoint_enumerator {conf_enumerator, staples, origami};
    growthpoint_enumerator.enumerate();
    print_matrix(conf_enumerator.bound_state_weights());
}

void print_matrix(vector<vector<double>> matrix) {
    for (auto row: matrix) {
        for (auto element: row) {
            cout << element << " ";
        }
    cout << "\n";
    }
}

GrowthpointEnumerator::GrowthpointEnumerator(
        ConformationalEnumerator& conformational_enumerator,
        vector<pair<int, int>> staples,
        OrigamiSystem& origami_system):
        m_conformational_enumerator {conformational_enumerator},
        m_staples {staples},
        m_origami_system {origami_system} {

    m_conformational_enumerator.set_staples(staples);
}

void GrowthpointEnumerator::enumerate() {

    // Iterate through staple identities
    for (size_t i {0}; i != m_staples.size(); i++) {

        // Update staple identity list
        int staple_ident {m_staples[i].first};
        int num_remaining {m_staples[i].second - 1};
        if (num_remaining == 0) {
            m_staples.erase(m_staples.begin() + i);
        }
        else {
            m_staples[i] = {staple_ident, num_remaining};
        }

        // Iterate through system domains for growthpoint
        for (size_t j {0}; j != m_unbound_system_domains.size(); j++) {

            // Update unbound system domain list
            Domain* old_domain {m_unbound_system_domains[j]};
            m_unbound_system_domains.erase(m_unbound_system_domains.begin() + j);
            size_t staple_length {m_origami_system.m_identities[staple_ident].size()};

            // Iterate through staple domains for growthpoint
            for (size_t d_i {0}; d_i != staple_length; d_i++) {

                // Update unbound system domain list and recurse if staples remain
                m_growthpoints.push_back({{staple_ident, d_i},
                        {old_domain->m_c, old_domain->m_d}});
                vector<Domain*> new_unbound_domains {
                        m_conformational_enumerator.add_growthpoint(staple_ident,
                                d_i, old_domain)};
                if (not m_staples.empty()) {
                    m_unbound_system_domains.insert(m_unbound_system_domains.end(),
                            new_unbound_domains.begin(), new_unbound_domains.end());
                    enumerate();

                    // Revert unbound domain list
                    m_conformational_enumerator.remove_growthpoint(old_domain);
                    for (size_t k {0}; k != staple_length - 1; k++) {
                        m_unbound_system_domains.pop_back();
                    }
                }

                // Otherwise enumerate conformations
                else {

                    // Skip if growthpoint set already enumerated
                    if (not growthpoints_repeated()) {
                        m_conformational_enumerator.enumerate();
                        m_enumerated_growthpoints.push_back(m_growthpoints);
                    }
                }
                m_growthpoints.pop_back();
            }

            // Revert unbound domain list
            m_unbound_system_domains.insert(m_unbound_system_domains.begin() + j,
                    old_domain);
        }
        
        // Revert staple identity list
        m_staples.insert(m_staples.begin() + i, {staple_ident, num_remaining + 1});
    }
}

bool GrowthpointEnumerator::growthpoints_repeated() {
    bool repeated {false};
    size_t i {0};
    while (not repeated and i != m_enumerated_growthpoints.size()) {
        auto growthpoints {m_enumerated_growthpoints[i]};
        repeated = true;
        for (auto growthpoint: growthpoints) {
            if (count(m_growthpoints.begin(), m_growthpoints.end(), growthpoint) == 0) {
                repeated = false;
                break;
            }
        }
        i++;
    }
    return repeated;
}

ConformationalEnumerator::ConformationalEnumerator(OrigamiSystem& origami_system) :
        m_origami_system {origami_system} {

    // Unassign all domains
    for (auto chain: m_origami_system.m_domains) {
        for (auto domain: chain) {
            m_origami_system.unassign_domain(*domain);
        }
    }

    // Set first domain to origin
	m_origami_system.set_domain_config(*m_origami_system.m_domains[0][0], {0, 0, 0}, {1, 0, 0});

    // Delete all staple chains
    if (m_origami_system.num_staples() > 0) {
        for (size_t i {1}; i != m_origami_system.m_domains.size(); i ++) {
            m_origami_system.delete_chain(m_origami_system.m_chain_indices[i]);
        }
    }
}

void ConformationalEnumerator::enumerate() {
    create_domains_stack();
    Domain* starting_domain {m_domains.back()};
    VectorThree origin {0, 0, 0};
    enumerate_domain(starting_domain, origin);
}

void ConformationalEnumerator::set_staples(vector<pair<int, int>> staples) {
    for (auto staple: staples) {

        // Only one of the N! combos is calculated, so don't divide by N!
        m_prefix *= pow(1 / m_origami_system.m_volume, staple.second);

        for (int i {0}; i != staple.second; i++) {
            int c_i {m_origami_system.add_chain(staple.first)};
            m_identity_to_indices[staple.first].push_back(c_i);

            // Update bound states matrix to correct size
            m_bound_state_weights.push_back({});
            size_t staple_length {m_origami_system.m_identities[staple.first].size()};
            for (size_t j {0}; j != m_bound_state_weights.size(); j++) {
                for (size_t k {0}; k != staple_length; k++) {
                    m_bound_state_weights[j].push_back(0);
                }
            }
        }
    }
}

vector<Domain*> ConformationalEnumerator::add_growthpoint(
        int new_c_ident,
        int new_d_i,
        Domain* old_domain) {

    // Remove staple from stack of available
    int new_c_i {m_identity_to_indices[new_c_ident].back()};
    m_identity_to_indices[new_c_ident].pop_back();

    int new_c_i_index {index(m_origami_system.m_chain_indices, new_c_i)};
    vector<Domain*> staple {m_origami_system.m_domains[new_c_i_index]};
    Domain* new_domain {staple[new_d_i]};
    m_growthpoints[old_domain] = new_domain;

    // Create list of unbound domains on staple
    staple.erase(staple.begin() + new_d_i);

    return staple;
}

void ConformationalEnumerator::remove_growthpoint(
        Domain* old_domain) {
    Domain* new_domain {m_growthpoints[old_domain]};
    m_growthpoints.erase(old_domain);

    // Add staple to stack of available staples
    m_identity_to_indices[new_domain->m_c_ident].push_back(new_domain->m_c);
}

vector<vector<double>> ConformationalEnumerator::bound_state_weights() {
    auto normalized_weights {m_bound_state_weights};
    for (size_t i {0}; i != m_bound_state_weights.size(); i++) {
        for (size_t j {0}; j != m_bound_state_weights[i].size(); j++) {
            normalized_weights[i][j] /= m_partition_f;
        }
    }
    return normalized_weights;
}

void ConformationalEnumerator::enumerate_domain(Domain* domain, VectorThree p_prev) {
    int multiplier {1};

    // Iterate through all positions
    for (auto p_vec: vectors) {
        VectorThree p_new = p_prev + p_vec;

        // Check if domain will be entering a bound state
        bool is_growthpoint {m_growthpoints.count(domain) > 0};
        Occupancy p_occ {m_origami_system.position_occupancy(p_new)};
        bool is_occupied {p_occ == Occupancy::bound or p_occ == Occupancy::misbound};

        // Cannot have more than two domains bound at a given position
        if (is_growthpoint and is_occupied) {
            continue;
        }
        else if (is_growthpoint or is_occupied) {
            Domain* other_domain;
            if (is_growthpoint) {

                // Store position for terminal domains
                m_prev_growthpoint_p = p_new;
                other_domain = m_growthpoints[domain];
            }
            else {
                other_domain = m_origami_system.unbound_domain_at(p_new);
                multiplier *= calculate_multiplier(domain, other_domain);
            }

            bool domains_complementary {m_origami_system.check_domains_complementary(
                    *domain, *other_domain)};
            if (domains_complementary) {
                for (auto o_new: vectors) {
                    m_energy += m_origami_system.set_domain_config(*domain, p_new, o_new);
                }
            }

            // Can skip iteration over orientations if not complementary
            else {
                multiplier *= 6;
                m_energy += m_origami_system.set_domain_config(*domain, p_new, {0, 0, 0});
            }
        }

        // Continue growing next domain
        m_multiplier *= multiplier;
        if (not m_domains.empty()) {
            Domain* next_domain {m_domains.back()};
            m_domains.pop_back();
            VectorThree new_p_prev;

            // Previous domain is last growthpoint if end-of-staple reached
            bool domain_is_terminal {domain->m_c != next_domain->m_c};
            if (domain_is_terminal) {
                new_p_prev = m_prev_growthpoint_p;
            }
            else {
                new_p_prev = p_new;
            }
            enumerate_domain(domain, new_p_prev);
        }

        // Save relevant values if system fully grown
        else {
            calc_and_save_weights();
        }
        m_multiplier /= multiplier;
        m_origami_system.unassign_domain(*domain);
        m_domains.push_back(domain);
    }

    return;
}

void ConformationalEnumerator::create_domains_stack() {
    for (size_t i {1}; i != m_origami_system.m_domains[0].size(); i++) {
        Domain* domain {m_origami_system.m_domains[0][i]};
        bool is_growthpoint {m_growthpoints.count(domain) > 0};
        m_domains.push_back(domain);
        if (is_growthpoint) {
            Domain* new_domain {m_growthpoints[domain]};
            create_staple_stack(new_domain);
        }
    }
	m_domains.erase(m_domains.begin());
    std::reverse(m_domains.begin(), m_domains.end());
}

void ConformationalEnumerator::create_staple_stack(Domain* domain) {

	// Add growth domain
	m_domains.push_back(domain);

	int c_i_index {index(m_origami_system.m_chain_indices, domain->m_c)};
	vector<Domain*> staple {m_origami_system.m_domains[c_i_index]};

    // Add domains in three prime direction (staple domains increase in 3' direction)
    auto first_iter3 {staple.begin() + domain->m_d + 1};
    auto last_iter3 {staple.end()};
	m_domains.insert(m_domains.end(), first_iter3, last_iter3);

    // Add domains in five prime direction
    auto first_iter5 {staple.begin()};
    auto last_iter5 {staple.begin() + domain->m_d};
    vector<Domain*> domains_five_prime {first_iter5, last_iter5};
    std::reverse(domains_five_prime.begin(), domains_five_prime.end());
    auto first_iter5_reverse {domains_five_prime.begin()};
    auto last_iter5_reverse {domains_five_prime.end()};
	m_domains.insert(m_domains.end(), first_iter5_reverse, last_iter5_reverse);

    return;
}

double ConformationalEnumerator::calculate_multiplier(Domain* domain, Domain* other_domain) {
    // This only works for staples that are only 2 domains

    int multiplier {1};
    //  No overcounting if binding to self
    if (domain->m_c == other_domain->m_c) {
        multiplier = 1;
    }
    else {
        int involved_staples {0};

        // Count staples associated with domain
        bool domain_on_scaffold {false};
        if (domain->m_c == 0) {
            domain_on_scaffold = true;
        }
        else {
            involved_staples++;
        }
        Domain* next_domain {domain};
        while (not domain_on_scaffold) {
            if (next_domain->m_forward_domain == nullptr) {
                next_domain = next_domain->m_backward_domain;
            }
            else {
                next_domain = next_domain->m_forward_domain;
            }
            if (next_domain->m_c == 0) {
                domain_on_scaffold = true;
            }
            else {
                involved_staples++;
            }
        }

        // Count staples associated with other domain
        domain_on_scaffold = false;
        if (other_domain->m_c == 0) {
            domain_on_scaffold = true;
        }
        else {
            involved_staples++;
        }
        next_domain = other_domain;
        while (not domain_on_scaffold) {
            if (next_domain->m_forward_domain == nullptr) {
                next_domain = next_domain->m_backward_domain;
            }
            else {
                next_domain = next_domain->m_forward_domain;
            }
            if (next_domain->m_c == 0) {
                domain_on_scaffold = true;
            }
            else {
                involved_staples++;
            }
        }
        multiplier = 1 / (involved_staples + 1);
    }
    return multiplier;
}

void ConformationalEnumerator::calc_and_save_weights() {
    double weight {m_prefix * exp(-m_energy) * m_multiplier};
    m_partition_f += weight;
    m_bound_state_weights[m_origami_system.num_staples()][
            m_origami_system.num_fully_bound_domains()] += weight;
}
