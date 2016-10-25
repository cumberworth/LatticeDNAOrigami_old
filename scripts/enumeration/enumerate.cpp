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
    cout << "\nWARNING: Not for staples other than length 2.\n\n";
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
    ConformationalEnumerator conf_enumerator {origami, 2};
    //conf_enumerator.add_staple(1);
    //conf_enumerator.add_staple(2);
    //GrowthpointEnumerator growthpoint_enumerator {conf_enumerator, origami};
    //growthpoint_enumerator.enumerate();
    //conf_enumerator.enumerate();
    conf_enumerator.add_staple(1);
    //GrowthpointEnumerator growthpoint_enumerator10 {conf_enumerator, origami};
    //growthpoint_enumerator10.enumerate();
    conf_enumerator.add_staple(1);
    GrowthpointEnumerator growthpoint_enumerator20 {conf_enumerator, origami};
    growthpoint_enumerator20.enumerate();
    //conf_enumerator.remove_staple(1);
    //conf_enumerator.remove_staple(1);
    //conf_enumerator.add_staple(2);
    //GrowthpointEnumerator growthpoint_enumerator01 {conf_enumerator, origami};
    //growthpoint_enumerator01.enumerate();
    //conf_enumerator.add_staple(2);
    //GrowthpointEnumerator growthpoint_enumerator02 {conf_enumerator, origami};
    //growthpoint_enumerator02.enumerate();
    //conf_enumerator.remove_staple(2);
    //conf_enumerator.add_staple(1);
    //GrowthpointEnumerator growthpoint_enumerator11 {conf_enumerator, origami};
    //growthpoint_enumerator11.enumerate();
    //conf_enumerator.add_staple(1);
    //GrowthpointEnumerator growthpoint_enumerator21 {conf_enumerator, origami};
    //growthpoint_enumerator21.enumerate();
    //conf_enumerator.remove_staple(1);
    //conf_enumerator.add_staple(2);
    //GrowthpointEnumerator growthpoint_enumerator12 {conf_enumerator, origami};
    //growthpoint_enumerator12.enumerate();
    //conf_enumerator.add_staple(1);
    //GrowthpointEnumerator growthpoint_enumerator22 {conf_enumerator, origami};
    //growthpoint_enumerator22.enumerate();
    print_matrix(conf_enumerator.normalize_weights(
                    conf_enumerator.m_bound_state_weights),
            input_parameters.m_counts_output_filename);
    cout << conf_enumerator.m_num_configs << "\n";
    cout << "\n";
    cout << conf_enumerator.average_energy() << "\n";
}

void print_matrix(vector<vector<double>> matrix, string filename) {
    ofstream output {filename};
    //cout << "Staples vs number of fully bound domains\n";
    for (auto row: matrix) {
        for (auto element: row) {
            output << element << " ";
        }
    output << "\n";
    }
}

GrowthpointEnumerator::GrowthpointEnumerator(
        ConformationalEnumerator& conformational_enumerator,
        OrigamiSystem& origami_system):
        m_conformational_enumerator {conformational_enumerator},
        m_origami_system {origami_system} {

    m_unbound_system_domains = m_origami_system.get_chain(0);
    for (size_t c_ident {1}; c_ident != m_origami_system.m_identities.size(); c_ident++) {
        int num_staples_c_i {m_origami_system.num_staples_of_ident(c_ident)};
        if (num_staples_c_i != 0) {
            m_staples.push_back({c_ident, num_staples_c_i});
        }
    }
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
                    for (size_t k {0}; k != staple_length - 1; k++) {
                        m_unbound_system_domains.pop_back();
                    }
                }

                // Otherwise enumerate conformations
                else {

                    // Skip if growthpoint set already enumerated
                    if (not growthpoints_repeated()) {
                        m_conformational_enumerator.enumerate();
                        cout << "Growthpoint set " << m_enumerated_growthpoints.size() + 1 << "\n";
                        m_enumerated_growthpoints.push_back(m_growthpoints);
                    }
                }
                m_conformational_enumerator.remove_growthpoint(old_domain);
                m_growthpoints.pop_back();
            }

            // Revert unbound domain list
            m_unbound_system_domains.insert(m_unbound_system_domains.begin() + j,
                    old_domain);
        }
        
        // Revert staple identity list
        bool staple_remain {false};
        for (size_t staple_ident_i {0}; staple_ident_i != m_staples.size(); staple_ident_i++) {
            if (m_staples[staple_ident_i].first == staple_ident) {
                m_staples[staple_ident_i].second++;
                staple_remain = true;
                break;
            }
        }
        if (not staple_remain) {
            m_staples.insert(m_staples.begin() + i, {staple_ident, num_remaining + 1});
        }
    }
}

bool GrowthpointEnumerator::growthpoints_repeated() {
    bool repeated {false};
    size_t i {0};
    while (not repeated and i != m_enumerated_growthpoints.size()) {
        auto growthpoints {m_enumerated_growthpoints[i]};
        if (growthpoints.size() != m_growthpoints.size()) {
            i++;
            continue;
        }
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

ConformationalEnumerator::ConformationalEnumerator(OrigamiSystem& origami_system,
        int max_num_staples ) :
        m_origami_system {origami_system} {

    // Unassign all domains
    auto all_chains {m_origami_system.get_chains()};
    for (auto chain: all_chains) {
        for (auto domain: chain) {
            m_origami_system.unassign_domain(*domain);
        }
    }

    // Delete all staple chains
    if (m_origami_system.num_staples() > 0) {
        for (size_t i {1}; i != all_chains.size(); i ++) {
            m_origami_system.delete_chain(all_chains[i][0]->m_c);
        }
    }

    // Setup the scaffold count for skipping orientations
    for (auto d_ident: m_origami_system.m_identities[0]) {
        m_identities_to_num_unassigned[d_ident] = 1;
    }
    for (int i {0}; i != max_num_staples + 1; i++) {
        add_weight_matrix_entry();
    }
}

void ConformationalEnumerator::enumerate() {
    create_domains_stack();
    Domain* starting_domain {m_domains.back()};
    m_domains.pop_back();
    VectorThree p_new {0, 0, 0};
    VectorThree o_new {1, 0, 0};
    m_identities_to_num_unassigned[starting_domain->m_d_ident] -= 1;
    m_origami_system.set_domain_config(*starting_domain, p_new, o_new);
    bool is_growthpoint {m_growthpoints.count(starting_domain) > 0};
    Domain* next_domain {m_domains.back()};
    m_domains.pop_back();
    if (is_growthpoint) {
        m_prev_growthpoint_p = p_new;
        bool domains_complementary {m_origami_system.check_domains_complementary(
                *starting_domain, *next_domain)};
        double pos_multiplier {1};
        if (domains_complementary) {
            o_new = -o_new;
        }
        else {
            pos_multiplier = 6;
            VectorThree o_new {0, 0, 0};
        }
        m_multiplier *= pos_multiplier;
        m_identities_to_num_unassigned[next_domain->m_d_ident] -= 1;
        m_energy += m_origami_system.set_domain_config(*next_domain, p_new, o_new);
        Domain* next_next_domain = m_domains.back();
        m_domains.pop_back();
        enumerate_domain(next_next_domain, p_new);
        m_identities_to_num_unassigned[next_domain->m_d_ident] += 1;
        m_multiplier /= pos_multiplier;
        m_energy += m_origami_system.unassign_domain(*next_domain);
    }
    else {
        enumerate_domain(next_domain, p_new);
    }
    m_identities_to_num_unassigned[starting_domain->m_d_ident] += 1;
    m_origami_system.unassign_domain(*starting_domain);
}

void ConformationalEnumerator::add_staple(int staple) {

    // Only one of the N! combos is calculated, so don't divide by N!
    m_prefix /= m_origami_system.m_volume;

    int c_i {m_origami_system.add_chain(staple)};
    m_identity_to_indices[staple].push_back(c_i);
    for (auto d_ident: m_origami_system.m_identities[staple]) {
        if (m_identities_to_num_unassigned.count(d_ident) == 0) {
            m_identities_to_num_unassigned[d_ident] = 1;
        }
        else {
            m_identities_to_num_unassigned[d_ident]++;
        }
    }
}

void ConformationalEnumerator::remove_staple(int staple) {

    // Only one of the N! combos is calculated, so don't divide by N!
    m_prefix *= m_origami_system.m_volume;

    int c_i {m_identity_to_indices[staple].back()};
    m_identity_to_indices[staple].pop_back();
    m_origami_system.delete_chain(c_i);
    for (auto d_ident: m_origami_system.m_identities[staple]) {
        m_identities_to_num_unassigned[d_ident]--;
    }
}

void ConformationalEnumerator::add_weight_matrix_entry() {
    // Update bound states matrix to correct size
    size_t max_fully_bound_domain_pairs {m_origami_system.m_identities[0].size() + 1};
    m_bound_state_weights.push_back({});
    m_misbound_state_weights.push_back({});
    for (size_t i {0}; i != max_fully_bound_domain_pairs; i++) {
        m_bound_state_weights.back().push_back(0);
        m_misbound_state_weights.back().push_back(0);
    }
}

vector<Domain*> ConformationalEnumerator::add_growthpoint(
        int new_c_ident,
        int new_d_i,
        Domain* old_domain) {

    // Remove staple from stack of available
    int new_c_i {m_identity_to_indices[new_c_ident].back()};
    m_identity_to_indices[new_c_ident].pop_back();

    vector<Domain*> staple {m_origami_system.get_chain(new_c_i)};
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

vector<vector<double>> ConformationalEnumerator::normalize_weights(
        vector<vector<double>> weights) {
    auto normalized_weights {weights};
    for (size_t i {0}; i != weights.size(); i++) {
        for (size_t j {0}; j != weights[i].size(); j++) {
            normalized_weights[i][j] /= m_partition_f;
        }
    }
    return normalized_weights;
}

double ConformationalEnumerator::average_energy() {
    return m_average_energy / m_partition_f;
}

void ConformationalEnumerator::enumerate_domain(Domain* domain, VectorThree p_prev) {

    // Iterate through all positions
    for (auto p_vec: vectors) {
        VectorThree p_new = p_prev + p_vec;

        // Check if domain will be entering a bound state
        bool is_growthpoint {m_growthpoints.count(domain) > 0};
        Occupancy p_occ {m_origami_system.position_occupancy(p_new)};
        bool is_occupied {p_occ == Occupancy::unbound};
        bool is_bound {p_occ == Occupancy::bound or p_occ == Occupancy::misbound};

        // Cannot have more than two domains bound at a given position
        if ((is_growthpoint and is_occupied) or is_bound) {
            continue;
        }
        else if (is_growthpoint) {
            set_growthpoint_domains(domain, p_new);
        }
        else if (is_occupied) {
            set_bound_domain(domain, p_new);
        }
        // May be able to introduce some tricks to skip iterations over orientations
        else {
            set_unbound_domain(domain, p_new);
        }
    }
    m_domains.push_back(domain);
}

void ConformationalEnumerator::set_growthpoint_domains(
        Domain* domain,
        VectorThree p_new) {

    // Store position for terminal domains
    // Only for two domain staples
    VectorThree prev_prev_growthpoint_p;
    if (domain->m_c == 0) {
        prev_prev_growthpoint_p = m_prev_growthpoint_p;
        m_prev_growthpoint_p = p_new;
    }

    // Should always be next in stack
    Domain* bound_domain {m_domains.back()};
    m_domains.pop_back();
    bool domains_complementary {m_origami_system.check_domains_complementary(
            *domain, *bound_domain)};
    if (domains_complementary) {
        for (auto o_new: vectors) {

            m_energy += m_origami_system.set_domain_config(*domain, p_new, o_new);
            o_new = -o_new;
            m_energy += m_origami_system.set_domain_config(*bound_domain, p_new, o_new);
            if (m_origami_system.m_constraints_violated == true) {
                m_origami_system.m_constraints_violated = false;
                m_energy += m_origami_system.unassign_domain(*domain);
                continue;
            }
            m_identities_to_num_unassigned[domain->m_d_ident] -= 1;
            m_identities_to_num_unassigned[bound_domain->m_d_ident] -= 1;
            grow_next_domain(bound_domain, p_new);
            m_identities_to_num_unassigned[domain->m_d_ident] += 1;
            m_identities_to_num_unassigned[bound_domain->m_d_ident] += 1;
            m_energy += m_origami_system.unassign_domain(*domain);
        }
    }
    else {
        double pos_multiplier {36};
        VectorThree o_new {0, 0, 0};
        m_energy += m_origami_system.set_domain_config(*domain, p_new, o_new);
        m_energy += m_origami_system.set_domain_config(*bound_domain, p_new, o_new);
        m_multiplier *= pos_multiplier;
        m_identities_to_num_unassigned[domain->m_d_ident] -= 1;
        m_identities_to_num_unassigned[bound_domain->m_d_ident] -= 1;
        grow_next_domain(bound_domain, p_new);
        m_identities_to_num_unassigned[domain->m_d_ident] += 1;
        m_identities_to_num_unassigned[bound_domain->m_d_ident] += 1;
        m_energy += m_origami_system.unassign_domain(*domain);
        m_multiplier /= pos_multiplier;
    }
    m_domains.push_back(bound_domain);

    // Revert position for terminal domains
    if (domain->m_c == 0) {
        m_prev_growthpoint_p = prev_prev_growthpoint_p;
    }
}

void ConformationalEnumerator::set_bound_domain(
        Domain* domain,
        VectorThree p_new) {

    Domain* occ_domain = m_origami_system.unbound_domain_at(p_new);
    bool domains_complementary {m_origami_system.check_domains_complementary(
            *domain, *occ_domain)};
    if (domains_complementary) {
        VectorThree o_new {-occ_domain->m_ore};
        m_energy += m_origami_system.set_domain_config(*domain, p_new, o_new);
        if (m_origami_system.m_constraints_violated == true) {
            m_origami_system.m_constraints_violated = false;
            return;
        }
        else {
            double pos_multiplier {calc_multiplier(domain, occ_domain)};
            m_multiplier *= pos_multiplier;
            m_identities_to_num_unassigned[domain->m_d_ident] -= 1;
            grow_next_domain(domain, p_new);
            m_identities_to_num_unassigned[domain->m_d_ident] += 1;
            m_multiplier /= pos_multiplier;
        }
    }
    else {
        VectorThree o_new {0, 0, 0};
        m_energy += m_origami_system.set_domain_config(*domain, p_new, o_new);
        double pos_multiplier {6 * calc_multiplier(domain, occ_domain)};
        m_multiplier *= pos_multiplier;
        m_identities_to_num_unassigned[domain->m_d_ident] -= 1;
        grow_next_domain(domain, p_new);
        m_identities_to_num_unassigned[domain->m_d_ident] += 1;
        m_multiplier /= pos_multiplier;
    }
}

void ConformationalEnumerator::set_unbound_domain(
        Domain* domain,
        VectorThree p_new) {

    if (m_identities_to_num_unassigned[-domain->m_d_ident] == 0) {
        VectorThree o_new {0, 0, 0};
        m_identities_to_num_unassigned[domain->m_d_ident] -= 1;
        m_energy += m_origami_system.set_domain_config(*domain, p_new, o_new);
        double pos_multiplier {6};
        m_multiplier *= pos_multiplier;
        grow_next_domain(domain, p_new);
        m_identities_to_num_unassigned[domain->m_d_ident] += 1;
        m_multiplier /= pos_multiplier;
    }
    else {
        for (auto o_new: vectors) {
            m_energy += m_origami_system.set_domain_config(*domain, p_new, o_new);
            if (m_origami_system.m_constraints_violated) {
                m_origami_system.m_constraints_violated = false;
            }
            else {
                m_identities_to_num_unassigned[domain->m_d_ident] -= 1;
                grow_next_domain(domain, p_new);
                m_identities_to_num_unassigned[domain->m_d_ident] += 1;
            }
        }
    }
}

void ConformationalEnumerator::grow_next_domain(Domain* domain, VectorThree p_new) {

    // Continue growing next domain
    if (not m_domains.empty()) {
        Domain* next_domain {m_domains.back()};
        m_domains.pop_back();
        VectorThree new_p_prev;

        // Previous domain is last growthpoint if end-of-staple reached
        // Note this assumes that growtpoints are done together
        // Another place where only correct for two domain staples
        bool domain_is_terminal {domain->m_c != next_domain->m_c};
        if (domain_is_terminal) {
            new_p_prev = m_prev_growthpoint_p;
        }
        else {
            new_p_prev = p_new;
        }
        enumerate_domain(next_domain, new_p_prev);
    }

    // Save relevant values if system fully grown
    else {
        calc_and_save_weights();
    }
    m_energy += m_origami_system.unassign_domain(*domain);
}

void ConformationalEnumerator::create_domains_stack() {

    // Reset stack
    m_domains.clear();

    // Iterate through scaffold domains and grow out staples if growthpoint
    for (auto domain: m_origami_system.get_chain(0)) {
        bool is_growthpoint {m_growthpoints.count(domain) > 0};
        m_domains.push_back(domain);
        if (is_growthpoint) {
            Domain* new_domain {m_growthpoints[domain]};
            create_staple_stack(new_domain);
        }
    }

    // Growing domains at the end of the vector first
    std::reverse(m_domains.begin(), m_domains.end());
}

void ConformationalEnumerator::create_staple_stack(Domain* domain) {

    // Add growth domain
    m_domains.push_back(domain);

    vector<Domain*> staple {m_origami_system.get_chain(domain->m_c)};

    // Iterate through domains in three prime direction
    for (size_t d_i = domain->m_d + 1; d_i != staple.size(); d_i++) {
        Domain* domain {staple[d_i]};
        bool is_growthpoint {m_growthpoints.count(domain) > 0};
        m_domains.push_back(domain);
        if (is_growthpoint) {
            Domain* new_domain {m_growthpoints[domain]};
            create_staple_stack(new_domain);
        }
    }

    // Add domains in five prime direction
    for (int d_i {domain->m_d - 1}; d_i != -1; d_i--) {
        Domain* domain {staple[d_i]};
        bool is_growthpoint {m_growthpoints.count(domain) > 0};
        m_domains.push_back(domain);
        if (is_growthpoint) {
            Domain* new_domain {m_growthpoints[domain]};
            create_staple_stack(new_domain);
        }
    }

    return;
}

double ConformationalEnumerator::calc_multiplier(Domain* domain, Domain* other_domain) {
    // This only works for staples that are only 2 domains

    double multiplier {1};
    //  No overcounting if binding to self
    if (domain->m_c == other_domain->m_c) {
        multiplier = 1;
    }
    else {
        int involved_staples {0};
        involved_staples += count_involved_staples(domain);
        involved_staples += count_involved_staples(other_domain);
        multiplier /= (involved_staples + 1);
    }
    return multiplier;
}

int ConformationalEnumerator::count_involved_staples(Domain* domain) {
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
        next_domain = next_domain->m_bound_domain;
        if (next_domain->m_c == 0) {
            domain_on_scaffold = true;
        }
        else {
            involved_staples++;
        }
    }
    return involved_staples;
}

void ConformationalEnumerator::calc_and_save_weights() {
    m_num_configs += m_multiplier;
    double weight {m_prefix * exp(-m_energy) * m_multiplier};
    m_average_energy += m_energy * weight;
    m_partition_f += weight;
    m_bound_state_weights[m_origami_system.num_staples()][
            m_origami_system.num_fully_bound_domain_pairs()] += weight;
    m_misbound_state_weights[m_origami_system.num_staples()][
            m_origami_system.num_misbound_domain_pairs()] += weight;
}
