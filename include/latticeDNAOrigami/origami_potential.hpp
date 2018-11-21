#ifndef ORIGAMI_POTENTIAL_H
#define ORIGAMI_POTENTIAL_H

#include <vector>

#include "domain.hpp"
#include "hash.hpp"
#include "nearest_neighbour.hpp"
#include "parser.hpp"
#include "utility.hpp"

namespace potential {

using std::pair;
using std::string;
using std::unordered_map;
using std::vector;

using domain::Domain;
using nearestNeighbour::ThermoOfHybrid;
using parser::InputParameters;
using utility::VectorThree;

// Shared constraint checkers
bool check_domain_orientations_opposing(const Domain& cd_i, const Domain& cd_j);
bool check_doubly_contig(const Domain& cd_1, const Domain& cd_2);

template <typename T>
bool check_domains_exist_and_bound(const T& domain) {
    if (domain == nullptr) {
        return false;
    }
    return domain->m_state == utility::Occupancy::bound;
}

template <typename T, typename... Args>
bool check_domains_exist_and_bound(const T& domain, Args... args) {
    return check_domains_exist_and_bound(domain) and
           check_domains_exist_and_bound(args...);
}

bool doubly_contiguous_helix(const Domain& cd_1, const Domain& cd_2);

struct DeltaConfig {
    double e {0};
    int stacked_pairs {0};
    int linear_helices {0};
    int stacked_juncts {0};
};

// Forward declaration
class OrigamiPotential;

class BindingPotential {
  public:
    BindingPotential(OrigamiPotential& pot);
    DeltaConfig bind_domains(const Domain& cd_i, const Domain& cd_j);
    DeltaConfig check_stacking(const Domain& cd_i, const Domain& cd_j);

    bool m_constraints_violated {};

  private:
    OrigamiPotential& m_pot;
    DeltaConfig m_delta_config;

    void check_constraints(const Domain& cd);
    bool check_possible_doubly_contig_helix(
            const Domain& cd_1,
            const Domain& cd_2);

    /** Check for unstacked single junctions from first two domains
     *
     * The domains passed are the first two domains of the junction.
     * A total of nine combinations of domains will be tested,
     * including two different kink pairs
     */
    void check_forward_single_junction(const Domain& cd_1, const Domain& cd_2);
    bool check_pair_stacked(const Domain& cd_1, const Domain& cd_2);

    /** Check for unstacked single junctions
     *
     * This will subtract a stacked pair from configurations in which
     * the next domain vectors of the first and third pairs are
     * parallel to each other, antiparallel to the kink pair next
     * domain vector, and the kink pair is not agreeing with where a
     * crossover should occur given the domain lengths
     */
    void check_single_junction(
            const Domain* cd_j1,
            const Domain* cd_j2,
            const Domain* cd_j3,
            const Domain* cd_j4,
            const Domain* cd_k1,
            const Domain* cd_k2);

    /** Check for unstacked single junctions from last two domains
     *
     * The domains passed are the last two domains of the junction.
     * A total of nine combinations of domains will be tested,
     * including two different kink pairs
     */
    void check_backward_single_junction(const Domain& cd_1, const Domain& cd_2);
    void check_possible_doubly_contig_junction(
            const Domain& cd_1,
            const Domain& cd_2);
    void check_doubly_contig_junction(
            const Domain& cd_1,
            const Domain& cd_2,
            const Domain& cd_3,
            const Domain& cd_4);
    bool check_regular_pair_constraints(
            const Domain& cd_1,
            const Domain& cd_2,
            const int i);
    void check_edge_pair_junction(
            const Domain& cd_1,
            const Domain& cd_2,
            const int i);
    void check_central_linear_helix(const Domain& cd_i, const Domain& cd_j);
    void check_linear_helix(
            const Domain* cd_1,
            const Domain* cd_2,
            const int i);
    void check_linear_helix(
            const Domain* cd_h1,
            const Domain* cd_h2,
            const Domain* cd_h3);

    /** Check for unstacked single junctions from kink pair
     *
     * The domains passed are the kink pair. A total of nine
     * combinations of domains will be tested.
     */
    void check_central_single_junction(const Domain& cd_1, const Domain& cd_2);
};

class MisbindingPotential {
  public:
    MisbindingPotential(OrigamiPotential& pot);

    double bind_domains(const Domain& cd_i, const Domain& cd_j);
    bool m_constraints_violated {};

  private:
    OrigamiPotential& m_pot;
};

class OrigamiPotential {
    // Interface to origami potential
  public:
    OrigamiPotential(
            vector<vector<int>>& m_identities,
            vector<vector<string>>& sequences,
            InputParameters& params);

    bool m_constraints_violated {};

    void update_temp(double temp, double stacking_mult = 1);

    // Domain interactions
    DeltaConfig bind_domain(Domain& cd_i);
    bool check_domains_complementary(const Domain& cd_i, const Domain& cd_j);
    DeltaConfig check_stacking(Domain& cd_i, Domain& cd_j);

    // Energy calculations
    double hybridization_energy(const Domain& cd_i, const Domain& cd_j) const;
    double hybridization_enthalpy(const Domain& cd_i, const Domain& cd_j) const;
    double hybridization_entropy(const Domain& cd_i, const Domain& cd_j) const;
    double stacking_energy(const Domain& cd_i, const Domain& cd_j) const;

  private:
    string m_energy_filebase {};
    double m_temp {};
    const double m_cation_M {}; // Cation concentration (mol/L)
    const vector<vector<int>> m_identities {}; // Domain identities
    const vector<vector<string>> m_sequences {}; // Domain sequences

    // Containers for binding rules
    BindingPotential m_binding_pot;
    MisbindingPotential m_misbinding_pot;
    string m_stacking_pot {};
    string m_hybridization_pot {};

    // Stacking energy if constant
    double m_stacking_ene {0};

    // Hybridization enthalpy and entropy if constant
    double m_binding_h {};
    double m_binding_s {};
    double m_misbinding_h {};
    double m_misbinding_s {};

    // CONSIDER DEFINING TYPE FOR THESE TABLES
    // Energy tables index by chain/domain identity pair
    unordered_map<pair<int, int>, double> m_hybridization_energies {};
    unordered_map<pair<int, int>, double> m_hybridization_enthalpies {};
    unordered_map<pair<int, int>, double> m_hybridization_entropies {};
    unordered_map<pair<int, int>, double> m_stacking_energies {};

    // Energies tables indexed by temperature
    unordered_map<double, unordered_map<pair<int, int>, double>>
            m_hybridization_energy_tables {};
    unordered_map<double, unordered_map<pair<int, int>, double>>
            m_hybridization_enthalpy_tables {};
    unordered_map<double, unordered_map<pair<int, int>, double>>
            m_hybridization_entropy_tables {};
    unordered_map<pair<double, double>, unordered_map<pair<int, int>, double>>
            m_stacking_energy_tables {};

    // Energy table preperation
    void get_energies();
    bool read_energies_from_file();
    void write_energies_to_file();
    void calc_energies();
    void calc_hybridization_energy(
            const string& seq_i,
            const string& seq_j,
            pair<int, int> key);
    void calc_hybridization_energy(pair<int, int> key);
    void calc_stacking_energy(
            const string& seq_i,
            const string& seq_j,
            pair<int, int> key);
    void calc_stacking_energy(pair<int, int> key);
};

} // namespace potential

#endif // ORIGAMI_POTENTIAL_H
