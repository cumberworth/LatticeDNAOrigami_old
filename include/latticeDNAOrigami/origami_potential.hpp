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

using domainContainer::Domain;
using nearestNeighbour::ThermoOfHybrid;
using parser::InputParameters;
using utility::VectorThree;

// Shared constraint checkers
bool check_domain_orientations_opposing(Domain& cd_i, Domain& cd_j);
bool check_doubly_contig(Domain* cd_1, Domain* cd_2);
bool check_domains_exist_and_bound(vector<Domain*> cdv);
bool doubly_contiguous_helix(Domain* cd_1, Domain* cd_2);

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
    virtual ~BindingPotential() {}

    virtual DeltaConfig bind_domains(Domain& cd_i, Domain& cd_j) = 0;
    virtual DeltaConfig check_stacking(Domain& cd_i, Domain& cd_j);
    bool m_constraints_violated;

  protected:
    OrigamiPotential& m_pot;
    DeltaConfig m_delta_config;

    virtual void internal_check_stacking(Domain& cd_i, Domain& cd_j);
    virtual void check_pair_stacking(
            Domain* cd_1,
            Domain* cd_2,
            bool doubly_contig) = 0;
};

class RestrictiveBindingPotential: public BindingPotential {

  public:
    using BindingPotential::BindingPotential;
    DeltaConfig bind_domains(Domain& cd_i, Domain& cd_j) override;
    void check_pair_stacking(Domain* cd_1, Domain* cd_2, bool doubly_contig)
            override;

  private:
    // Top level interaction checks
    bool check_domain_pair_constraints(Domain& cd_i);
    bool check_helical_constraints(Domain& cd_1, Domain& cd_2);

    // Linear helix checks
    bool check_linear_helix(VectorThree ndr_1, Domain& cd_2);
    bool check_linear_helix_rear(Domain& cd_3);

    // Junction checks
    bool check_doubly_contiguous_junction(Domain& cd_2, Domain& cd_3);
    bool doubly_contiguous_junction(Domain& cd_1, Domain& cd_2);
    bool check_doubly_contiguous_junction(
            Domain& cd_1,
            Domain& cd_2,
            Domain& cd_3,
            Domain& cd_4);
    bool check_junction_front(Domain& cd_1);
    bool check_junction_rear(Domain& cd_4);
};

/** Base class for binding potentials that allow bending at backbone nicks
 *
 */
class FlexibleBindingPotential: public BindingPotential {

  public:
    using BindingPotential::BindingPotential;
    DeltaConfig bind_domains(Domain& cd_i, Domain& cd_j) override;

  protected:
    virtual void check_constraints(Domain* cd) = 0;
    virtual void check_doubly_contig_junction(
            Domain* cd_1,
            Domain* cd_2,
            Domain* cd_3,
            Domain* cd_4) = 0;
    virtual bool check_regular_pair_constraints(
            Domain* cd_1,
            Domain* cd_2,
            int i);
    bool check_possible_doubly_contig_helix(Domain* cd_1, Domain* cd_2);
    void check_possible_doubly_contig_junction(Domain* cd_1, Domain* cd_2);
    bool check_pair_stacked(Domain* cd_1, Domain* cd_2);
    void check_edge_pair_junction(Domain* cd_1, Domain* cd_2, int i);
};

/** Helices must be linear to be stacked */
class LinearFlexibleBindingPotential: public FlexibleBindingPotential {

  public:
    using FlexibleBindingPotential::FlexibleBindingPotential;
    DeltaConfig bind_domains(Domain& cd_i, Domain& cd_j) override;
    DeltaConfig check_stacking(Domain& cd_i, Domain& cd_j) override;
    void check_pair_stacking(Domain* cd_1, Domain* cd_2, bool doubly_contig)
            override;

  private:
    void check_constraints(Domain* cd) override;
    void check_doubly_contig_junction(
            Domain* cd_1,
            Domain* cd_2,
            Domain* cd_3,
            Domain* cd_4) override;
    void check_linear_helix(Domain* cd_1, Domain* cd_2, int i);
    void check_linear_helix(Domain* cd_h1, Domain* cd_h2, Domain* cd_h3);
    void check_central_linear_helix(Domain& cd_i, Domain& cd_j);

  protected:
    // The following are ugly methods to enforce the ugly four body
    // stacking rule that would be much easier to enforce if the
    // domains had explicit helical axis vectors

    /** Check for unstacked single junctions from first two domains
     *
     * The domains passed are the first two domains of the junction.
     * A total of nine combinations of domains will be tested,
     * including two different kink pairs
     */
    void check_forward_single_junction(Domain* cd_1, Domain* cd_2);

    /** Check for unstacked single junctions from last two domains
     *
     * The domains passed are the last two domains of the junction.
     * A total of nine combinations of domains will be tested,
     * including two different kink pairs
     */
    void check_backward_single_junction(Domain* cd_1, Domain* cd_2);

    /** Check for unstacked single junctions from kink pair
     *
     * The domains passed are the kink pair. A total of nine
     * combinations of domains will be tested.
     */
    void check_central_single_junction(Domain* cd_1, Domain* cd_2);

    /** Check for unstacked single junctions
     *
     * This will subtract a stacked pair from configurations in which
     * the next domain vectors of the first and third pairs are
     * parallel to each other, antiparallel to the kink pair next
     * domain vector, and the kink pair is not agreeing with where a
     * crossover should occur given the domain lengths
     */
    void check_single_junction(
            Domain* cd_j1,
            Domain* cd_j2,
            Domain* cd_j3,
            Domain* cd_j4,
            Domain* cd_k1,
            Domain* cd_k2);
};

/** Only some configurations allowed at kinks */
class ConKinkLinearFlexibleBindingPotential:
        public LinearFlexibleBindingPotential {

  public:
    using LinearFlexibleBindingPotential::LinearFlexibleBindingPotential;

    bool check_regular_pair_constraints(Domain* cd_1, Domain* cd_2, int i);
};

/** Helices can be fully stacked and nonlinear */
class NonLinearFlexibleBindingPotential: public FlexibleBindingPotential {

  public:
    using FlexibleBindingPotential::FlexibleBindingPotential;
    void check_pair_stacking(Domain* cd_1, Domain* cd_2, bool doubly_contig)
            override;

  private:
    void check_constraints(Domain* cd) override;
    void check_doubly_contig_junction(
            Domain* cd_1,
            Domain* cd_2,
            Domain* cd_3,
            Domain* cd_4) override;
};

class MisbindingPotential {
  public:
    MisbindingPotential(OrigamiPotential& pot);
    virtual ~MisbindingPotential() {}

    virtual double bind_domains(Domain& cd_i, Domain& cd_j) = 0;
    bool m_constraints_violated;

  protected:
    OrigamiPotential& m_pot;
};

/**
 * Misbound domanis must have opposing orientation vectors
 */
class OpposingMisbindingPotential: public MisbindingPotential {

  public:
    using MisbindingPotential::MisbindingPotential;
    double bind_domains(Domain& cd_i, Domain& cd_j) override;
};

/**
 * Misbound domains must have opposing orientation vectors
 */
class DisallowedMisbindingPotential: public MisbindingPotential {

  public:
    using MisbindingPotential::MisbindingPotential;
    double bind_domains(Domain&, Domain&) override;
};

class OrigamiPotential {
    // Interface to origami potential
  public:
    OrigamiPotential(
            const vector<vector<int>> m_identities,
            const vector<vector<string>>& sequences,
            InputParameters& params);
    ~OrigamiPotential();

    bool m_constraints_violated;

    void update_temp(double temp, double stacking_mult = 1);

    // Domain interactions
    DeltaConfig bind_domain(Domain& cd_i);
    bool check_domains_complementary(Domain& cd_i, Domain& cd_j);
    DeltaConfig check_stacking(Domain& cd_i, Domain& cd_j);

    // Energy calculations
    double hybridization_energy(const Domain& cd_i, const Domain& cd_j) const;
    double hybridization_enthalpy(const Domain& cd_i, const Domain& cd_j) const;
    double hybridization_entropy(const Domain& cd_i, const Domain& cd_j) const;
    double stacking_energy(const Domain& cd_i, const Domain& cd_j) const;

  private:
    string m_energy_filebase;
    double m_temp;
    const double m_cation_M; // Cation concentration (mol/L)
    const vector<vector<int>> m_identities; // Domain identities
    const vector<vector<string>> m_sequences; // Domain sequences

    // Containers for binding rules
    BindingPotential* m_binding_pot;
    MisbindingPotential* m_misbinding_pot;
    string m_stacking_pot;
    string m_hybridization_pot;

    // Stacking energy if constant
    double m_stacking_ene {0};

    // Hybridization enthalpy and entropy if constant
    double m_binding_h;
    double m_binding_s;
    double m_misbinding_h;
    double m_misbinding_s;

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
            string seq_i,
            string seq_j,
            pair<int, int> key);
    void calc_hybridization_energy(pair<int, int> key);
    void calc_stacking_energy(string seq_i, string seq_j, pair<int, int> key);
    void calc_stacking_energy(pair<int, int> key);
};

} // namespace potential

#endif // ORIGAMI_POTENTIAL_H