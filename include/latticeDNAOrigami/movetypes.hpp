#ifndef MOVETYPES_H
#define MOVETYPES_H

#include <deque>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "bias_functions.hpp"
#include "domain.hpp"
#include "files.hpp"
#include "ideal_random_walk.hpp"
#include "order_params.hpp"
#include "origami_system.hpp"
#include "parser.hpp"
#include "random_gens.hpp"
#include "top_constraint_points.hpp"
#include "utility.hpp"

namespace movetypes {

using std::cout;
using std::deque;
using std::ostream;
using std::pair;
using std::set;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

using biasFunctions::SystemBiases;
using domain::Domain;
using files::OrigamiOutputFile;
using idealRandomWalk::IdealRandomWalks;
using orderParams::SystemOrderParams;
using origami::OrigamiSystem;
using parser::InputParameters;
using randomGen::RandomGens;
using topConstraintPoints::Constraintpoints;
using utility::VectorThree;

typedef pair<Domain*, Domain*> domainPairT;

/** Tracker for general movetype info */
struct MovetypeTracking {
    int attempts;
    int accepts;
};

/** Parent class of all movetypes
 *
 * It serves both to define the interface and provides some shared
 * implementation details, including shared data.
 */
class MCMovetype {
  public:
    MCMovetype(
            OrigamiSystem& origami_system,
            RandomGens& random_gens,
            IdealRandomWalks& ideal_random_walks,
            vector<std::unique_ptr<OrigamiOutputFile>>& config_files,
            string& label,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);
    MCMovetype(const MCMovetype&) = delete;
    MCMovetype& operator=(const MCMovetype&) = delete;
    virtual ~MCMovetype() {};
    bool attempt_move(unsigned long long step);

    /** Reset origami system to state before move attempt */
    virtual void reset_origami();
    void write_log_summary_header(ostream& log_stream);
    virtual void write_log_summary(ostream& log_stream) = 0;

    string get_label();
    int get_attempts();
    int get_accepts();

  protected:
    virtual void add_external_bias() = 0;
    virtual bool internal_attempt_move() = 0;
    virtual void add_tracker(bool accepted) = 0;
    virtual void reset_internal();

    // Shared general utility functions

    /** Return a random domain in the system
     *
     * Uniform over the set of all domains, rather than uniform over
     * chains and then over the constituent domains
     */
    Domain& select_random_domain();
    int select_random_staple_identity();
    int select_random_staple_of_identity(int c_i_ident);
    VectorThree select_random_position(VectorThree p_prev);
    VectorThree select_random_orientation();
    bool test_acceptance(double p_ratio);

    /** Test if staple is anchoring other staples to the system
     *
     * Connecting staples, or anchor staples, are staples that if
     * removed would leave the system in a state with staples
     * unconnected to the scaffold
     */
    bool staple_is_connector(const vector<Domain>& staple);

    /** Find all staples in bound network to given domains */
    set<int> find_staples(const vector<Domain*>& domains);

    /** Write the config to move file
     *
     * The move file is for recording the configuration as the chains
     * are grown.
     */
    void write_config();

    OrigamiSystem& m_origami_system;
    RandomGens& m_random_gens;
    IdealRandomWalks& m_ideal_random_walks;
    vector<std::unique_ptr<OrigamiOutputFile>>& m_config_files;
    string m_label;
    SystemOrderParams& m_ops;
    SystemBiases& m_biases;
    InputParameters& m_params;
    bool m_rejected {false};
    size_t m_config_output_freq;

    // Staple maxes
    size_t m_max_total_staples;
    size_t m_max_type_staples;

    // Lists of modified domains for move reversal
    vector<pair<int, int>> m_modified_domains {};
    vector<pair<int, int>> m_assigned_domains {};
    vector<int> m_added_chains {};

    // Domains mapped to previous configs
    unordered_map<pair<int, int>, VectorThree> m_prev_pos {};
    unordered_map<pair<int, int>, VectorThree> m_prev_ore {};

    // Modifier to correct for excluded volume and overcounting
    double m_modifier {1};

    MovetypeTracking m_general_tracker {0, 0};
    unsigned long long m_step {0};

    // Check if scaffold domain bound to network of given staple
    bool scan_for_scaffold_domain(
            const Domain& domain,
            set<int>& participating_chains);

    /** Find all domains bound directly to give domains */
    vector<domainPairT> find_bound_domains(
            vector<Domain>& selected_chain);
};

class IdentityMCMovetype: public MCMovetype {
  public:
    IdentityMCMovetype(
            OrigamiSystem& origami_system,
            RandomGens& random_gens,
            IdealRandomWalks& ideal_random_walks,
            vector<std::unique_ptr<OrigamiOutputFile>>& config_files,
            string& label,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);
    IdentityMCMovetype(const IdentityMCMovetype&) = delete;
    IdentityMCMovetype& operator=(const IdentityMCMovetype&) = delete;
    bool attempt_move(unsigned long long) { return true; };
};

/** Parent class of moves with chain regrowth */
class RegrowthMCMovetype: virtual public MCMovetype {

  public:
    RegrowthMCMovetype(
            OrigamiSystem& origami_system,
            RandomGens& random_gens,
            IdealRandomWalks& ideal_random_walks,
            vector<std::unique_ptr<OrigamiOutputFile>>& config_files,
            string& label,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);
    RegrowthMCMovetype(const RegrowthMCMovetype&) = delete;
    RegrowthMCMovetype& operator=(const RegrowthMCMovetype&) = delete;

  protected:
    void reset_internal() override;

    /** Grow given contiguous domains from a chain */
    virtual void grow_chain(vector<Domain*> domains) = 0;

    /** Set domain to have complementary orientation to given */
    double set_growth_point(
            Domain& growth_domain_new,
            Domain& growth_domain_old);

    /** Grow staple out from given index in both directions
     *
     * The indices passed to grow chain should include the growth
     * point.
     */
    void grow_staple(int d_i_index, vector<Domain>& selected_chain);

    pair<Domain*, Domain*> select_new_growthpoint(
            vector<Domain>& selected_chain);

    /** Select a growthpoint from set of possible */
    domainPairT select_old_growthpoint(vector<domainPairT> bound_domains);

    /** Number of staple domains (mis)bound to external chains **/
    size_t num_bound_staple_domains(const vector<Domain>& staple);

    // Store old positions and orientations
    unordered_map<pair<int, int>, VectorThree> m_old_pos {};
    unordered_map<pair<int, int>, VectorThree> m_old_ore {};
};

/** Parent class of moves with constant topology regrowth */
class CTRegrowthMCMovetype: virtual public RegrowthMCMovetype {

  public:
    CTRegrowthMCMovetype(
            OrigamiSystem& origami_system,
            RandomGens& random_gens,
            IdealRandomWalks& ideal_random_walks,
            vector<std::unique_ptr<OrigamiOutputFile>>& config_files,
            string& label,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params,
            size_t num_excluded_staples,
            size_t max_regrowth,
            size_t max_seg_regrowth);
    CTRegrowthMCMovetype(const CTRegrowthMCMovetype&) = delete;
    CTRegrowthMCMovetype& operator=(const CTRegrowthMCMovetype&) = delete;

  protected:
    void reset_internal() override;

    void sel_excluded_staples();

    /** Select scaffold segment to be regrown */
    vector<Domain*> select_indices(
            vector<Domain*> segment,
            size_t min_length,
            int seg = 0);
    vector<Domain*> select_indices(
            vector<Domain>& segment,
            size_t min_length,
            int seg = 0);

    /** Select noncontiguous scaffold segments
     *
     * Selection begins by selectig a random domain in the given
     * segment, a random direction, a random maximum segment
     * length and a random maximum number of domains to regrow length.
     * Domains are added to this first segment until the maxium
     * segment length is reached, the maximum number of domains to
     * regrow is reached. If an added domain is bound to a staple
     * that is bound to another scaffold domain is reached, and if the
     * other scaffold domain is not among those already selected for
     * regrowth and not contiguous to a domain selected for regrowth,
     * then two new segments will be created. Creation of a new
     * segment requires a maximum segment length to be selected. If
     * this is non-zero, the first domain is added to the segment. The
     * order of regrowth direction is selected randomly.
     */
    void select_noncontig_segs(
            vector<Domain>& given_seg,
            vector<vector<Domain*>>& segs,
            vector<vector<vector<Domain*>>>& paired_segs,
            vector<Domain*>& seg_stems,
            vector<int>& dirs);

    /** Check if excluded staples are still bound to system */
    bool excluded_staples_bound();

    int m_dir {};
    size_t m_num_excluded_staples;
    vector<int> m_excluded_staples;
    Constraintpoints m_constraintpoints {
            m_origami_system,
            m_ideal_random_walks}; // For fixed end biases
    vector<Domain>& m_scaffold;

    // Maximum number of scaffold domains to regrow
    size_t m_max_regrowth;

  private:
    bool fill_seg(
            Domain* start_d,
            size_t max_length,
            size_t seg_max_length,
            int dir,
            set<Domain*>& domains,
            deque<Domain*>& possible_stems,
            vector<Domain*>& seg);
    size_t m_max_seg_regrowth;
};

/**
 * Template function for updating movetype specific trackers
 */
template <typename T>
void add_tracker(
        T tracker,
        unordered_map<T, MovetypeTracking>& trackers,
        bool accepted) {
    if (trackers.find(tracker) == trackers.end()) {
        trackers[tracker] = {1, accepted};
    }
    else {
        trackers[tracker].attempts++;
        trackers[tracker].accepts += accepted;
    }
}
} // namespace movetypes

#endif // MOVETYPES_H
