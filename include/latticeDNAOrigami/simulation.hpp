#ifndef SIMULATION_H
#define SIMULATION_H

#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <boost/process.hpp>

#include "bias_functions.hpp"
#include "files.hpp"
#include "ideal_random_walk.hpp"
#include "movetypes.hpp"
#include "order_params.hpp"
#include "origami_system.hpp"
#include "parser.hpp"
#include "random_gens.hpp"

namespace simulation {

using std::map;
using std::ofstream;
using std::ostream;
using std::set;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;
using std::chrono::steady_clock;

namespace bp = boost::process;

using biasFunctions::SystemBiases;
using files::OrigamiMovetypeFile;
using files::OrigamiOrientationOutputFile;
using files::OrigamiOutputFile;
using files::OrigamiStateOutputFile;
using files::OrigamiVCFOutputFile;
using files::OrigamiVSFOutputFile;
using idealRandomWalk::IdealRandomWalks;
using movetypes::MCMovetype;
using orderParams::SystemOrderParams;
using origami::OrigamiSystem;
using parser::InputParameters;
using randomGen::RandomGens;

vector<std::unique_ptr<OrigamiOutputFile>> setup_output_files(
        InputParameters& params,
        const string& output_filebase,
        OrigamiSystem& origami,
        SystemOrderParams& ops,
        SystemBiases& biases);

void setup_config_files(
        const string& filebase,
        const size_t max_total_staples,
        const size_t freq,
        OrigamiSystem& origami,
        vector<std::unique_ptr<OrigamiOutputFile>>& files);

class GCMCSimulation {
  public:
    GCMCSimulation(
            OrigamiSystem& origami_system,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);
    GCMCSimulation(
            OrigamiSystem& origami_system,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params,
            ostream& logging_stream);
    virtual ~GCMCSimulation() {}
    virtual void run() = 0;

  protected:
    void initialize(InputParameters params);
    virtual void update_internal(unsigned long long step) = 0;

    void construct_movetypes(InputParameters& params);
    std::unique_ptr<MCMovetype> setup_orientation_movetype(
            size_t i,
            const string& type,
            string& label,
            OrigamiMovetypeFile& movetypes_file);
    std::unique_ptr<MCMovetype> setup_staple_exchange_movetype(
            size_t i,
            const string& type,
            string& label,
            OrigamiMovetypeFile& movetypes_file);
    std::unique_ptr<MCMovetype> setup_staple_regrowth_movetype(
            size_t i,
            const string& type,
            string& label,
            OrigamiMovetypeFile& movetypes_file);
    std::unique_ptr<MCMovetype> setup_scaffold_regrowth_movetype(
            size_t i,
            const string& type,
            string& label,
            OrigamiMovetypeFile& movetypes_file);
    std::unique_ptr<MCMovetype> setup_rg_movetype(
            size_t i,
            const string& type,
            string& label,
            OrigamiMovetypeFile& movetypes_file);
    void set_max_dur(unsigned long long dur);
    unsigned long long simulate(
            unsigned long long steps,
            unsigned long long start_step = 0,
            bool summarize = true,
            steady_clock::time_point = steady_clock::now());
    MCMovetype& select_movetype();
    void write_log_entry(
            const unsigned long long step,
            bool accepted,
            MCMovetype& movetype);
    void write_log_summary();
    void setup_vmd_pipe();
    void pipe_to_vmd();

    OrigamiSystem& m_origami_system;
    SystemOrderParams& m_ops;
    SystemBiases& m_biases;

    ostream& m_logging_stream;
    ofstream m_logging_file {};
    size_t m_logging_freq {};
    size_t m_centering_freq {};
    size_t m_centering_domain {};
    size_t m_constraint_check_freq {};
    size_t m_vmd_pipe_freq {};
    double m_max_duration {};
    InputParameters& m_params;
    vector<std::unique_ptr<OrigamiOutputFile>> m_output_files {};
    vector<std::unique_ptr<OrigamiOutputFile>> m_config_per_move_files {};
    vector<unique_ptr<MCMovetype>> m_movetypes {};
    vector<double> m_movetype_freqs {};
    vector<double> m_cumulative_probs {};
    RandomGens m_random_gens {};
    IdealRandomWalks m_ideal_random_walks {};

    // VMD realtime visualization
    std::unique_ptr<OrigamiVSFOutputFile> m_vmd_struct_file {};
    std::unique_ptr<OrigamiVCFOutputFile> m_vmd_coors_file {};
    std::unique_ptr<OrigamiStateOutputFile> m_vmd_states_file {};
    std::unique_ptr<OrigamiOrientationOutputFile> m_vmd_ores_file {};
    std::unique_ptr<bp::child> vmd_proc {};
};
} // namespace simulation

#endif // SIMULATION_H
