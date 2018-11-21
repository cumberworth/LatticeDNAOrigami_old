#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <utility>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/process.hpp>

#include "cb_movetypes.hpp"
#include "files.hpp"
#include "met_movetypes.hpp"
#include "movetypes.hpp"
#include "orientation_movetype.hpp"
#include "random_gens.hpp"
#include "rg_movetypes.hpp"
#include "simulation.hpp"
#include "utility.hpp"

namespace simulation {

using std::cout;
using std::chrono::steady_clock;

namespace bp = boost::process;

using files::OrigamiCountsOutputFile;
using files::OrigamiEnergiesOutputFile;
using files::OrigamiMovetypeFile;
using files::OrigamiOrderParamsOutputFile;
using files::OrigamiStaplesBoundOutputFile;
using files::OrigamiStaplesFullyBoundOutputFile;
using files::OrigamiTimesOutputFile;
using files::OrigamiTrajOutputFile;

vector<std::unique_ptr<OrigamiOutputFile>> setup_output_files(
        InputParameters& params,
        const string& output_filebase,
        OrigamiSystem& origami,
        SystemOrderParams& ops,
        SystemBiases& biases) {

    // Hack to get a vsf file
    OrigamiVSFOutputFile vsf_file {
            output_filebase + ".vsf", 0, params.m_max_total_staples, origami};
    vsf_file.write(0, 0);

    vector<std::unique_ptr<OrigamiOutputFile>> outs {};
    if (params.m_configs_output_freq != 0) {
        auto config_out {std::make_unique<OrigamiTrajOutputFile>(
                output_filebase + ".trj",
                params.m_configs_output_freq,
                params.m_max_total_staples,
                origami)};
        outs.push_back(std::move(config_out));
    }
    if (params.m_vtf_output_freq != 0) {
        setup_config_files(
                output_filebase,
                params.m_max_total_staples,
                params.m_vtf_output_freq,
                origami,
                outs);
    }
    if (params.m_counts_output_freq != 0) {
        auto counts_out {std::make_unique<OrigamiCountsOutputFile>(
                output_filebase + ".counts",
                params.m_counts_output_freq,
                params.m_max_total_staples,
                origami)};
        outs.push_back(std::move(counts_out));
        auto staples_bound {std::make_unique<OrigamiStaplesBoundOutputFile>(
                output_filebase + ".staples",
                params.m_counts_output_freq,
                params.m_max_total_staples,
                origami)};
        outs.push_back(std::move(staples_bound));
        auto staples_fully_bound {
                std::make_unique<OrigamiStaplesFullyBoundOutputFile>(
                        output_filebase + ".staplestates",
                        params.m_counts_output_freq,
                        params.m_max_total_staples,
                        origami)};
        outs.push_back(std::move(staples_fully_bound));
    }
    if (params.m_times_output_freq != 0) {
        auto times_out {std::make_unique<OrigamiTimesOutputFile>(
                output_filebase + ".times",
                params.m_times_output_freq,
                params.m_max_total_staples,
                origami)};
        outs.push_back(std::move(times_out));
    }
    if (params.m_energies_output_freq != 0) {
        auto energies_out {std::make_unique<OrigamiEnergiesOutputFile>(
                output_filebase + ".ene",
                params.m_energies_output_freq,
                params.m_max_total_staples,
                origami,
                biases)};
        outs.push_back(std::move(energies_out));
    }
    if (params.m_order_params_output_freq != 0) {
        auto order_params_out {std::make_unique<OrigamiOrderParamsOutputFile>(
                output_filebase + ".ops",
                params.m_order_params_output_freq,
                params.m_max_total_staples,
                origami,
                ops,
                params.m_ops_to_output)};
        outs.push_back(std::move(order_params_out));
    }

    return outs;
}

void setup_config_files(
        const string& filebase,
        const size_t max_total_staples,
        const size_t freq,
        OrigamiSystem& origami,
        vector<std::unique_ptr<OrigamiOutputFile>>& files) {

    auto vcf_out {std::make_unique<OrigamiVCFOutputFile>(
            filebase + ".vcf", freq, max_total_staples, origami)};
    files.push_back(std::move(vcf_out));

    auto states_out {std::make_unique<OrigamiStateOutputFile>(
            filebase + ".states", freq, max_total_staples, origami)};
    files.push_back(std::move(states_out));

    auto ores_out {std::make_unique<OrigamiOrientationOutputFile>(
            filebase + ".ores", freq, max_total_staples, origami)};
    files.push_back(std::move(ores_out));
}

GCMCSimulation::GCMCSimulation(
        OrigamiSystem& origami_system,
        SystemOrderParams& ops,
        SystemBiases& biases,
        InputParameters& params,
        ostream& logging_stream):
        m_origami_system {origami_system},
        m_ops {ops},
        m_biases {biases},
        m_logging_stream {logging_stream},
        m_params {params} {

    initialize(params);
}

GCMCSimulation::GCMCSimulation(
        OrigamiSystem& origami_system,
        SystemOrderParams& ops,
        SystemBiases& biases,
        InputParameters& params):
        m_origami_system {origami_system},
        m_ops {ops},
        m_biases {biases},
        m_logging_stream {m_logging_file},
        m_params {params} {

    initialize(params);
}

void GCMCSimulation::initialize(InputParameters params) {
    m_logging_freq = params.m_logging_freq;
    m_centering_freq = params.m_centering_freq;
    m_centering_domain = params.m_centering_domain;
    m_constraint_check_freq = params.m_constraint_check_freq;
    m_vmd_pipe_freq = params.m_vmd_pipe_freq;
    m_max_duration = params.m_max_duration;

    if (m_params.m_random_seed != 0) {
        m_random_gens.set_seed(m_params.m_random_seed);
    }

    if (m_vmd_pipe_freq != 0) {
        setup_vmd_pipe();
    }

    // HACK (files won't be right if filebase changed in derived
    // constructor)
    if (params.m_vcf_per_domain) {
        setup_config_files(
                params.m_output_filebase + "_move",
                params.m_max_total_staples,
                params.m_configs_output_freq,
                m_origami_system,
                m_config_per_move_files);
    }

    // Constructor movetypes
    construct_movetypes(params);

    // Create cumulative probability array
    double cum_prob {0};
    for (size_t i {0}; i != m_movetypes.size(); i++) {
        cum_prob += m_movetype_freqs[i];
        m_cumulative_probs.push_back(cum_prob);
    }

    // Load precalculated ideal random walk count data
    if (not params.m_num_walks_filename.empty()) {
        std::ifstream num_walks_file {params.m_num_walks_filename};
        boost::archive::binary_iarchive num_walks_arch {num_walks_file};
        num_walks_arch >> m_ideal_random_walks;
    }
}

void GCMCSimulation::construct_movetypes(InputParameters& params) {
    OrigamiMovetypeFile movetypes_file {params.m_movetype_filename};
    vector<string> types {movetypes_file.get_types()};
    vector<string> labels {movetypes_file.get_labels()};
    m_movetype_freqs = movetypes_file.get_freqs();
    for (size_t i {0}; i != types.size(); i++) {
        std::unique_ptr<MCMovetype> movetype {};
        string label {labels[i]};

        // This is still sort of ugly. What if I end up wanting to share
        // options between all CB moves, or between scaffold regrowth and
        // scaffold transform, for example?
        string type {types[i]};
        if (type == "OrientationRotation") {
            movetype =
                    setup_orientation_movetype(i, type, label, movetypes_file);
        }
        else if (type == "MetStapleExchange") {
            movetype = setup_staple_exchange_movetype(
                    i, type, label, movetypes_file);
        }
        else if (type == "MetStapleRegrowth" or type == "CBStapleRegrowth") {
            movetype = setup_staple_regrowth_movetype(
                    i, type, label, movetypes_file);
        }
        else if (
                type == "CTCBScaffoldRegrowth" or
                type == "CTCBJumpScaffoldRegrowth") {
            movetype = setup_scaffold_regrowth_movetype(
                    i, type, label, movetypes_file);
        }
        else if (
                type == "CTRGScaffoldRegrowth" or
                type == "CTRGJumpScaffoldRegrowth") {
            movetype = setup_rg_movetype(i, type, label, movetypes_file);
        }
        else {
            cout << "No such movetype\n";
            throw utility::SimulationMisuse {};
        }
        m_movetypes.emplace_back(std::move(movetype));
    }
}

void GCMCSimulation::set_max_dur(unsigned long long dur) { m_max_duration = dur; }

std::unique_ptr<MCMovetype> GCMCSimulation::setup_orientation_movetype(
        size_t,
        const string&,
        string& label,
        OrigamiMovetypeFile&) {

    auto movetype {std::make_unique<movetypes::OrientationRotationMCMovetype>(
            m_origami_system,
            m_random_gens,
            m_ideal_random_walks,
            m_config_per_move_files,
            label,
            m_ops,
            m_biases,
            m_params)};

    return movetype;
}

std::unique_ptr<MCMovetype> GCMCSimulation::setup_staple_exchange_movetype(
        size_t i,
        const string&,
        string& label,
        OrigamiMovetypeFile& movetypes_file) {

    vector<double> exchange_mults {
            movetypes_file.get_double_vector_option(i, "exchange_mults")};
    bool adaptive_exchange {
            movetypes_file.get_bool_option(i, "adaptive_exchange")};
    auto movetype {std::make_unique<movetypes::MetStapleExchangeMCMovetype>(
            m_origami_system,
            m_random_gens,
            m_ideal_random_walks,
            m_config_per_move_files,
            label,
            m_ops,
            m_biases,
            m_params,
            exchange_mults,
            adaptive_exchange)};

    return movetype;
}

std::unique_ptr<MCMovetype> GCMCSimulation::setup_staple_regrowth_movetype(
        size_t,
        const string& type,
        string& label,
        OrigamiMovetypeFile&) {

    std::unique_ptr<movetypes::MCMovetype> movetype {};
    if (type == "MetStapleRegrowth") {
        movetype = std::make_unique<movetypes::MetStapleRegrowthMCMovetype>(
                m_origami_system,
                m_random_gens,
                m_ideal_random_walks,
                m_config_per_move_files,
                label,
                m_ops,
                m_biases,
                m_params);
    }
    else if (type == "CBStapleRegrowth") {
        movetype = std::make_unique<movetypes::CBStapleRegrowthMCMovetype>(
                m_origami_system,
                m_random_gens,
                m_ideal_random_walks,
                m_config_per_move_files,
                label,
                m_ops,
                m_biases,
                m_params);
    }
    else {
        throw utility::SimulationMisuse {};
    }

    return movetype;
}

std::unique_ptr<MCMovetype> GCMCSimulation::setup_scaffold_regrowth_movetype(
        size_t i,
        const string& type,
        string& label,
        OrigamiMovetypeFile& movetypes_file) {

    size_t excluded_staples {0};
    size_t max_regrowth {static_cast<size_t>(movetypes_file.get_int_option(i, "max_regrowth"))};
    std::unique_ptr<movetypes::MCMovetype> movetype {};
    if (type == "CTCBScaffoldRegrowth") {
        movetype = std::make_unique<movetypes::CTCBScaffoldRegrowthMCMovetype>(
                m_origami_system,
                m_random_gens,
                m_ideal_random_walks,
                m_config_per_move_files,
                label,
                m_ops,
                m_biases,
                m_params,
                excluded_staples,
                max_regrowth);
    }
    else if (type == "CTCBJumpScaffoldRegrowth") {
        size_t max_seg_regrowth {
                static_cast<size_t>(movetypes_file.get_int_option(i, "max_seg_regrowth"))};
        movetype =
                std::make_unique<movetypes::CTCBJumpScaffoldRegrowthMCMovetype>(
                        m_origami_system,
                        m_random_gens,
                        m_ideal_random_walks,
                        m_config_per_move_files,
                        label,
                        m_ops,
                        m_biases,
                        m_params,
                        excluded_staples,
                        max_regrowth,
                        max_seg_regrowth);
    }

    return movetype;
}

std::unique_ptr<MCMovetype> GCMCSimulation::setup_rg_movetype(
        size_t i,
        const string& type,
        string& label,
        OrigamiMovetypeFile& movetypes_file) {

    size_t excluded_staples {0};
    size_t max_num_recoils {static_cast<size_t>(movetypes_file.get_int_option(i, "max_num_recoils"))};
    size_t max_c_attempts {static_cast<size_t>(movetypes_file.get_int_option(i, "max_c_attempts"))};
    size_t max_regrowth {static_cast<size_t>(movetypes_file.get_int_option(i, "max_regrowth"))};
    std::unique_ptr<movetypes::MCMovetype> movetype {};
    if (type == "CTRGScaffoldRegrowth") {
        movetype =
                std::make_unique<movetypes::CTRGScaffoldRegrowthMCMovetype>(
                        m_origami_system,
                        m_random_gens,
                        m_ideal_random_walks,
                        m_config_per_move_files,
                        label,
                        m_ops,
                        m_biases,
                        m_params,
                        excluded_staples,
                        max_num_recoils,
                        max_c_attempts,
                        max_regrowth);
    }
    else if (type == "CTRGJumpScaffoldRegrowth") {
        size_t max_seg_regrowth {
                static_cast<size_t>(movetypes_file.get_int_option(i, "max_seg_regrowth"))};
        movetype =
                std::make_unique<movetypes::CTRGJumpScaffoldRegrowthMCMovetype>(
                        m_origami_system,
                        m_random_gens,
                        m_ideal_random_walks,
                        m_config_per_move_files,
                        label,
                        m_ops,
                        m_biases,
                        m_params,
                        excluded_staples,
                        max_num_recoils,
                        max_c_attempts,
                        max_regrowth,
                        max_seg_regrowth);
    }

    return movetype;
}

unsigned long long GCMCSimulation::simulate(
        unsigned long long steps,
        unsigned long long start_step,
        bool summarize,
        steady_clock::time_point start) {

    unsigned long long step {start_step + 1};
    for (; step != (steps + start_step + 1); step++) {

        // Pick movetype and apply
        MCMovetype& movetype {select_movetype()};
        bool accepted;
        double old_ene {m_origami_system.energy()};
        accepted = movetype.attempt_move(step);
        if (not accepted) {
            // cout << "reset\n";
            movetype.reset_origami();
            m_ops.update_move_params();
            m_biases.calc_move();
        }
        double new_ene {m_origami_system.energy()};
        if (new_ene - old_ene >= 100) {
            cout << "Hang on\n";
        }

        // Center and check constraints
        if (m_centering_freq != 0 and step % m_centering_freq == 0) {
            m_origami_system.center(m_centering_domain);
        }
        if (m_constraint_check_freq != 0 and
            step % m_constraint_check_freq == 0) {
            // cout << "check\n";
            try {
                m_origami_system.check_all_constraints();
            } catch (utility::OrigamiMisuse) {
                write_log_entry(step, accepted, movetype);
                std::chrono::duration<double> dt {
                        (steady_clock::now() - start)};
                for (auto& output_file: m_output_files) {
                    output_file->write(step, dt.count());
                }
                cout << "Origami misuse at constraint check\n";
                break;
            }
        }

        std::chrono::duration<double> dt {(steady_clock::now() - start)};
        if (dt.count() > m_max_duration) {
            cout << "Maximum time allowed reached\n";
            break;
        }

        // Write log entry to standard out
        if (m_logging_freq != 0 and step % m_logging_freq == 0) {
            write_log_entry(step, accepted, movetype);
        }

        // VMD pipe
        if (m_vmd_pipe_freq != 0 and step % m_vmd_pipe_freq == 0) {
            pipe_to_vmd();
        }

        // Update internal simulation variables
        update_internal(step);

        // Write system properties to file
        for (auto& output_file: m_output_files) {
            if (output_file->m_write_freq != 0 and
                step % output_file->m_write_freq == 0) {
                output_file->write(step, dt.count());
            }
        }
    }
    if (summarize) {
        write_log_summary();
    }

    return step;
}

MCMovetype& GCMCSimulation::select_movetype() {
    double prob {m_random_gens.uniform_real()};
    size_t i;
    for (i = 0; i != m_cumulative_probs.size(); i++) {
        if (prob < m_cumulative_probs[i]) {
            break;
        }
    }

    return *m_movetypes[i];
}

void GCMCSimulation::write_log_entry(
        const unsigned long long step,
        bool accepted,
        MCMovetype& movetype) {

    m_logging_stream << "Step: " << step << "\n";
    m_logging_stream << "Temperature: " << m_origami_system.m_temp << "\n";
    m_logging_stream << "Bound staples: " << m_origami_system.num_staples()
                     << "\n";
    m_logging_stream << "Unique bound staples: "
                     << m_origami_system.num_unique_staples() << "\n";
    m_logging_stream << "Fully bound domain pairs: "
                     << m_origami_system.num_fully_bound_domain_pairs() << "\n";
    m_logging_stream << "Misbound domain pairs: "
                     << m_origami_system.num_misbound_domain_pairs() << "\n";
    m_logging_stream << "Stacked domain pairs: "
                     << m_origami_system.num_stacked_domain_pairs() << "\n";
    m_logging_stream << "Linear helix triplets: "
                     << m_origami_system.num_linear_helix_trips() << "\n";
    m_logging_stream << "Stacked junction quadruplets: "
                     << m_origami_system.num_stacked_junct_quads() << "\n";
    m_logging_stream << "Staple counts: ";
    for (auto staple_count: m_origami_system.get_staple_counts()) {
        m_logging_stream << staple_count << " ";
    }
    m_logging_stream << "\n";
    m_logging_stream << "System energy: " << m_origami_system.energy() << "\n";
    m_logging_stream << "Total external bias: " << m_biases.get_total_bias()
                     << "\n";
    m_logging_stream << "Domain update external bias: "
                     << m_biases.get_domain_update_bias() << "\n";
    m_logging_stream << "Move update external bias: "
                     << m_biases.get_move_update_bias() << "\n";
    m_logging_stream << "Movetype: " << movetype.get_label() << "\n";
    m_logging_stream << "Accepted: " << std::boolalpha << accepted << "\n";
    m_logging_stream << "\n";
}

void GCMCSimulation::write_log_summary() {
    m_logging_stream << "Run summary"
                     << "\n\n";
    ofstream movetype_sum_stream {m_params.m_output_filebase + ".moves"};
    for (auto& movetype: m_movetypes) {
        movetype->write_log_summary_header(m_logging_stream);
        movetype->write_log_summary(movetype_sum_stream);
        m_logging_stream << "\n";
    }
    movetype_sum_stream.close();
}

void GCMCSimulation::setup_vmd_pipe() {
    string output_filebase {m_params.m_output_filebase + "_vmd"};
    m_vmd_struct_file = std::make_unique<OrigamiVSFOutputFile>(
            output_filebase + ".vsf",
            0,
            m_params.m_max_total_staples,
            m_origami_system);

    m_vmd_coors_file = std::make_unique<OrigamiVCFOutputFile>(
            output_filebase + ".vcf",
            0,
            m_params.m_max_total_staples,
            m_origami_system);

    m_vmd_states_file = std::make_unique<OrigamiStateOutputFile>(
            output_filebase + ".states",
            0,
            m_params.m_max_total_staples,
            m_origami_system);

    m_vmd_ores_file = std::make_unique<OrigamiOrientationOutputFile>(
            output_filebase + ".ores",
            0,
            m_params.m_max_total_staples,
            m_origami_system);

    pipe_to_vmd();
    if (m_params.m_create_vmd_instance) {
        vmd_proc = std::make_unique<bp::child>(
                bp::search_path("vmd"),
                "-e",
                m_params.m_vmd_file_dir + "/pipe.tcl",
                "-args",
                m_params.m_vmd_file_dir,
                m_params.m_output_filebase + "_vmd",
                bp::std_out > "/dev/null");
    }
}

void GCMCSimulation::pipe_to_vmd() {
    m_vmd_struct_file->open_write_close();
    m_vmd_coors_file->open_write_close();
    m_vmd_states_file->open_write_close();
    m_vmd_ores_file->open_write_close();
}
} // namespace simulation
