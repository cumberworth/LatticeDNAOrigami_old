#ifndef PTMC_SIMULATION_H
#define PTMC_SIMULATION_H

#include <iostream>
#include <utility>
#include <vector>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "bias_functions.hpp"
#include "order_params.hpp"
#include "origami_system.hpp"
#include "parser.hpp"
#include "simulation.hpp"

namespace ptmc {

using std::ofstream;
using std::pair;
using std::vector;

namespace mpi = boost::mpi;

using biasFunctions::SystemBiases;
using orderParams::SystemOrderParams;
using origami::OrigamiSystem;
using parser::InputParameters;
using simulation::GCMCSimulation;

class PTGCMCSimulation: public GCMCSimulation {
    // Base method for parallel tempering in GC ensemble
  public:
    PTGCMCSimulation(
            OrigamiSystem& origami_system,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);
    void run();

  protected:
    // MPI variables
    mpi::environment m_env;
    mpi::communicator m_world;

    // General PTMC parameters
    size_t m_rank {static_cast<size_t>(m_world.rank())};
    size_t m_master_rep {0};
    size_t m_num_reps;
    unsigned long long m_swaps;
    double m_max_pt_dur;
    size_t m_exchange_interval;
    size_t m_config_output_freq;

    ofstream m_swapfile; // Only used by master

    // Could be safer to have only one vector instead of seperating
    // control and dependent. Then no chance of using index on wrong
    // vector
    // Vectors of current replica quantities
    vector<double> m_replica_control_qs = vector<double>(4);
    vector<double> m_replica_dependent_qs = vector<double>(4);

    // Vectors of quantities accross all replicas (used by master only)
    vector<vector<double>> m_control_qs {};

    // Index into the control qs to replica with those qs
    vector<size_t> m_q_to_repi;

    // Indices into control quantities vector for type
    size_t m_temp_i {0};
    size_t m_staple_u_i {1};
    size_t m_bias_mult_i {2};
    size_t m_stacking_mult_i {3};

    // Indices into dependent quantities vector for type
    size_t m_enthalpy_i {0};
    size_t m_staples_i {1};
    size_t m_bias_i {2};
    size_t m_stacking_i {3};

    // Indices of quantities that will be exchanged
    vector<size_t> m_exchange_q_is;

    // Initialization methods
    virtual void initialize_control_qs(InputParameters& params) = 0;
    void initialize_swap_file(InputParameters& params);

    // Communication methods
    void slave_send(size_t swap_i);
    bool slave_receive(size_t swap_i);
    void master_receive(size_t swap_i, vector<vector<double>>& dependent_qs);
    void master_send(size_t swap_i);
    void master_send_kill(size_t swap_i);
    void master_get_dependent_qs(vector<vector<double>>&);
    virtual void update_control_qs() = 0;
    void update_dependent_qs();

    // Exchange methods
    virtual void attempt_exchange(size_t swap_i) = 0;
    bool test_acceptance(double p_accept);
    double calc_acceptance_p(
            vector<pair<double, double>> control_q_pairs,
            vector<pair<double, double>>
                    dependent_q_pairs); // rep1 and rep2 values

    // Output methods
    void write_swap_entry(unsigned long long step);
    virtual void write_acceptance_freqs() = 0;

    void update_internal(unsigned long long) {};
};

class OneDPTGCMCSimulation: public PTGCMCSimulation {
  public:
    OneDPTGCMCSimulation(
            OrigamiSystem& origami_system,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);

  protected:
    void initialize_control_qs(InputParameters& params) override;
    void attempt_exchange(size_t swap_i) override;
    void write_acceptance_freqs() override;

    vector<size_t> m_attempt_count;
    vector<size_t> m_swap_count;
};

class TwoDPTGCMCSimulation: public PTGCMCSimulation {
  public:
    TwoDPTGCMCSimulation(
            OrigamiSystem& origami_system,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);

  protected:
    void initialize_control_qs(InputParameters& params) override;
    void attempt_exchange(size_t swap_i) override;
    void write_acceptance_freqs() override;

  private:
    void update_control_qs() override;

    size_t m_v1_dim;
    size_t m_v2_dim;
    vector<double> m_v1s;
    vector<double> m_v2s;
    vector<size_t> m_i_starts {0, 0, 1, 0};
    vector<size_t> m_j_starts {0, 0, 0, 1};
    vector<size_t> m_i_incrs {2, 1, 2, 1};
    vector<size_t> m_j_incrs {1, 2, 1, 2};
    vector<size_t> m_i_ends {m_v1_dim - 1, m_v1_dim, m_v1_dim - 1, m_v1_dim};
    vector<size_t> m_j_ends {m_v2_dim, m_v2_dim - 1, m_v2_dim, m_v2_dim - 1};
    vector<size_t> m_rep_incrs {m_v2_dim, 1, m_v2_dim, 1};
    vector<vector<vector<size_t>>> m_attempt_count;
    vector<vector<vector<size_t>>> m_swap_count;
};

class TPTGCMCSimulation: public OneDPTGCMCSimulation {
    // Exchange temperatures
  public:
    TPTGCMCSimulation(
            OrigamiSystem& origami_system,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);

  private:
    void update_control_qs() override;
};

class UTPTGCMCSimulation: public OneDPTGCMCSimulation {
    // Exchange temperatures and staple chemical potentials
  public:
    UTPTGCMCSimulation(
            OrigamiSystem& origami_system,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);

  private:
    void update_control_qs() override;
};

class HUTPTGCMCSimulation: public OneDPTGCMCSimulation {
    // Exchange temperatures, staple chemical potentials, and bias multpliers
  public:
    HUTPTGCMCSimulation(
            OrigamiSystem& origami_system,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);

  private:
    void initialize_exchange_vector();
    void update_control_qs() override;
};
} // namespace ptmc

#endif // PTMC_SIMULATION_H
