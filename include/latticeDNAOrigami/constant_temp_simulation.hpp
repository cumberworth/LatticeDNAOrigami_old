#ifndef CONSTANT_TEMP_SIMULATION_H
#define CONSTANT_TEMP_SIMULATION_H

#include <vector>

#include "bias_functions.hpp"
#include "order_params.hpp"
#include "origami_system.hpp"
#include "parser.hpp"
#include "simulation.hpp"

namespace constantTemp {

using std::vector;

using biasFunctions::SystemBiases;
using orderParams::SystemOrderParams;
using origami::OrigamiSystem;
using parser::InputParameters;
using simulation::GCMCSimulation;

class ConstantTGCMCSimulation: public GCMCSimulation {
  public:
    ConstantTGCMCSimulation(
            OrigamiSystem& origami_system,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);
    void run() { simulate(m_steps); }
    vector<double> get_energies();
    vector<int> get_staples();
    vector<int> get_domains();

    int m_op_freq {0};

  private:
    void update_internal(long long int step);
    long long int m_steps;
    vector<double> m_enes {};
    vector<int> m_staples {};
    vector<int> m_domains {};
};
} // namespace constantTemp

#endif // CONSTANT_TEMP_SIMULATION_H