// main.cpp

#include <iostream>
#include <memory>

#include "annealing_simulation.hpp"
#include "bias_functions.hpp"
#include "constant_temp_simulation.hpp"
#include "domain.hpp"
#include "files.hpp"
#include "nearest_neighbour.hpp"
#include "order_params.hpp"
#include "parser.hpp"
#include "ptmc_simulation.hpp"
//#include "us_simulation.hpp"

int main(int argc, char* argv[]) {

    using std::cout;

    parser::InputParameters params {argc, argv};
    auto origami = origami::setup_origami(params);
    orderParams::SystemOrderParams& ops {origami.get_system_order_params()};
    biasFunctions::SystemBiases& biases {origami.get_system_biases()};

    // Select simulation type
    if (params.m_simulation_type == "constant_temp") {
        constantTemp::ConstantTGCMCSimulation sim {
                origami, ops, biases, params};
        sim.run();
    }
    else if (params.m_simulation_type == "annealing") {
        annealing::AnnealingGCMCSimulation sim {
                origami, ops, biases, params};
        sim.run();
    }
    else if (params.m_simulation_type == "t_parallel_tempering") {
        ptmc::TPTGCMCSimulation sim {origami, ops, biases, params};
        sim.run();
    }
    else if (params.m_simulation_type == "ut_parallel_tempering") {
        ptmc::UTPTGCMCSimulation sim {origami, ops, biases, params};
        sim.run();
    }
    else if (params.m_simulation_type == "hut_parallel_tempering") {
        ptmc::HUTPTGCMCSimulation sim {origami, ops, biases, params};
        sim.run();
    }
    else if (params.m_simulation_type == "2d_parallel_tempering") {
        ptmc::TwoDPTGCMCSimulation sim {origami, ops, biases, params};
        sim.run();
    }
//    else if (params.m_simulation_type == "umbrella_sampling") {
//        us::SimpleUSGCMCSimulation sim {*origami, ops, biases, params};
//        sim.run();
//    }
//    else if (params.m_simulation_type == "mw_umbrella_sampling") {
//        us::MWUSGCMCSimulation sim {*origami, ops, biases, params};
//        sim.run();
//    }
    else {
        std::exit(1);
    }

}
