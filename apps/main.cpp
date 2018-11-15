// main.cpp

#include <iostream>

#include "annealing_simulation.hpp"
#include "bias_functions.hpp"
#include "constant_temp_simulation.hpp"
#include "domain.hpp"
#include "enumerate.hpp"
#include "files.hpp"
#include "nearest_neighbour.hpp"
#include "order_params.hpp"
#include "parser.hpp"
#include "ptmc_simulation.hpp"
#include "us_simulation.hpp"

int main(int argc, char* argv[]) {

    using std::cout;

    parser::InputParameters params {argc, argv};
    origami::OrigamiSystem* origami {origami::setup_origami(params)};
    orderParams::SystemOrderParams& ops {origami->get_system_order_params()};
    biasFunctions::SystemBiases& biases {origami->get_system_biases()};
    select_sim_type_and_run(params, origami, ops, biases);
}

void select_sim_type_and_run() {
    if (params.m_simulation_type == "constant_temp") {
        construct_sim_type_and_run<constantTemp::ConstantTGCMCSimulation>(
                *origami, ops, biases, params);
    }
    else if (params.m_simulation_type == "annealing") {
        construct_sim_type_and_run<annealing::AnnealingGCMCSimulation>(
                *origami, ops, biases, params);
    }
    else if (params.m_simulation_type == "t_parallel_tempering") {
        construct_sim_type_and_run<ptmc::TPTGCMCSimulation>(
                *origami, ops, biases, params);
    }
    else if (params.m_simulation_type == "ut_parallel_tempering") {
        construct_sim_type_and_run<ptmc::UTPTGCMCSimulation>(
                *origami, ops, biases, params);
    }
    else if (params.m_simulation_type == "hut_parallel_tempering") {
        construct_sim_type_and_run<ptmc::HUTPTGCMCSimulation>(
                *origami, ops, biases, params);
    }
    else if (params.m_simulation_type == "2d_parallel_tempering") {
        construct_sim_type_and_run<ptmc::TwoDPTGCMCSimulation>(
                *origami, ops, biases, params);
    }
    else if (params.m_simulation_type == "umbrella_sampling") {
        construct_sim_type_and_run<us::SimpleUSGCMCSimulation>(
                *origami, ops, biases, params);
    }
    else if (params.m_simulation_type == "mw_umbrella_sampling") {
        construct_sim_type_and_run<us::MWUSGCMCSimulation>(
                *origami, ops, biases, params);
    }
    else {
        cout << "No such simulation type.\n";
        std::exit(1);
    }
}
//            cout << "Running serial constant temperature simulation\n";
//            cout << "Running serial annealing simulation\n";
//            cout << "Running T parallel tempering simulation\n";
//            cout << "Running uT parallel tempering simulation\n";
//            cout << "Running HuT parallel tempering simulation\n";
//            cout << "Running 2D (T and stacking) parallel tempering
//            simulation\n"; cout << "Running single window US simulation\n";
//            cout << "Running multi window US simulation\n";
