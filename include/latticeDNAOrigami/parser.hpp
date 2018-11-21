#ifndef PARSER_H
#define PARSER_H

#include "boost/program_options.hpp"

#include <string>
#include <vector>

namespace parser {

namespace po = boost::program_options;

using std::string;
using std::vector;

class InputParameters {
  public:
    InputParameters(int argc, char* argv[]);

    // This constructor is for testing purposes
    InputParameters() {};

    // System input parameters
    string m_origami_input_filename {};
    string m_binding_pot {};
    string m_misbinding_pot {};
    string m_stacking_pot {};
    string m_hybridization_pot {};
    double m_temp {}; // K
    double m_staple_M {}; // mol/L
    double m_cation_M {}; // mol/L
    double m_temp_for_staple_u {}; // K
    double m_staple_u_mult {};
    bool m_constant_staple_M {};
    double m_stacking_ene {};
    double m_binding_h {};
    double m_binding_s {};
    double m_misbinding_h {};
    double m_misbinding_s {};
    size_t m_min_total_staples {};
    size_t m_max_total_staples {};
    size_t m_max_type_staples {};
    vector<int> m_excluded_staples {};
    bool m_domain_update_biases_present {};
    string m_ops_filename {};
    string m_bias_funcs_filename {};
    double m_bias_funcs_mult {};
    string m_energy_filebase {};
    string m_simulation_type {};

    // General simulation parameters
    size_t m_random_seed {};
    string m_movetype_filename {};
    string m_num_walks_filename {};
    string m_restart_traj_file {};
    vector<string> m_restart_traj_files {};
    string m_restart_traj_filebase {};
    string m_restart_traj_postfix {};
    unsigned long long m_restart_step {};
    vector<int> m_restart_steps {};
    string m_vmd_file_dir {};
    size_t m_logging_freq {};
    size_t m_centering_freq {};
    size_t m_centering_domain {};
    size_t m_constraint_check_freq {};
    bool m_allow_nonsensical_ps {};
    double m_max_duration {};

    // Constant temperature parameters
    unsigned long long m_ct_steps {};

    // Enumerator parameters
    bool m_enumerate_staples_only {};

    // Annealing simulation parameters
    double m_max_temp {};
    double m_min_temp {};
    double m_temp_interval {};
    unsigned long long m_steps_per_temp {};

    // Parallel tempering simulation parameters
    vector<double> m_temps {};
    size_t m_num_reps {};
    size_t m_exchange_interval {};
    unsigned long long m_swaps {};
    double m_max_pt_dur {};
    vector<double> m_bias_mults {};
    vector<double> m_stacking_mults {};
    vector<double> m_chem_pot_mults {};
    string m_restart_swap_file {};

    // Umbrella sampling simulation parameters
    string m_us_grid_bias_tag {};
    size_t m_max_num_iters {};
    double m_max_D_bias {};
    unsigned long long m_equil_steps {};
    unsigned long long m_max_equil_dur {};
    unsigned long long m_iter_steps {};
    unsigned long long m_max_iter_dur {};
    unsigned long long m_prod_steps {};
    unsigned long long m_max_prod_dur {};
    double m_max_rel_P_diff {};
    string m_biases_file {};
    string m_biases_filebase {};
    bool m_multi_window {};
    string m_windows_file {};
    string m_local_dir {};
    string m_central_dir {};

    // Output options
    string m_output_filebase {};
    size_t m_configs_output_freq {};
    size_t m_vtf_output_freq {};
    bool m_vcf_per_domain {};
    size_t m_counts_output_freq {};
    size_t m_times_output_freq {};
    size_t m_energies_output_freq {};
    vector<string> m_ops_to_output {};
    size_t m_order_params_output_freq {};
    size_t m_vmd_pipe_freq {};
    bool m_create_vmd_instance {};

  private:
    void set_default_sim_options();
    void process_custom_types(const po::variables_map& vm);
};
} // namespace parser

#endif // PARSER_H
