// nearest_neighbour.h

#ifndef NEAREST_NEIGHBOUR_H
#define NEAREST_NEIGHBOUR_H

#include <utility>
#include <string>
#include <map>
#include <unordered_map>

using std::pair;
using std::string;
using std::map;
using std::unordered_map;

namespace NearestNeighbour {

    // Molar gas consant (J/K/mol)
    const double R {8.3144598};

    // J/cal
    const double J_Per_Cal {4.184};

    // santalucia2004; kcal/mol
    const unordered_map<string, double> NN_Enthalpy {
            {"AA/TT", -7.6},
            {"AT/TA", -7.2},
            {"TA/AT", -7.2},
            {"CA/GT", -8.5},
            {"GT/CA", -8.4},
            {"CT/GA", -7.8},
            {"GA/CT", -8.2},
            {"CG/GC", -10.6},
            {"GC/CG", -9.8},
            {"GG/CC", -8.0},
            {"INITIATION", 0.2},
            {"TERMINAL_AT_PENALTY", 2.2},
            {"SYMMETRY_CORRECTION", 0}};
    
    // kcal/mol/K
    const unordered_map<string, double> NN_Entropy {
            {"AA/TT", -0.0213},
            {"AT/TA", -0.0204},
            {"TA/AT", -0.0213},
            {"CA/GT", -0.0227},
            {"GT/CA", -0.0224},
            {"CT/GA", -0.0210},
            {"GA/CT", -0.0222},
            {"CG/GC", -0.0272},
            {"GC/CG", -0.0244},
            {"GG/CC", -0.0199},
            {"INITIATION", -0.0057},
            {"TERMINAL_AT_PENALTY", 0.0069},
            {"SYMMETRY_CORRECTION", -0.0014}};
    
    const map<char, char> Complementary_Base_Pairs {
    		{'A', 'T'}, {'T', 'A'}, {'G', 'C'}, {'C', 'G'}};

	double calc_hybridization_energy(string seq, double temp, double cation_M);
	string find_longest_contig_complement(string seq_i, string seq_j);
    pair<double, double> calc_hybridization_H_and_S(string seq, double cation_M);
    string calc_complementary_seq(string seq);
    bool seq_is_palindromic(string seq);
}

#endif // NEAREST_NEIGHBOUR_H
