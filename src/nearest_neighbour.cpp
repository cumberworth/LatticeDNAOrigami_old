// nearest_neighbour.h

#include <utility>
#include <string>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iostream>

#include "nearest_neighbour.h"

using std::cout;
using std::tuple;
using std::string;
using std::vector;
using std::tie;
using std::log;
using std::reverse;

using namespace NearestNeighbour;

double NearestNeighbour::calc_hybridization_energy(
        string seq,
        double temp,
        double cation_M) {
    double DH;
    double DS;
    tuple<double, double> tied_result {calc_hybridization_H_and_S(seq, cation_M)};
    DH = std::get<0>(tied_result);
    DS = std::get<1>(tied_result);
    double DG {DH - temp * DS};
    
    // Convert from kcal/mol to unitless dimension
    DG = DG * J_Per_Cal * 1000 / R / temp;

    return DG;
}

double NearestNeighbour::calc_stacking_energy(
        string seq_i,
        string seq_j,
        double temp,
        double cation_M) {
    string nuc_i_back {seq_i.back()};
    string nuc_j_front {seq_j.front()};
    return 0;
}

tuple<double, double> NearestNeighbour::calc_hybridization_H_and_S(string seq, double cation_M) {
    string comp_seq {calc_comp_seq(seq)};

    // Initiation free energy
    double DH_init {NN_Enthalpy.at("INITIATION")};
    double DS_init {NN_Entropy.at("INITIATION")};

    // Symmetry penalty for palindromic sequences
    double DS_sym {0};
    if (seq_is_palindromic(seq)) {
        DS_sym = NN_Entropy.at("SYMMETRY_CORRECTION");
    }

    // Stacking energies (bound)
    double DH_stack {0};
    double DS_stack {0};
    for (unsigned int nuc_i {0}; nuc_i != seq.size() - 1; nuc_i++) {
        string c1 {seq[nuc_i]};
        string c2 {seq[nuc_i + 1]};
        string seq_pair {c1 + c2};
        string c3 {comp_seq[nuc_i]};
        string c4 {comp_seq[nuc_i + 1]};
        string comp_seq_pair {c3 + c4};
        string key {seq_pair + "/" + comp_seq_pair};
        DH_stack += NN_Enthalpy.at(key);
        DS_stack += NN_Entropy.at(key);
    }

    // Terminal AT penalties
    int terminal_at_pairs {0};
    if (seq.front() == 'A' or seq.front() == 'T') {
        terminal_at_pairs += 1;
    }
    if (seq.back() == 'A' or seq.front() == 'T') {
        terminal_at_pairs +=1;
    }
    double DH_at {NN_Enthalpy.at("TERMINAL_AT_PENALTY")};
    double DS_at {NN_Entropy.at("TERMINAL_AT_PENALTY")};

    // Sum and correct for salt
    double DH_hybrid = DH_init + DH_stack + DH_at;
    double DS_hybrid = DS_init + DS_sym + DS_stack + DS_at;
    DS_hybrid += 0.368 * (seq.size() / 2) * log(cation_M) / 1000;

    tuple<double, double> tied_result {tie(DH_hybrid, DS_hybrid)};
    return tied_result;
}

vector<string> NearestNeighbour::find_longest_contig_complement(
        string seq_i,
        string seq_j) {
    // Find smallest sequence
    string seq_three;
    string seq_five;
    if (seq_i.size() < seq_j.size()) {
        seq_three = seq_j;
        reverse(seq_three.begin(), seq_three.end());
        seq_five = seq_i;
    }
    else {
        seq_three = seq_i;
        reverse(seq_three.begin(), seq_three.end());
        seq_five = seq_j;
    }

    seq_three = calc_comp_seq(seq_three);

    // Iterate through all lengths and starting points
    vector<string> comp_seqs {};
    for (int subseq_len {(int)seq_three.size()}; subseq_len != 0; subseq_len--) {
        for (unsigned int start_i {0}; start_i != (seq_three.size() - subseq_len
                    + 1); start_i++) {
            string subseq {seq_three.substr(start_i, subseq_len)};
            std::size_t seq_i = seq_five.find(subseq);
            if (seq_i == string::npos) {
                continue;
            }
            else {
                string comp_seq {seq_five.substr(seq_i, subseq_len)};
                comp_seqs.push_back(comp_seq);
                // Only considers one instance of the subsequence. To be 
                // consistent consider checking for all instances and weight
                // appropreatelly
            }
        }
        if (comp_seqs.empty()) {
            continue;
        }
        else {
            return comp_seqs;
        }
    }
    return comp_seqs;
}

string NearestNeighbour::calc_comp_seq(string seq) {
    // Return the complementary DNA sequence.
    string comp_seq {};
    for (auto base: seq) {
        comp_seq += Complementary_Base_Pairs.at(base);
	}
    return comp_seq;
}

bool NearestNeighbour::seq_is_palindromic(string seq) {
    string comp_seq {calc_comp_seq(seq)};
    string reverse_comp_seq {comp_seq};
    reverse(reverse_comp_seq.begin(), reverse_comp_seq.end());
    bool palindromic {reverse_comp_seq == seq};
    return palindromic;
}