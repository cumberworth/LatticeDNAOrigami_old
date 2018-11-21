#include <algorithm>
#include <fstream>

#include <boost/math/special_functions/factorials.hpp>

#include "ideal_random_walk.hpp"

namespace idealRandomWalk {

using boost::math::factorial;

long double IdealRandomWalks::num_walks(
        VectorThree start_pos,
        VectorThree end_pos,
        unsigned int steps) {

    // Check stored values
    VectorThree DR {end_pos - start_pos};

    // Only work with one permutation of DR
    DR = DR.absolute().sort();
    pair<VectorThree, int> walk_key {DR, steps};
    if (m_num_walks.find(walk_key) != m_num_walks.end()) {
        return m_num_walks.at(walk_key);
    }
    int DX {DR[0]};
    int DY {DR[1]};
    int DZ {DR[2]};
    auto cast_steps {static_cast<int>(steps)};
    int Nminus {(cast_steps - DX - DY - DZ) / 2};
    int Nplus {(cast_steps - DX - DY + DZ) / 2};
    long double walks {0};

    int DR_sum {DX + DY + DZ};
    if (DR_sum > cast_steps or (cast_steps - DR_sum) % 2 != 0) {

        // Add entry
        m_num_walks[walk_key] = walks;
        return walks;
    }

    // Need some negative steps to reach a negative location
    for (int ybar {0}; ybar != Nminus + 1; ybar++) {
        for (int xbar {0}; xbar != Nminus + 1 - ybar; xbar++) {
            auto f1 {factorial<long double>(static_cast<unsigned int>(steps))};
            auto f2 {factorial<long double>(static_cast<unsigned int>(xbar))};
            if (xbar + DX < 0) {
                continue;
            }

            auto f3 {factorial<long double>(static_cast<unsigned int>(xbar + DX))};
            auto f4 {factorial<long double>(static_cast<unsigned int>(ybar))};
            if (ybar + DY < 0) {
                continue;
            }

            auto f5 {factorial<long double>(static_cast<unsigned int>(ybar + DY))};
            auto f6 {factorial<long double>(static_cast<unsigned int>(Nminus - xbar - ybar))};
            if (Nplus - xbar - ybar < 0) {
                continue;
            }

            auto f7 {factorial<long double>(static_cast<unsigned int>(Nplus - xbar - ybar))};
            walks += f1 / (f2 * f3 * f4 * f5 * f6 * f7);
        }
    }

    // Add entries
    m_num_walks[walk_key] = walks;

    return walks;
}

void IdealRandomWalks::delete_entry(
        VectorThree start_pos,
        VectorThree end_pos,
        int steps) {

    VectorThree DR {end_pos - start_pos};
    DR = DR.absolute().sort();
    pair<VectorThree, int> walk_key {DR, steps};
    m_num_walks.erase(walk_key);
}
} // namespace idealRandomWalk
