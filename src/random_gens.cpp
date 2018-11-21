#include <iostream>
#include <memory>
#include <random>

#include "random_gens.hpp"

namespace randomGen {

RandomGens::RandomGens() {

    // Seed random number generator
    std::random_device true_random_engine {};
    auto seed {true_random_engine()};
    m_random_engine.seed(seed);
}

void RandomGens::set_seed(unsigned int seed) { m_random_engine.seed(seed); }

double RandomGens::uniform_real() {
    return m_uniform_real_dist(m_random_engine);
}

int RandomGens::uniform_int(int lower, int upper) {
    pair<int, int> key {lower, upper};
    m_uniform_int_dists.try_emplace(key, lower, upper);

    return m_uniform_int_dists[key](m_random_engine);
}
} // namespace randomGen
