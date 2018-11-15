// random_gens.cpp

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

RandomGens::~RandomGens() {
    for (auto const key: m_uniform_int_dists) {
        m_uniform_int_dists.erase(key.first);
    }
}

void RandomGens::set_seed(int seed) { m_random_engine.seed(seed); }

double RandomGens::uniform_real() {
    return m_uniform_real_dist(m_random_engine);
}

int RandomGens::uniform_int(int lower, int upper) {

    // Check if distribution used previously
    pair<int, int> key {lower, upper};
    int random_int;
    if (m_uniform_int_dists.find(key) != m_uniform_int_dists.end()) {
        auto dist {m_uniform_int_dists.at(key)};
        random_int = dist(m_random_engine);
    }

    // If not, make it and store it
    else {
        auto dist {std::make_unique<std::uniform_int_distribution<int>>(
                lower, upper)};
        random_int = (*dist)(m_random_engine);
        m_uniform_int_dists.insert({key, *dist});
    }

    return random_int;
}
} // namespace randomGen
