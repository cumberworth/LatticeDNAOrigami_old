#ifndef ORIENTATION_MOVETYPE_H
#define ORIENTATION_MOVETYPE_H

#include "movetypes.hpp"

namespace movetypes {

class OrientationRotationMCMovetype: public MCMovetype {

  public:
    OrientationRotationMCMovetype(
            OrigamiSystem& origami_system,
            RandomGens& random_gens,
            IdealRandomWalks& ideal_random_walks,
            vector<std::unique_ptr<OrigamiOutputFile>>& config_files,
            string& label,
            SystemOrderParams& ops,
            SystemBiases& biases,
            InputParameters& params);
    OrientationRotationMCMovetype(const OrientationRotationMCMovetype&) =
            delete;
    OrientationRotationMCMovetype& operator=(
            const OrientationRotationMCMovetype&) = delete;

    void write_log_summary(ostream& log_stream) override final;

  private:
    bool internal_attempt_move() override;
    void add_external_bias() override final {}
    void add_tracker(bool accepted) override;
    utility::OrientationRotationTracking m_tracker {};
};
} // namespace movetypes

#endif // ORIENTATION_MOVETYPE_H
