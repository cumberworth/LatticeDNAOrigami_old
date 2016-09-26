//origami_system.h

#ifndef ORIGAMI_SYSTEM_H
#define ORIGAMI_SYSTEM_H

#include <vector>
#include <map>
#include <unordered_map>
#include <utility>
#include <string>
#include <valarray>

#include "utility.h"
#include "nearest_neighbour.h"
#include "hash.h"
#include "domain.h"

using std::vector;
using std::pair;
using std::unordered_map;
using std::string;
using std::valarray;

using namespace Utility;
using namespace DomainContainer;

namespace Origami{

    // For passing information between file objects and origami system
    struct Chain {
        int index;
        int identity;
        vector<VectorThree> positions;
        vector<VectorThree> orientations;
    };
    using Chains = vector<Chain>;

    struct ConstraintViolation {};

    class OrigamiSystem {
        // Cubic lattice domain-level resolution model of DNA origami
        public:
    
            // Configuration independent system properties
            const vector<vector<int>> m_identities;
            const vector<vector<string>> m_sequences;
            const double m_temp;
            const double m_cation_M;
            const double m_staple_M;
            const double m_volume;
            const bool m_cyclic;
            const int c_scaffold {0};
    
            // Constructor and destructor
            OrigamiSystem(
                    const vector<vector<int>>& identities,
                    const vector<vector<string>>& sequences,
                    const Chains& chains,
                    double temp,
                    double lattice_site_volume,
                    double cation_M,
                    double staple_M,
                    bool cyclic);
            ~OrigamiSystem() = default;
    
            // Copy and move
            OrigamiSystem(const OrigamiSystem&) = default;
            OrigamiSystem& operator=(const OrigamiSystem&) = default;
            OrigamiSystem(OrigamiSystem&&) = default;
            OrigamiSystem& operator=(OrigamiSystem&&) = default;
    
            // Configuration properties
            vector<vector<Domain*>> m_domains {};
//            inline unordered_map<int, int> chain_lengths() const {return m_chain_lengths;};
            inline int num_staples() const {return m_domains.size() - 1;};
            int m_num_domains {0};
            inline int num_fully_bound_domains() const {return m_num_fully_bound_domains;};
            inline double energy() const {return m_energy;};
    
            // Staple properties
            inline int num_staples_of_ident(int staple_ident) const {return
                    m_identity_to_index[staple_ident].size();};
            inline vector<int> complimentary_scaffold_domains(int staple_ident)
                    const {return m_staple_ident_to_scaffold_ds[staple_ident];};
    
            // Configuration accessors
            Chains chains() const;
            Occupancy position_occupancy(VectorThree pos) const;
            inline Domain* unbound_domain_at(VectorThree pos) const {return
                    m_pos_to_unbound_d.at(pos);};
    
            // Constraint checkers
            void check_all_constraints();
            double check_domain_constraints(
                    Domain& cd_i,
                    VectorThree pos,
                    VectorThree ore);
    
            // Configuration modifiers
            double unassign_domain(Domain& cd_i);
            int add_chain(int c_i_ident);
            int add_chain(int c_i_ident, int uc_i);
            void delete_chain(int c_i);
            void set_checked_domain_config(
                    Domain& cd_i,
                    VectorThree pos,
                    VectorThree ore);
            double set_domain_config(
                    Domain& cd_i,
                    VectorThree position,
                    VectorThree orientation);
            void set_domain_orientation(Domain& cd_i, VectorThree ore);
            void centre();

            // Keep track of all the chains of each type
            vector<vector<int>> m_identity_to_index {};

            // May need to know chain index by position in domains array directly
            vector<int> m_chain_indices {};
            
            // Configuration checkers
            bool check_domains_complementary(Domain& cd_i, Domain& cd_j);

            // Keeps track of unbound domains but indexed by position
            unordered_map<VectorThree, Domain*> m_pos_to_unbound_d {};
            
            // The index that should be assigned to the next added chain
            int m_current_c_i {};

        protected:
            virtual double bind_noncomplementary_domains(Domain& cd_i, Domain& cd_j);

        private:
    
            // Data

            // Keeps track of all scaffold domains complementary to a domain on
            // a given staple. Only tracks staple identity to the scaffold domain
            // indices
            vector<vector<int>> m_staple_ident_to_scaffold_ds {};

            // May need to access the chain type by index in m_domains only
            vector<int> m_chain_identities {};

            // The state of all positiions occupied by a domain index by position
            unordered_map<VectorThree, Occupancy> m_position_occupancies {};

            // Number of fully complimentary domains bound
            int m_num_fully_bound_domains {};

            // Energy tables index by chain/domain identity pair
            unordered_map<pair<int, int>, double> m_hybridization_energies {};
            unordered_map<pair<int, int>, double> m_stacking_energies {};

            // Current total energy of system
            double m_energy {0};
    
            // Intializers
            void initialize_complementary_associations();
            void initialize_energies();
            void initialize_config(Chains chains);

            // Accessors
            double hybridization_energy(const Domain& cd_i, const Domain& cd_j) const;
            double stacking_energy(const Domain& cd_i, const Domain& cd_j) const;
    
            // States updates
            double unassign_bound_domain(Domain& cd_i);
            void unassign_unbound_domain(Domain& cd_i);
            void update_domain(Domain& cd_i, VectorThree pos, VectorThree ore);
            void update_occupancies(
                    Domain& cd_i,
                    VectorThree position);
    
            // Constraint checkers
            double bind_domain(Domain& cd_i);
            double bind_complementary_domains(Domain& cd_i, Domain& cd_j);
            double check_stacking(Domain& cd_new, Domain& cd_old);
            void check_domain_pair_constraints(Domain& cd_i);
            void check_helical_constraints(Domain& cd_1, Domain& cd_2);
            void check_linear_helix_rear(Domain& cd_3);
            void check_linear_helix(VectorThree ndr_1, Domain& cd_2);
            void check_junction_front(Domain& cd_1);
            void check_junction_rear(Domain& cd_4);
            bool doubly_contiguous_junction(Domain& cd_1, Domain& cd_2);
            void check_doubly_contiguous_junction(Domain& cd_2, Domain& cd_3);
            void check_doubly_contiguous_junction(
                    Domain& cd_1,
                    Domain& cd_2,
                    Domain& cd_3,
                    Domain& cd_4);
            void check_domain_orientations_opposing(Domain& cd_i, Domain& cd_j);
    };

    class OrigamiSystemWithoutMisbinding: public OrigamiSystem {
        public:
            using OrigamiSystem::OrigamiSystem;

        protected:
            double bind_noncomplementary_domains(Domain& cd_i, Domain& cd_j);
    };

    double molarity_to_lattice_volume(double molarity, double lattice_site_volume);

}

#endif // ORIGAMI_H