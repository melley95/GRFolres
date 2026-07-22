/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef CHIANDPHI2TAGGINGCRITERION_HPP_
#define CHIANDPHI2TAGGINGCRITERION_HPP_

#include "Cell.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "FourthOrderDerivatives.hpp"
#include "Tensor.hpp"

class ChiAndPhi2TaggingCriterion
{
  protected:
    const double m_dx;
    const FourthOrderDerivatives m_deriv;
    const double m_threshold_chi;
    const double m_threshold_phi;

    /// Vars object for chi and phi2
    template <class data_t> struct Vars
    {
        data_t chi; //!< Conformal factor
        data_t phi2; // Collapse field
        data_t Pi2;

        template <typename mapping_function_t>
        void enum_mapping(mapping_function_t mapping_function)
        {
            using namespace VarsTools; // define_enum_mapping is part of
                                       // VarsTools
            define_enum_mapping(mapping_function, c_chi, chi);
            define_enum_mapping(mapping_function, c_phi2, phi2);
            define_enum_mapping(mapping_function, c_Pi2, Pi2);

        }
    };

  public:
    ChiAndPhi2TaggingCriterion(const double dx, const double threshold_chi,
                              const double threshold_phi)
        : m_dx(dx), m_deriv(dx), m_threshold_chi(threshold_chi),
          m_threshold_phi(threshold_phi){};

    template <class data_t> void compute(Cell<data_t> current_cell) const
    {
        const auto d2 = m_deriv.template diff2<Vars>(current_cell);


        data_t mod_d2_chi = 0;
        data_t mod_d2_phi = 0;

        FOR(idir, jdir)
        {
            mod_d2_chi += d2.chi[idir][jdir] * d2.chi[idir][jdir];

            mod_d2_phi += d2.Pi2[idir][jdir] * d2.Pi2[idir][jdir] +
                          d2.phi2[idir][jdir] * d2.phi2[idir][jdir];
        }

        data_t criterion_chi = m_dx / m_threshold_chi * sqrt(mod_d2_chi);

        data_t criterion_phi = m_dx / m_threshold_phi * sqrt(mod_d2_phi);

        data_t criterion = simd_max(criterion_chi, criterion_phi);

        // Write back into the flattened Chombo box
        current_cell.store_vars(criterion, 0);
    }
};

#endif /* CHIANDPHI2TAGGINGCRITERION_HPP_ */
