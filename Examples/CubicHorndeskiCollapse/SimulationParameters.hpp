/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#ifndef SIMULATIONPARAMETERS_HPP_
#define SIMULATIONPARAMETERS_HPP_

// General includes
#include "GRParmParse.hpp"
#include "ModifiedGravitySimulationParametersBase.hpp"

// Problem specific includes:
#include "BoostedBH.hpp"
#include "CCZ4.hpp"
#include "CouplingAndPotential.hpp"
#include "CubicHorndeski.hpp"
#include "InitialScalarData.hpp"
#include "KerrBH.hpp"

class SimulationParameters : public ModifiedGravitySimulationParametersBase<
                                 CubicHorndeski<CouplingAndPotential>>
{
  public:
    SimulationParameters(GRParmParse &pp)
        : ModifiedGravitySimulationParametersBase(pp)
    {
        read_params(pp);
        check_params();
    }

    /// Read parameters
    void read_params(GRParmParse &pp)
    {
        // Do we want puncture tracking and constraint norm calculation?
     //   pp.load("track_punctures", track_punctures, false);
      //  pp.load("puncture_tracking_level", puncture_tracking_level, max_level);
        pp.load("calculate_constraint_norms", calculate_constraint_norms,
                false);

        // Coupling and potential
        pp.load("scalar_mass", coupling_and_potential_params.scalar_mass);
        pp.load("g3", coupling_and_potential_params.g3);
        pp.load("g2", coupling_and_potential_params.g2);

        // Initial data
        pp.load("G_Newton", G_Newton, 1.0);

        // Initial scalar field data
        initial_params.center =
            center; // already read in SimulationParametersBase
        pp.load("scalar_amplitude", initial_params.amplitude, 0.);
        pp.load("scalar_width", initial_params.width, 1.0);
        pp.load("scalar_r0", initial_params.r0, 0.);

        // Initial Kerr data
        pp.load("kerr_mass", kerr_params.mass);
        pp.load("kerr_spin", kerr_params.spin);
        pp.load("kerr_center", kerr_params.center, center);

        // Lineout params
        pp.load("lineout_num_points", lineout_num_points, 10);

#ifdef USE_AHFINDER
        pp.load("AH_initial_guess", AH_initial_guess, 0.5 * kerr_params.mass);
#endif
    }

    void check_params()
    {
        warn_parameter("kerr_mass", kerr_params.mass, kerr_params.mass >= 0.0,
                       "should be >= 0.0");
        check_parameter("kerr_spin", kerr_params.spin,
                        std::abs(kerr_params.spin) <= kerr_params.mass,
                        "must satisfy |a| <= M = " +
                            std::to_string(kerr_params.mass));
        FOR(idir)
        {
            std::string name = "kerr_center[" + std::to_string(idir) + "]";
            warn_parameter(
                name, kerr_params.center[idir],
                (kerr_params.center[idir] >= 0) &&
                    (kerr_params.center[idir] <= (ivN[idir] + 1) * coarsest_dx),
                "should be within the computational domain");
        }
    }

    bool calculate_constraint_norms;


    // Collection of parameters necessary for initial conditions
    double G_Newton;
    InitialScalarData::params_t initial_params;
    CouplingAndPotential::params_t coupling_and_potential_params;
    KerrBH::params_t kerr_params;

    int lineout_num_points;

#ifdef USE_AHFINDER
    double AH_initial_guess;
#endif
};

#endif /* SIMULATIONPARAMETERS_HPP_ */
