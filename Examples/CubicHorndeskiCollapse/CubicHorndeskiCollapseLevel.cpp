/* GRChombo
 * Copyright 2012 The GRChombo collaboration.
 * Please refer to LICENSE in GRChombo's root directory.
 */

#include "CubicHorndeskiCollapseLevel.hpp"
#include "AMRReductions.hpp"
// #include "BinaryBH.hpp"
#include "BoxLoops.hpp"
#include "ChiExtractionTaggingCriterion.hpp"
// #include "ChiPunctureExtractionTaggingCriterion.hpp"
#include "ComputePack.hpp"
#include "InitialScalarData.hpp"
#include "ModifiedCCZ4RHS.hpp"
#include "ModifiedGravityConstraints.hpp"
// #include "ModifiedGravityWeyl4.hpp"
#include "NanCheck.hpp"
#include "PositiveChiAndAlpha.hpp"
// #include "PunctureTracker.hpp"
#include "RhoDiagnostics.hpp"
#include "ScalarExtraction.hpp"
#include "SetValue.hpp"
#include "SixthOrderDerivatives.hpp"
#include "SmallDataIO.hpp"
#include "TraceARemoval.hpp"
#include "WeylExtraction.hpp"
#include "KerrBH.hpp"
#include "GammaCalculator.hpp"
#include "CustomExtraction.hpp"

// Things to do during the advance step after RK4 steps
void CubicHorndeskiCollapseLevel::specificAdvance()
{
    // Enforce the trace free A_ij condition and positive chi and alpha
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha()),
                   m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    // Check for nan's
    if (m_p.nan_check)
        BoxLoops::loop(NanCheck("NaNCheck in specific Advance: "), m_state_new,
                       m_state_new, EXCLUDE_GHOST_CELLS, disable_simd());
}

// This initial data uses an approximation for the metric which
// is valid for small boosts
void CubicHorndeskiCollapseLevel::initialData()
{
    CH_TIME("CubicHorndeskiCollapseLevel::initialData");
    if (m_verbosity)
        pout() << "CubicHorndeskiCollapseLevel::initialData " << m_level << endl;

    // First set everything to zero then calculate initial data  Get the Kerr
    // solution in the variables, then calculate the \tilde\Gamma^i numerically
    // as these are non zero and not calculated in the Kerr ICs
    BoxLoops::loop(
        make_compute_pack(SetValue(0.), KerrBH(m_p.kerr_params, m_dx),
                          InitialScalarData(m_p.initial_params, m_dx)),
        m_state_new, m_state_new, INCLUDE_GHOST_CELLS);

    fillAllGhosts();
    BoxLoops::loop(GammaCalculator(m_dx), m_state_new, m_state_new,
                   EXCLUDE_GHOST_CELLS);

}

// Calculate RHS during RK4 substeps
void CubicHorndeskiCollapseLevel::specificEvalRHS(GRLevelData &a_soln,
                                                  GRLevelData &a_rhs,
                                                  const double a_time)
{
    // Enforce positive chi and alpha and trace free A
    BoxLoops::loop(make_compute_pack(TraceARemoval(), PositiveChiAndAlpha()),
                   a_soln, a_soln, INCLUDE_GHOST_CELLS);

    // Calculate ModifiedCCZ4 right hand side with theory_t = CubicHorndeski
    CouplingAndPotential coupling_and_potential(
        m_p.coupling_and_potential_params);
    CubicHorndeskiWithCouplingAndPotential cubic_horndeski(
        coupling_and_potential);
    ModifiedPunctureGauge modified_puncture_gauge(m_p.modified_ccz4_params);

    if (m_p.max_spatial_derivative_order == 4)
    {
        ModifiedCCZ4RHS<CubicHorndeskiWithCouplingAndPotential,
                        ModifiedPunctureGauge, FourthOrderDerivatives>
            my_ccz4_theory(cubic_horndeski, m_p.modified_ccz4_params,
                           modified_puncture_gauge, m_dx, m_p.sigma, m_p.center,
                           m_p.G_Newton);

        BoxLoops::loop(my_ccz4_theory, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
    else if (m_p.max_spatial_derivative_order == 6)
    {
        ModifiedCCZ4RHS<CubicHorndeskiWithCouplingAndPotential,
                        ModifiedPunctureGauge, SixthOrderDerivatives>
            my_ccz4_theory(cubic_horndeski, m_p.modified_ccz4_params,
                           modified_puncture_gauge, m_dx, m_p.sigma, m_p.center,
                           m_p.G_Newton);

        BoxLoops::loop(my_ccz4_theory, a_soln, a_rhs, EXCLUDE_GHOST_CELLS);
    }
}

// enforce trace removal during RK4 substeps
void CubicHorndeskiCollapseLevel::specificUpdateODE(GRLevelData &a_soln,
                                                    const GRLevelData &a_rhs,
                                                    Real a_dt)
{
    // Enforce the trace free A_ij condition
    BoxLoops::loop(TraceARemoval(), a_soln, a_soln, INCLUDE_GHOST_CELLS);
}

void CubicHorndeskiCollapseLevel::preTagCells()
{
    // We only use chi in the tagging criterion so only fill the ghosts for chi
    fillAllGhosts(VariableType::evolution, Interval(c_chi, c_chi));
}

// specify the cells to tag
void CubicHorndeskiCollapseLevel::computeTaggingCriterion(
    FArrayBox &tagging_criterion, const FArrayBox &current_state)
{
    
        BoxLoops::loop(ChiExtractionTaggingCriterion(m_dx, m_level,
                                                     m_p.extraction_params,
                                                     m_p.activate_extraction),
                       current_state, tagging_criterion);
}

void CubicHorndeskiCollapseLevel::specificPostTimeStep()
{
    CH_TIME("CubicHorndeskiCollapseLevel::specificPostTimeStep");

    bool first_step =
        (m_time == 0.); // this form is used when 'specificPostTimeStep' was
                        // called during setup at t=0 from Main
    // bool first_step = (m_time == m_dt); // if not called in Main


    if (m_p.calculate_constraint_norms)
    {
        CouplingAndPotential coupling_and_potential(
            m_p.coupling_and_potential_params);
        CubicHorndeskiWithCouplingAndPotential cubic_horndeski(
            coupling_and_potential);
        fillAllGhosts();
        BoxLoops::loop(
            ModifiedGravityConstraints<CubicHorndeskiWithCouplingAndPotential>(
                cubic_horndeski, m_dx, m_p.center, m_p.G_Newton, c_Ham,
                Interval(c_Mom1, c_Mom3)),
            m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);
        if (m_level == 0)
        {
            AMRReductions<VariableType::diagnostic> amr_reductions(m_gr_amr);
            double L2_Ham = amr_reductions.norm(c_Ham);
            double L2_Mom = amr_reductions.norm(Interval(c_Mom1, c_Mom3));
            SmallDataIO constraints_file(m_p.data_path + "constraint_norms",
                                         m_dt, m_time, m_restart_time,
                                         SmallDataIO::APPEND, first_step);
            constraints_file.remove_duplicate_time_data();
            if (first_step)
            {
                constraints_file.write_header_line({"L^2_Ham", "L^2_Mom"});
            }
            constraints_file.write_time_data_line({L2_Ham, L2_Mom});
        }




    }

    // do puncture tracking on requested level
    //if (m_p.track_punctures && m_level == m_p.puncture_tracking_level)
    //{
      //  CH_TIME("PunctureTracking");
        // only do the write out for every coarsest level timestep
      //  int coarsest_level = 0;
       // bool write_punctures = at_level_timestep_multiple(coarsest_level);
       // m_bh_amr.m_puncture_tracker.execute_tracking(m_time, m_restart_time,
         //                                            m_dt, write_punctures);
 //   }

#ifdef USE_AHFINDER
    if (m_p.AH_activate && m_level == m_p.AH_params.level_to_run)
    {
    //    if (m_p.AH_set_origins_to_punctures && m_p.track_punctures)
     //   {
      //      m_bh_amr.m_ah_finder.set_origins(
     //           m_bh_amr.m_puncture_tracker.get_puncture_coords());
      //  }
        m_bh_amr.m_ah_finder.solve(m_dt, m_time, m_restart_time);
    }
#endif
}

#ifdef CH_USE_HDF5
// Things to do before a plot level - need to calculate the Wyl scalars
void CubicHorndeskiCollapseLevel::prePlotLevel()
{
    CouplingAndPotential coupling_and_potential(
        m_p.coupling_and_potential_params);
    CubicHorndeskiWithCouplingAndPotential cubic_horndeski(
        coupling_and_potential);
    fillAllGhosts();
    BoxLoops::loop(
        ModifiedGravityConstraints<CubicHorndeskiWithCouplingAndPotential>(
            cubic_horndeski, m_dx, m_p.center, m_p.G_Newton, c_Ham, 
            Interval(c_Mom1, c_Mom3)), 
        m_state_new, m_state_diagnostics, EXCLUDE_GHOST_CELLS);


     // Use AMR Interpolator and do lineout data extraction
            // set up an interpolator
            // pass the boundary params so that we can use symmetries if
            // applicable
            AMRInterpolator<Lagrange<4>> interpolator(
                m_bh_amr, m_p.origin, m_p.dx, m_p.boundary_params,
                m_p.verbosity);

            // this should fill all ghosts including the boundary ones according
            // to the conditions set in params.txt
            interpolator.refresh();

            // set up the query and execute it
            std::array<double, CH_SPACEDIM> extraction_origin = {
                0., 0., 0.}; // specified point {x \in [0,L],y \in
                                           // [0,L], z \in [0,L]}


     // Ham lineout
    CustomExtraction Ham_extraction(c_Ham, m_p.lineout_num_points,
        m_p.L, extraction_origin, m_dt,
        m_time);
    Ham_extraction.execute_query(&interpolator,
     m_p.data_path + "Ham_lineout");

          // Ham abs lineout
    CustomExtraction Ham_abs_extraction(c_Ham_abs_sum, m_p.lineout_num_points,
        m_p.L, extraction_origin, m_dt,
        m_time);
    Ham_abs_extraction.execute_query(&interpolator,
     m_p.data_path + "Ham_abs_lineout");


     // Mom lineout
    CustomExtraction Mom_extraction(c_Mom1, m_p.lineout_num_points,
        m_p.L, extraction_origin, m_dt,
        m_time);
    Mom_extraction.execute_query(&interpolator,
     m_p.data_path + "Mom_lineout");

    // Mom abs lineout
    CustomExtraction Mom_abs_extraction(c_Mom_abs_sum, m_p.lineout_num_points,
        m_p.L, extraction_origin, m_dt,
        m_time);
    Mom_abs_extraction.execute_query(&interpolator,
     m_p.data_path + "Mom_abs_lineout");


    
    
}
#endif /* CH_USE_HDF5 */
