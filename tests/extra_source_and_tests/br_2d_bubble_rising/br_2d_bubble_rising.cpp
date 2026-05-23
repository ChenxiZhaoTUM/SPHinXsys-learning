/**
 * @file    br_2d_bubble_rising.cpp
 * @brief   2-D bubble rising simulation without heat transfer.
 * @details The original natural-convection/diffusion case has been reduced to
 *          two immiscible weakly-compressible fluids with gravity, viscosity,
 *          wall interaction and surface tension. No scalar transport field, scalar diffusion, scalar flux or scalar
 *          post-processing remains.
 */
#include "br_2d_bubble_rising.h"

using namespace SPH;
//----------------------------------------------------------------------
// Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    // Build up the SPH system.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    // Tag for run particle relaxation for the initial body fitted distribution.
    sph_system.setRunParticleRelaxation(false);
    // Tag for computation start with relaxed body fitted particles distribution.
    sph_system.setReloadParticles(true);
    // Handle command line arguments and override the tags for particle relaxation and reload.
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    // Creating bodies, materials and particles.
    //----------------------------------------------------------------------
    FluidBody liquid_body(sph_system, makeShared<LiquidBody>("LiquidBody"));
    liquid_body.defineComponentLevelSetShape("OuterBoundary");
    liquid_body.defineClosure<WeaklyCompressibleFluid, Viscosity>(
        ConstructArgs(rho0_l, c_f), mu_l);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? liquid_body.generateParticles<BaseParticles, Reload>(liquid_body.getName())
        : liquid_body.generateParticles<BaseParticles, Lattice>();

    FluidBody bubble_body(sph_system, makeShared<BubbleBody>("BubbleBody"));
    bubble_body.defineBodyLevelSetShape();
    bubble_body.defineClosure<WeaklyCompressibleFluid, Viscosity>(
        ConstructArgs(rho0_g, c_f), mu_g);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? bubble_body.generateParticles<BaseParticles, Reload>(bubble_body.getName())
        : bubble_body.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineMaterial<Solid>();
    wall_boundary.generateParticles<BaseParticles, Lattice>();
    
    SolidBody no_slip_wall(sph_system, makeShared<NoSlipWall>("NoSlipWall"));
    no_slip_wall.defineMaterial<Solid>();
    no_slip_wall.generateParticles<BaseParticles, Lattice>();

    ObserverBody bubble_observer(sph_system, "BubbleObserver");
    bubble_observer.generateParticles<ObserverParticles>(createObservationPoints());

    //----------------------------------------------------------------------
    // Define body relations.
    //----------------------------------------------------------------------
    InnerRelation liquid_inner(liquid_body);
    InnerRelation bubble_inner(bubble_body);

    ContactRelation liquid_contact_bubble(liquid_body, {&bubble_body});
    ContactRelation liquid_contact_wall(liquid_body, {&wall_boundary});
    ContactRelation liquid_contact_no_slip_wall(liquid_body, {&no_slip_wall});
    ContactRelation liquid_contact(liquid_body, {&bubble_body, &wall_boundary});

    ContactRelation bubble_contact_liquid(bubble_body, {&liquid_body});
    ContactRelation bubble_contact_wall(bubble_body, {&wall_boundary});
    ContactRelation bubble_contact_no_slip_wall(bubble_body, {&no_slip_wall});
    ContactRelation bubble_contact(bubble_body, {&liquid_body, &wall_boundary});

    ComplexRelation liquid_complex(liquid_inner, {&liquid_contact_bubble, &liquid_contact_wall, &liquid_contact_no_slip_wall});
    ComplexRelation bubble_complex(bubble_inner, {&bubble_contact_liquid, &bubble_contact_wall, &bubble_contact_no_slip_wall});

    ContactRelation observer_contact(bubble_observer, {&bubble_body});
    //----------------------------------------------------------------------
    //	Run particle relaxation for body-fitted distribution if chosen.
    //----------------------------------------------------------------------
    if (sph_system.RunParticleRelaxation())
    {
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_inserted_body_particles(bubble_body);
        SimpleDynamics<RandomizeParticlePosition> random_water_body_particles(liquid_body);
        BodyStatesRecordingToVtp write_real_body_states(sph_system);
        ReloadParticleIO write_real_body_particle_reload_files({&bubble_body, &liquid_body});
        RelaxationStepLevelSetCorrectionInner relaxation_step_inner(bubble_inner);
        RelaxationStepLevelSetCorrectionComplex relaxation_step_complex(
            InteractArgs(liquid_inner, std::string("OuterBoundary")), liquid_contact_bubble);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_inserted_body_particles.exec(0.25);
        random_water_body_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_complex.SurfaceBounding().exec();
        write_real_body_states.writeToFile(0);

        int ite_p = 0;
        while (ite_p < 1000)
        {
            relaxation_step_inner.exec();
            relaxation_step_complex.exec();
            ite_p += 1;
            if (ite_p % 200 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
                write_real_body_states.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process finish !" << std::endl;

        write_real_body_particle_reload_files.writeToFile(0);

        return 0;
    }
    //----------------------------------------------------------------------
    // Define numerical methods used in the simulation.
    //----------------------------------------------------------------------
    Gravity gravity(Vecd(0.0, -gravity_g));
    SimpleDynamics<GravityForce<Gravity>> liquid_gravity(liquid_body, gravity);
    SimpleDynamics<GravityForce<Gravity>> bubble_gravity(bubble_body, gravity);

    SimpleDynamics<NormalDirectionFromBodyShape> liquid_normal_direction(liquid_body);
    SimpleDynamics<NormalDirectionFromBodyShape> bubble_normal_direction(bubble_body);
    SimpleDynamics<NormalDirectionFromBodyShape> wall_normal_direction(wall_boundary);
    SimpleDynamics<NormalDirectionFromBodyShape> no_slip_wall_normal_direction(no_slip_wall);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex>
        liquid_kernel_correction_complex(InteractArgs(liquid_inner, 0.5), liquid_contact);
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex>
        bubble_kernel_correction_complex(InteractArgs(bubble_inner, 0.5), bubble_contact);

    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfWithWallRiemann>
        liquid_pressure_relaxation(liquid_inner, liquid_contact_bubble, liquid_contact_wall);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann>
        liquid_density_relaxation(liquid_inner, liquid_contact_bubble, liquid_contact_wall);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfWithWallRiemann>
        bubble_pressure_relaxation(bubble_inner, bubble_contact_liquid, bubble_contact_wall);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann>
        bubble_density_relaxation(bubble_inner, bubble_contact_liquid, bubble_contact_wall);

    InteractionWithUpdate<fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>>
        liquid_update_density_by_summation(liquid_inner, liquid_contact_bubble, liquid_contact_wall);
    InteractionWithUpdate<fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>>
        bubble_update_density_by_summation(bubble_inner, bubble_contact_liquid, bubble_contact_wall);

    InteractionWithUpdate<fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>>
        liquid_transport_correction(liquid_inner, liquid_contact_bubble, liquid_contact_wall);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>>
        bubble_transport_correction(bubble_inner, bubble_contact_liquid, bubble_contact_wall);

    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall>
        liquid_viscous_force(liquid_inner, liquid_contact_bubble, liquid_contact_no_slip_wall);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall>
        bubble_viscous_force(bubble_inner, bubble_contact_liquid, bubble_contact_no_slip_wall);

    InteractionDynamics<fluid_dynamics::SurfaceTensionStress>
        liquid_surface_tension_stress(liquid_contact_bubble, StdVec<Real>{surface_tension});
    InteractionDynamics<fluid_dynamics::SurfaceTensionStress>
        bubble_surface_tension_stress(bubble_contact_liquid, StdVec<Real>{surface_tension/10});
    InteractionWithUpdate<fluid_dynamics::SurfaceStressForceComplex>
        liquid_surface_tension_force(liquid_inner, liquid_contact_bubble);
    InteractionWithUpdate<fluid_dynamics::SurfaceStressForceComplex>
        bubble_surface_tension_force(bubble_inner, bubble_contact_liquid);
    InteractionWithUpdate<fluid_dynamics::InterfaceSharpnessForceComplex>
        liquid_interface_sharpness_force(liquid_inner, liquid_contact_bubble);
    InteractionWithUpdate<fluid_dynamics::InterfaceSharpnessForceComplex>
        bubble_interface_sharpness_force(bubble_inner, bubble_contact_liquid);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep>
        liquid_advection_time_step(liquid_body, U_f);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep>
        bubble_advection_time_step(bubble_body, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> liquid_acoustic_time_step(liquid_body);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> bubble_acoustic_time_step(bubble_body);

    //InteractionWithUpdate<fluid_dynamics::TargetFluidParticles>
    //    liquid_target_fluid_particles(liquid_contact_wall);
    //InteractionWithUpdate<fluid_dynamics::TargetFluidParticles>
    //    bubble_target_fluid_particles(bubble_contact_wall);
    //SimpleDynamics<solid_dynamics::FirstLayerFromFluids>
    //    target_wall_solid_particles(wall_boundary, liquid_body, bubble_body);

    ParticleSorting liquid_particle_sorting(liquid_body);
    ParticleSorting bubble_particle_sorting(bubble_body);

    //----------------------------------------------------------------------
    // I/O operations and observations.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(sph_system);
    write_states.addToWrite<Real>(liquid_body, "Pressure");
    //write_states.addToWrite<int>(liquid_body, "FirstLayerIndicator");
    //write_states.addToWrite<int>(liquid_body, "SecondLayerIndicator");
    write_states.addToWrite<Vecd>(liquid_body, "InterfaceSharpnessForce");

    write_states.addToWrite<Real>(bubble_body, "Pressure");
    //write_states.addToWrite<int>(bubble_body, "FirstLayerIndicator");
    //write_states.addToWrite<int>(bubble_body, "SecondLayerIndicator");
    write_states.addToWrite<Vecd>(bubble_body, "InterfaceSharpnessForce");

    write_states.addToWrite<Vecd>(wall_boundary, "NormalDirection");

    ObservedQuantityRecording<Vecd> write_recorded_bubble_velocity("Velocity", observer_contact);
    ReducedQuantityRecording<TotalKineticEnergy> write_liquid_kinetic_energy(liquid_body);
    ReducedQuantityRecording<TotalKineticEnergy> write_bubble_kinetic_energy(bubble_body);

    //----------------------------------------------------------------------
    // Prepare the simulation.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();

    liquid_normal_direction.exec();
    bubble_normal_direction.exec();
    wall_normal_direction.exec();
    no_slip_wall_normal_direction.exec();
    liquid_gravity.exec();
    bubble_gravity.exec();

    //----------------------------------------------------------------------
    // Time-stepping control.
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int number_of_iterations = 0;
    int screen_output_interval = 100;
    Real End_Time = 5.0;
    Real output_interval = 0.02;

    //----------------------------------------------------------------------
    // Statistics for CPU time.
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TickCount::interval_t interval;

    //----------------------------------------------------------------------
    // First output before the main loop.
    //----------------------------------------------------------------------
    write_states.writeToFile();
    write_recorded_bubble_velocity.writeToFile(number_of_iterations);
    write_liquid_kinetic_energy.writeToFile(number_of_iterations);
    write_bubble_kinetic_energy.writeToFile(number_of_iterations);

    //----------------------------------------------------------------------
    // Main loop starts here.
    //----------------------------------------------------------------------
    while (physical_time < End_Time)
    {
        Real integration_time = 0.0;
        while (integration_time < output_interval)
        {
            Real Dt_liquid = liquid_advection_time_step.exec();
            Real Dt_bubble = bubble_advection_time_step.exec();
            Real Dt = SMIN(Dt_liquid, Dt_bubble);

            liquid_update_density_by_summation.exec();
            bubble_update_density_by_summation.exec();

            liquid_kernel_correction_complex.exec();
            bubble_kernel_correction_complex.exec();

            liquid_viscous_force.exec();
            bubble_viscous_force.exec();
            liquid_transport_correction.exec();
            bubble_transport_correction.exec();

            liquid_surface_tension_stress.exec();
            bubble_surface_tension_stress.exec();
            liquid_surface_tension_force.exec();
            bubble_surface_tension_force.exec();
            liquid_interface_sharpness_force.exec();
            bubble_interface_sharpness_force.exec();

            size_t inner_ite_dt = 0;
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                Real dt_liquid = liquid_acoustic_time_step.exec();
                Real dt_bubble = bubble_acoustic_time_step.exec();
                Real dt = SMIN(SMIN(dt_liquid, dt_bubble), Dt - relaxation_time);

                liquid_pressure_relaxation.exec(dt);
                bubble_pressure_relaxation.exec(dt);
                liquid_density_relaxation.exec(dt);
                bubble_density_relaxation.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
                inner_ite_dt++;
            }

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9)
                          << "N=" << number_of_iterations
                          << "\tTime=" << physical_time
                          << "\tDt=" << Dt
                          << "\tDt/dt=" << inner_ite_dt << "\n";
            }
            number_of_iterations++;

            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                liquid_particle_sorting.exec();
                bubble_particle_sorting.exec();
            }

            liquid_body.updateCellLinkedList();
            bubble_body.updateCellLinkedList();
            liquid_complex.updateConfiguration();
            bubble_complex.updateConfiguration();

            //liquid_target_fluid_particles.exec();
            //bubble_target_fluid_particles.exec();
            //target_wall_solid_particles.exec();
            observer_contact.updateConfiguration();
        }

        TickCount t2 = TickCount::now();
        write_states.writeToFile();
        write_recorded_bubble_velocity.writeToFile(number_of_iterations);
        write_liquid_kinetic_energy.writeToFile(number_of_iterations);
        write_bubble_kinetic_energy.writeToFile(number_of_iterations);
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }

    TickCount t4 = TickCount::now();
    TickCount::interval_t tt;
    tt = t4 - t1 - interval;

    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
    std::cout << "Total physical time for computation: " << physical_time << " seconds." << std::endl;

    return 0;
}