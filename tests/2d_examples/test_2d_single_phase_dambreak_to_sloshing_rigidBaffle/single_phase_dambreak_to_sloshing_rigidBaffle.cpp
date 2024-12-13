/**
 * @file 	two_phase_dambreak.cpp
 * @brief 	2D two-phase dambreak flow.
 * @details This is the one of the basic test cases, also the first case for
 * 			understanding SPH method for multi-phase simulation.
 * @author 	Chi Zhang and Xiangyu Hu
 */
#include "single_phase_dambreak_to_sloshing_rigidBaffle.h"
#include "sphinxsys.h"
using namespace SPH;
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up an SPHSystem and IO environment.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, particle_spacing_ref);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating bodies with corresponding materials and particles.
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_water);
    water_block.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineAdaptationRatios(1.3, sph_system.resolution_ref_ / resolution_ref_solid);
    wall_boundary.defineMaterial<Solid>();
    wall_boundary.generateParticles<BaseParticles, Lattice>();

    ObserverBody water_pressure_observer(sph_system, "WaterPressureObserver");
    water_pressure_observer.generateParticles<ObserverParticles>(water_pressure_observation_location);
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //----------------------------------------------------------------------
    InnerRelation water_block_inner(water_block);
    ContactRelation water_wall_contact(water_block, {&wall_boundary});
    ContactRelation water_pressure_observer_contact(water_pressure_observer, {&water_block});
    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    // which is only used for update configuration.
    //----------------------------------------------------------------------
    ComplexRelation water_wall_complex(water_block_inner, water_wall_contact);
    //----------------------------------------------------------------------
    // Define the numerical methods used in the simulation.
    // Note that there may be data dependence on the sequence of constructions.
    // Generally, the geometric models or simple objects without data dependencies,
    // such as gravity, should be initiated first.
    // Then the major physical particle dynamics model should be introduced.
    // Finally, the auxiliary models such as time step estimator, initial condition,
    // boundary condition and other constraints should be defined.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);

    SimpleDynamics<InitialDensity> initial_density_condition(water_block);
    VariableGravity variable_gravity;
    SimpleDynamics<GravityForce<VariableGravity>> initialize_a_water_step(water_block, variable_gravity);

    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> fluid_pressure_relaxation(water_block_inner, water_wall_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> fluid_density_relaxation(water_block_inner, water_wall_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplexFreeSurface> fluid_density_by_summation(water_block_inner, water_wall_contact);

    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> 
		water_viscous_acceleration(water_block_inner, water_wall_contact);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> fluid_advection_time_step(water_block, U_ref);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> fluid_acoustic_time_step(water_block);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> water_block_kernel_correction_matrix(water_block_inner, water_wall_contact);
    InteractionDynamics<KernelGradientCorrectionInner> water_kernel_gradient_update(water_block_inner);
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseInner<Vec2d, FixedDampingRate>>>
        fluid_damping(0.2, water_block_inner, "Velocity", mu_water);

    /*-------------------------------------------------------------------------------*/
    /*--------------------------FREE SURFACE IDENTIFICATION--------------------------*/
    /*-------------------------------------------------------------------------------*/
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex>
        free_stream_surface_indicator(water_block_inner, water_wall_contact);
    /** Impose transport velocity formulation. */
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>>
        transport_velocity_correction(water_block_inner, water_wall_contact);
    //----------------------------------------------------------------------
    //	Recording.
    //----------------------------------------------------------------------
	BodyRegionByCell probe_E3(water_block, makeShared<MultiPolygonShape>(createWaveProbeShapeE3(), "PorbeE3"));
	BodyRegionByCell probe_E4(water_block, makeShared<MultiPolygonShape>(createWaveProbeShapeE4(), "PorbeE4"));
	ReducedQuantityRecording<UpperFrontInAxisDirection<BodyPartByCell>>
		probe_surface_height_3(probe_E3, "FreeSurfaceHeight");
	ReducedQuantityRecording<UpperFrontInAxisDirection<BodyPartByCell>>
		probe_surface_height_4(probe_E4, "FreeSurfaceHeight");
    ObservedQuantityRecording<Real> write_recorded_water_pressure("Pressure", water_pressure_observer_contact);
    ObservedQuantityRecording<Real> write_recorded_water_density("Density", water_pressure_observer_contact);
    //----------------------------------------------------------------------
    //	Define the configuration related particles dynamics.
    //----------------------------------------------------------------------
    ParticleSorting particle_sorting(water_block);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Vecd>(wall_boundary, "NormalDirection");
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    initial_density_condition.exec();
    wall_boundary_normal_direction.exec();
    water_block_kernel_correction_matrix.exec();
    water_kernel_gradient_update.exec();
    //----------------------------------------------------------------------
    //	Load restart file if necessary.
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    int restart_output_interval = screen_output_interval * 10;
    Real end_time = 16.0;
    Real output_interval = 0.05;
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    TimeInterval interval_computing_time_step;
    TimeInterval interval_computing_fluid_pressure_relaxation;
    TimeInterval interval_updating_configuration;
    TickCount time_instance;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    body_states_recording.writeToFile();
    probe_surface_height_3.writeToFile(0);
	probe_surface_height_4.writeToFile(0);
	write_recorded_water_pressure.writeToFile(0);
	write_recorded_water_density.writeToFile(0);
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (physical_time < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < output_interval)
        {
            /** outer loop for dual-time criteria time-stepping. */
            time_instance = TickCount::now();
            initialize_a_water_step.exec();
            //free_stream_surface_indicator.exec();

            Real advection_dt = fluid_advection_time_step.exec();
            fluid_density_by_summation.exec();
            water_viscous_acceleration.exec();
            //transport_velocity_correction.exec();

            interval_computing_time_step += TickCount::now() - time_instance;

            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            Real acoustic_dt = 0.0;
            while (relaxation_time < advection_dt)
            {
                /** inner loop for dual-time criteria time-stepping.  */
                acoustic_dt = fluid_acoustic_time_step.exec();

                if (physical_time < 1.0)
				{
					fluid_damping.exec(acoustic_dt);
				}

                fluid_pressure_relaxation.exec(acoustic_dt);
                fluid_density_relaxation.exec(acoustic_dt);
                relaxation_time += acoustic_dt;
                integration_time += acoustic_dt;
                physical_time += acoustic_dt;
            }
            interval_computing_fluid_pressure_relaxation += TickCount::now() - time_instance;

            /** screen output, write body observables and restart files  */
            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << physical_time
                          << "	advection_dt = " << advection_dt << "	acoustic_dt = " << acoustic_dt << "\n";
            }
            number_of_iterations++;

            /** Update cell linked list and configuration. */
            time_instance = TickCount::now();
            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                particle_sorting.exec();
            }
            water_block.updateCellLinkedList();
            water_wall_complex.updateConfiguration();
            water_pressure_observer_contact.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;
        }

        body_states_recording.writeToFile();
        probe_surface_height_3.writeToFile();
		probe_surface_height_4.writeToFile();

		write_recorded_water_pressure.writeToFile();
		write_recorded_water_density.writeToFile();

        TickCount t2 = TickCount::now();
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds()
              << " seconds." << std::endl;
    std::cout << std::fixed << std::setprecision(9) << "interval_computing_time_step ="
              << interval_computing_time_step.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9) << "interval_computing_fluid_pressure_relaxation = "
              << interval_computing_fluid_pressure_relaxation.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9) << "interval_updating_configuration = "
              << interval_updating_configuration.seconds() << "\n";

    return 0;
};