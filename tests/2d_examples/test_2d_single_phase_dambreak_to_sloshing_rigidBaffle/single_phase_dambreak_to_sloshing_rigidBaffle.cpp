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
    //	Build up the environment of a SPHSystem.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, particle_spacing_ref);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
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

    //ObserverBody tank_pressure_observer(sph_system, "TankPressureObserver");
    //tank_pressure_observer.generateParticles<ObserverParticles>(tank_pressure_observation_location);
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation water_inner(water_block);
    ContactRelation water_wall_contact(water_block, {&wall_boundary});
    //ContactRelation tank_contacts(wall_boundary, {&water_block, &air_block});

    ContactRelation water_pressure_observer_contact(water_pressure_observer, RealBodyVector{&water_block});
    //ContactRelation tank_pressure_observer_contact(tank_pressure_observer, RealBodyVector{&wall_boundary});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    /** Initialize particle acceleration. */
    SimpleDynamics<NormalDirectionFromSubShapeAndOp> inner_normal_direction(wall_boundary, "InnerWall");
    //InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> surface_particle_indicator(water_inner, water_wall_contact);

    SimpleDynamics<InitialDensity> initial_density_condition(water_block);
    VariableGravity variable_gravity;
    SimpleDynamics<GravityForce<VariableGravity>> initialize_a_water_step(water_block, variable_gravity);

    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> water_pressure_relaxation(water_inner, water_wall_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> water_density_relaxation(water_inner, water_wall_contact);

    InteractionWithUpdate<fluid_dynamics::DensitySummationComplexFreeSurface>
        update_water_density_by_summation(water_inner, water_wall_contact);
    
    /*InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>>
        water_transport_correction(water_inner, water_wall_contact);*/

    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> 
		water_viscous_acceleration(water_inner, water_wall_contact);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_water_advection_time_step_size(water_block, U_ref);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_water_time_step_size(water_block);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> water_block_kernel_correction_matrix(water_inner, water_wall_contact);
    InteractionDynamics<KernelGradientCorrectionInner> water_kernel_gradient_update(water_inner);
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseInner<Vec2d, FixedDampingRate>>>
        fluid_damping(0.2, water_inner, "Velocity", mu_water);

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
    //ObservedQuantityRecording<Vecd> write_recorded_tank_pressure("PressureForceFromFluid", tank_pressure_observer_contact);
    //----------------------------------------------------------------------
    //	Define the configuration related particles dynamics.
    //----------------------------------------------------------------------
    ParticleSorting water_particle_sorting(water_block);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Vecd>(wall_boundary, "NormalDirection"); // output for debug
    body_states_recording.addToWrite<Real>(water_block, "Pressure");

    RegressionTestDynamicTimeWarping<ReducedQuantityRecording<TotalMechanicalEnergy>>
        write_water_mechanical_energy(water_block, variable_gravity);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    initial_density_condition.exec();
    inner_normal_direction.exec();
    water_block_kernel_correction_matrix.exec();
    water_kernel_gradient_update.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    size_t number_of_iterations = 0;
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 15.0;
    Real output_interval = 0.05;
    Real dt = 0.0; /**< Default acoustic time step sizes. */
    /** statistics for computing CPU time. */
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    TimeInterval interval_computing_time_step;
    TimeInterval interval_computing_pressure_relaxation;
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
            /** Force Prior due to viscous force and gravity. */
            time_instance = TickCount::now();

            initialize_a_water_step.exec();
            //surface_particle_indicator.exec();

            Real Dt = get_water_advection_time_step_size.exec();

            update_water_density_by_summation.exec();

            water_viscous_acceleration.exec();
            //water_transport_correction.exec();

            interval_computing_time_step += TickCount::now() - time_instance;

            /** Dynamics including pressure relaxation. */
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                Real dt_f = get_water_time_step_size.exec();
                dt = SMIN(dt_f, Dt);

                if (physical_time < 0.1)
				{
					fluid_damping.exec(dt);
				}

                water_pressure_relaxation.exec(dt);
                water_density_relaxation.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
            }
            interval_computing_pressure_relaxation += TickCount::now() - time_instance;

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << physical_time
                          << "	Dt = " << Dt << "	dt = " << dt << "\n";

                if (number_of_iterations != 0 && number_of_iterations % observation_sample_interval == 0)
                {
                    write_water_mechanical_energy.writeToFile(number_of_iterations);
                }
            }
            number_of_iterations++;

            /** Update cell linked list and configuration. */
            time_instance = TickCount::now();
            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                water_particle_sorting.exec();
            }
            water_block.updateCellLinkedList();
            water_wall_contact.updateConfiguration();

            water_pressure_observer_contact.updateConfiguration();

            interval_updating_configuration += TickCount::now() - time_instance;
        }

        TickCount t2 = TickCount::now();
        body_states_recording.writeToFile();
		probe_surface_height_3.writeToFile();
		probe_surface_height_4.writeToFile();

		write_recorded_water_pressure.writeToFile();
		write_recorded_water_density.writeToFile();

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
    std::cout << std::fixed << std::setprecision(9) << "interval_computing_pressure_relaxation = "
              << interval_computing_pressure_relaxation.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9) << "interval_updating_configuration = "
              << interval_updating_configuration.seconds() << "\n";

    return 0;
}
