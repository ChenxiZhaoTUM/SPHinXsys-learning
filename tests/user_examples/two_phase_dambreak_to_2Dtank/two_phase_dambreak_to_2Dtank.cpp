/**
 * @file 	two_phase_dambreak.cpp
 * @brief 	2D two-phase dambreak flow.
 * @details This is the one of the basic test cases, also the first case for
 * 			understanding SPH method for multi-phase simulation.
 * @author 	Chi Zhang and Xiangyu Hu
 */
#include "two_phase_dambreak_to_2Dtank.h"
#include "sphinxsys.h"
using namespace SPH;

int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, particle_spacing_ref);
 //   sph_system.setRunParticleRelaxation(false);
	///** Tag for computation start with relaxed body fitted particles distribution.  */
	//sph_system.setReloadParticles(true);
	///** Tag for computation from restart files. 0: start with initial condition.    */
	//sph_system.setRestartStep(0);
	/** Handle command line arguments. */
#ifdef BOOST_AVAILABLE
	sph_system.handleCommandlineOptions(ac, av);
#endif
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    water_block.generateParticles<ParticleGeneratorLattice>();
    water_block.addBodyStateForRecording<Real>("Pressure");
	water_block.addBodyStateForRecording<Vecd>("Acceleration");

    FluidBody air_block(sph_system, makeShared<AirBlock>("AirBody"));
    air_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_a, c_f, mu_a);
    air_block.generateParticles<ParticleGeneratorLattice>();
    air_block.addBodyStateForRecording<Real>("Pressure");
	air_block.addBodyStateForRecording<Vecd>("Acceleration");

    SolidBody tank(sph_system, makeShared<WallBoundary>("Tank"));
    tank.defineParticlesAndMaterial<ElasticSolidParticles, SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
    tank.generateParticles<ParticleGeneratorLattice>();
    /*if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
	{
		tank.generateParticles<ParticleGeneratorReload>(io_environment, tank.getName());
	}
	else
	{
		tank.defineBodyLevelSetShape()->writeLevelSet(io_environment);
		tank.generateParticles<ParticleGeneratorLattice>();
	}*/
    tank.addBodyStateForRecording<Vecd>("NormalDirection");
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation tank_inner(tank);

    ContactRelation water_tank_contact(water_block, { &tank });
	ContactRelation air_tank_contact(air_block, { &tank });
	ContactRelation tank_contacts(tank, RealBodyVector{ &water_block, &air_block });

    ComplexRelation water_air_complex(water_block, {&air_block});
    ComplexRelation air_water_complex(air_block, {&water_block});
    ComplexRelation water_air_tank_complex(water_block, RealBodyVector{ &air_block, &tank });
	ComplexRelation water_tank_complex_for_damping(water_block, { &tank });
    //--------------------------------------------------------------------------------
	//	Run particle relaxation for body-fitted distribution if chosen.
	//--------------------------------------------------------------------------------
	//if (sph_system.RunParticleRelaxation())
	//{
	//	//----------------------------------------------------------------------------
	//	//	Methods used for particle relaxation.
	//	//----------------------------------------------------------------------------
	//	/** Random reset the insert body particle position. */
	//	SimpleDynamics<RandomizeParticlePosition> random_tank_particles(tank);
	//	/** Write the body state to Vtp file. */
	//	BodyStatesRecordingToVtp write_tank_to_vtp(io_environment, { &tank });
	//	/** Write the particle reload files. */
	//	ReloadParticleIO write_tank_particle_reload_files(io_environment, tank, "Tank");
	//	/** A Physics relaxation step. */
	//	relax_dynamics::RelaxationStepInner tank_relaxation_step_inner(tank_inner);

	//	//----------------------------------------------------------------------------
	//	//	Particle relaxation starts here.
	//	//----------------------------------------------------------------------------
	//	random_tank_particles.exec(0.25);
	//	tank_relaxation_step_inner.SurfaceBounding().exec();
	//	write_tank_to_vtp.writeToFile(0);

	//	int ite_p = 0;
	//	while (ite_p < 1000)
	//	{
	//		tank_relaxation_step_inner.exec();
	//		ite_p += 1;
	//		if (ite_p % 200 == 0)
	//		{
	//			std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the tank N = " << ite_p << "\n";
	//			write_tank_to_vtp.writeToFile(ite_p);
	//		}
	//	}
	//	std::cout << "The physics relaxation process of finish !" << std::endl;

	//	/** Output results. */
	//	write_tank_particle_reload_files.writeToFile(0);

	//	return 0;
	//}

    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    /** Initialize particle acceleration. */
    
	SimpleDynamics<TimeStepInitialization> initialize_a_water_step(water_block, makeShared<VariableGravitySecond>());
	SimpleDynamics<TimeStepInitialization> initialize_a_air_step(air_block, makeShared<VariableGravitySecond>());
    /** Evaluation of density by summation approach. */
    InteractionWithUpdate<fluid_dynamics::DensitySummationFreeSurfaceComplex>
        update_water_density_by_summation(water_tank_contact, water_air_complex.getInnerRelation());
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplex>
        update_air_density_by_summation(air_tank_contact, air_water_complex);
    InteractionDynamics<fluid_dynamics::TransportVelocityCorrectionComplex<AllParticles>>
        air_transport_correction(air_tank_contact, air_water_complex);
    InteractionDynamics<fluid_dynamics::ViscousAccelerationMultiPhaseWithWall>
		water_viscous_acceleration(water_tank_contact, water_air_complex);
	InteractionDynamics<fluid_dynamics::ViscousAccelerationMultiPhaseWithWall>
		air_viscous_acceleration(air_tank_contact, air_water_complex);
	/** Computing vorticity in the flow. */
	InteractionDynamics<fluid_dynamics::VorticityInner> compute_vorticity(water_air_tank_complex.getInnerRelation());
    /** Time step size without considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_water_advection_time_step_size(water_block, U_ref);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_air_advection_time_step_size(air_block, U_ref);
    /** Time step size with considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_water_time_step_size(water_block);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_air_time_step_size(air_block);
    /** Pressure relaxation for water by using position verlet time stepping. */
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfRiemannWithWall>
        water_pressure_relaxation(water_tank_contact, water_air_complex);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfRiemannWithWall>
        water_density_relaxation(water_tank_contact, water_air_complex);
    /*Dynamics1Level<fluid_dynamics::Integration1stHalfRiemannWithWall>
		water_pressure_relaxation(water_tank_contact, water_air_complex.getInnerRelation());
	Dynamics1Level<fluid_dynamics::Integration2ndHalfRiemannWithWall>
		water_density_relaxation(water_tank_contact, water_air_complex.getInnerRelation());*/
    /** Extend Pressure relaxation is used for air. */
    Dynamics1Level<fluid_dynamics::ExtendMultiPhaseIntegration1stHalfRiemannWithWall>
        air_pressure_relaxation(air_tank_contact, air_water_complex, 2.0);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfRiemannWithWall>
        air_density_relaxation(air_tank_contact, air_water_complex);

    DampingWithRandomChoice<InteractionSplit<DampingPairwiseWithWall<Vecd, DampingPairwiseInner>>>
        fluid_damping(0.2, water_tank_complex_for_damping, "Velocity", viscous_dynamics);

    //--------------------------------------------------------------------------------
	//	Algorithms of FSI.
	//--------------------------------------------------------------------------------
	/** Force exerted on elastic body due to fluid pressure and viscosity. */
	InteractionDynamics<solid_dynamics::ViscousForceFromFluid> viscous_force_on_tack(tank_contacts);
	InteractionDynamics<solid_dynamics::AllForceAccelerationFromFluid>
		fluid_force_on_tank_update(tank_contacts, viscous_force_on_tack);
	/** Average velocity of the elastic body. */
	solid_dynamics::AverageVelocityAndAcceleration tank_average_velocity_and_acceleration(tank);

    //----------------------------------------------------------------------
	//	Algorithms of solid dynamics.
	//----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> inner_normal_direction(tank);
    InteractionWithUpdate<KernelCorrectionMatrixInner> tank_corrected_configuration(tank_inner);
	/** Time step size of elastic body. */
	ReduceDynamics<solid_dynamics::AcousticTimeStepSize> tank_acoustic_time_step(tank);
	/** Stress relaxation for the elastic body. */
	Dynamics1Level<solid_dynamics::Integration1stHalfPK2> tank_stress_relaxation_1st_half(tank_inner);
	Dynamics1Level<solid_dynamics::Integration2ndHalf> tank_stress_relaxation_2nd_half(tank_inner);
	DampingWithRandomChoice<InteractionSplit<DampingBySplittingInner<Vecd>>>
        tank_damping(0.2, tank_inner, "Velocity", physical_viscosity);

    /** Exert constrain on tank. */
	SimpleDynamics<solid_dynamics::ConstrainSolidBodyMassCenter> constrain_mass_center_1(tank);
	ReduceDynamics<QuantitySummation<Real>> compute_total_mass_(tank, "MassiveMeasure");
	ReduceDynamics<QuantityMassPosition> compute_mass_position_(tank);
	Vecd mass_center = compute_mass_position_.exec() / compute_total_mass_.exec();
	Real moment_of_inertia = Real(0.0);
	ReduceDynamics<QuantityMomentOfInertia<Real>> compute_moment_of_inertia(tank, mass_center);
	moment_of_inertia = compute_moment_of_inertia.exec();;
	SimpleDynamics<Constrain2DSolidBodyRotation> constrain_rotation(tank, mass_center, moment_of_inertia);

	/** Update normal direction. */
	SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> tank_update_normal_direction(tank);

    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    /** Output the body states. */
    BodyStatesRecordingToVtp body_states_recording(io_environment, sph_system.real_bodies_);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    inner_normal_direction.exec();
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    /** Output the start states of bodies. */
    body_states_recording.writeToFile(0);
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = 0;
    int screen_output_interval = 100;
    Real end_time = 20.0;
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
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < output_interval)
        {
            /** Acceleration due to viscous force and gravity. */
            time_instance = TickCount::now();
            initialize_a_water_step.exec();
            initialize_a_air_step.exec();

            Real Dt_f = get_water_advection_time_step_size.exec();
            Real Dt_a = get_air_advection_time_step_size.exec();
            Real Dt = SMIN(Dt_f, Dt_a);

            update_water_density_by_summation.exec();
            update_air_density_by_summation.exec();
            air_transport_correction.exec();
            water_viscous_acceleration.exec();
			air_viscous_acceleration.exec();

            /** FSI for viscous force. */
			viscous_force_on_tack.exec();

			/** Update normal direction on elastic body. */
			tank_update_normal_direction.exec();

            interval_computing_time_step += TickCount::now() - time_instance;

            /** Dynamics including pressure relaxation. */
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                Real dt_f = get_water_time_step_size.exec();
                Real dt_a = get_air_time_step_size.exec();
                dt = SMIN(SMIN(dt_f, dt_a), Dt);

                if (GlobalStaticVariables::physical_time_ < 1.0)
                {
                    fluid_damping.exec(dt);
                }

                water_pressure_relaxation.exec(dt);
                air_pressure_relaxation.exec(dt);
                /** FSI for pressure force. */
				fluid_force_on_tank_update.exec();
                /** Fluid density relaxation. */
                water_density_relaxation.exec(dt);
                air_density_relaxation.exec(dt);

                Real dt_s_sum = 0.0;
                tank_average_velocity_and_acceleration.initialize_displacement_.exec();

                while (dt_s_sum < dt)
                {
                    Real dt_s = SMIN(tank_acoustic_time_step.exec(), dt - dt_s_sum);
                    tank_stress_relaxation_1st_half.exec(dt_s);
                    
                    constrain_rotation.exec();
					constrain_mass_center_1.exec();

                    if (GlobalStaticVariables::physical_time_ < 1.0)
                    {
                        tank_damping.exec(dt_s);
                        constrain_rotation.exec();
					    constrain_mass_center_1.exec();
                    }

                    tank_stress_relaxation_2nd_half.exec(dt_s);
                    dt_s_sum += dt_s;
                }
                tank_average_velocity_and_acceleration.update_averages_.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
            }
            interval_computing_pressure_relaxation += TickCount::now() - time_instance;

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << GlobalStaticVariables::physical_time_
                          << "	Dt = " << Dt << "	dt = " << dt << "\n";
            }
            number_of_iterations++;

            /** Update cell linked list and configuration. */
            time_instance = TickCount::now();

            water_block.updateCellLinkedListWithParticleSort(100);
            water_air_complex.updateConfiguration();
            water_tank_contact.updateConfiguration();
            water_tank_complex_for_damping.updateConfiguration();

            air_block.updateCellLinkedListWithParticleSort(100);
            air_water_complex.updateConfiguration();
            air_tank_contact.updateConfiguration();

            tank.updateCellLinkedList();
			tank_contacts.updateConfiguration();

            interval_updating_configuration += TickCount::now() - time_instance;
        }

        TickCount t2 = TickCount::now();
        compute_vorticity.exec();

        body_states_recording.writeToFile();
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
