/**
* @file 	elastic_tank_basic_case.cpp
* @brief 	Sloshing in marine LNG fuel tank with elastic material under roll excitation
* @details
* @author
*/
#include "elastic_tank_basic_case.h"

using namespace SPH;  /** Namespace cite here. */
//------------------------------------------------------------------------------------
//	Main program starts here.
//------------------------------------------------------------------------------------
int main(int ac, char* av[])
{
	//--------------------------------------------------------------------------------
	//	Build up the environment of a SPHSystem.
	//--------------------------------------------------------------------------------
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	/** Tag for run particle relaxation for the initial body fitted distribution.   */
	sph_system.setRunParticleRelaxation(false);
	/** Tag for computation start with relaxed body fitted particles distribution.  */
	sph_system.setReloadParticles(true);
	/** Tag for computation from restart files. 0: start with initial condition.    */
	sph_system.setRestartStep(0);
	/** Handle command line arguments. */
#ifdef BOOST_AVAILABLE
	sph_system.handleCommandlineOptions(ac, av);
#endif
	IOEnvironment io_environment(sph_system);

	//--------------------------------------------------------------------------------
	//	Creating body, materials and particles.
	//--------------------------------------------------------------------------------
	SolidBody tank(sph_system, makeShared<Tank>("Tank"));
	tank.defineAdaptation<SPHAdaptation>(1.15, 2.0);  // can not use damping
	tank.defineParticlesAndMaterial<ElasticSolidParticles, SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
	if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
	{
		tank.generateParticles<ParticleGeneratorReload>(io_environment, tank.getName());
	}
	else
	{
		tank.defineBodyLevelSetShape()->writeLevelSet(io_environment);
		tank.generateParticles<ParticleGeneratorLattice>();
	}
	tank.addBodyStateForRecording<Vecd>("NormalDirection");

	FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
	water_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
	water_block.generateParticles<ParticleGeneratorLattice>();
	water_block.addBodyStateForRecording<Real>("Pressure");

	FluidBody air_block(sph_system, makeShared<AirBlock>("AirBody"));
	air_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_a, c_f, mu_a);
	air_block.generateParticles<ParticleGeneratorLattice>();
	air_block.addBodyStateForRecording<Real>("Pressure");

	/*ObserverBody tank_observer(sph_system, "TankObserver");
	tank_observer.generateParticles<TankObserverParticleGenerator>();*/

	//--------------------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//--------------------------------------------------------------------------------
	InnerRelation tank_inner(tank);
	InnerRelation water_inner(water_block);

	ContactRelation water_block_contact(water_block, { &tank });
	ContactRelation air_block_contact(air_block, { &tank });
	ContactRelation tank_contacts(tank, RealBodyVector{ &water_block, &air_block });
	//ContactRelation tank_observer_contact(tank_observer, { &tank });

	ComplexRelation water_air_complex(water_block, { &air_block });
	ComplexRelation air_water_complex(air_block, { &water_block });
	ComplexRelation water_air_tank_complex(water_block, RealBodyVector{ &air_block, &tank });
	ComplexRelation water_tank_complex_for_damping(water_block, { &tank });
	 
	//--------------------------------------------------------------------------------
	//	Run particle relaxation for body-fitted distribution if chosen.
	//--------------------------------------------------------------------------------
	if (sph_system.RunParticleRelaxation())
	{
		//----------------------------------------------------------------------------
		//	Methods used for particle relaxation.
		//----------------------------------------------------------------------------
		/** Random reset the insert body particle position. */
		SimpleDynamics<RandomizeParticlePosition> random_tank_particles(tank);
		/** Write the body state to Vtp file. */
		BodyStatesRecordingToVtp write_tank_to_vtp(io_environment, { &tank });
		/** Write the particle reload files. */
		ReloadParticleIO write_tank_particle_reload_files(io_environment, tank, "Tank");
		/** A Physics relaxation step. */
		relax_dynamics::RelaxationStepInner tank_relaxation_step_inner(tank_inner);

		//----------------------------------------------------------------------------
		//	Particle relaxation starts here.
		//----------------------------------------------------------------------------
		random_tank_particles.exec(0.25);
		tank_relaxation_step_inner.SurfaceBounding().exec();
		write_tank_to_vtp.writeToFile(0);

		int ite_p = 0;
		while (ite_p < 1000)
		{
			tank_relaxation_step_inner.exec();
			ite_p += 1;
			if (ite_p % 200 == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the tank N = " << ite_p << "\n";
				write_tank_to_vtp.writeToFile(ite_p);
			}
		}
		std::cout << "The physics relaxation process of finish !" << std::endl;

		/** Output results. */
		write_tank_particle_reload_files.writeToFile(0);

		return 0;
	}

	//--------------------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//--------------------------------------------------------------------------------
	/*InteractionDynamics<InterpolatingAQuantity<Vecd>>
		interpolation_observer_position(tank_observer_contact, "Position", "Position");*/

	//--------------------------------------------------------------------------------
	//	Algorithms of fluid dynamics.
	//--------------------------------------------------------------------------------
	SimpleDynamics<InitialDensity> initial_density_condition(water_block);

	/** Time step initialization of fluid body. */
	/*SimpleDynamics<TimeStepInitializationForVariableGravity> initialize_a_water_step(water_block, makeShared<VariableGravity>());
	SimpleDynamics<TimeStepInitializationForVariableGravity> initialize_a_air_step(air_block, makeShared<VariableGravity>());*/

	SimpleDynamics<TimeStepInitialization> initialize_a_water_step(water_block, makeShared<VariableGravitySecond>());
	SimpleDynamics<TimeStepInitialization> initialize_a_air_step(air_block, makeShared<VariableGravitySecond>());

	InteractionWithUpdate<fluid_dynamics::DensitySummationFreeSurfaceComplex>
		water_density_by_summation(water_block_contact, water_air_complex.getInnerRelation());
	InteractionWithUpdate<fluid_dynamics::DensitySummationComplex>
		air_density_by_summation(air_block_contact, air_water_complex);
	InteractionDynamics<fluid_dynamics::TransportVelocityCorrectionComplex<AllParticles>>
		air_transport_correction(air_block_contact, air_water_complex);
	InteractionDynamics<fluid_dynamics::ViscousAccelerationMultiPhaseWithWall>
		water_viscous_acceleration(water_block_contact, water_air_complex);
	InteractionDynamics<fluid_dynamics::ViscousAccelerationMultiPhaseWithWall>
		air_viscous_acceleration(air_block_contact, air_water_complex);
	/** Computing vorticity in the flow. */
	InteractionDynamics<fluid_dynamics::VorticityInner> compute_vorticity(water_air_tank_complex.getInnerRelation());

	/** Time step size of fluid body. */
	ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> water_advection_time_step(water_block, U_max);
	ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> air_advection_time_step(air_block, U_max);
	ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> water_acoustic_time_step(water_block);
	ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> air_acoustic_time_step(air_block);

	/** Riemann slover for pressure and density relaxation. */
	Dynamics1Level<fluid_dynamics::Integration1stHalfRiemannWithWall>
		water_pressure_relaxation(water_block_contact, water_air_complex.getInnerRelation());
	Dynamics1Level<fluid_dynamics::Integration2ndHalfRiemannWithWall>
		water_density_relaxation(water_block_contact, water_air_complex.getInnerRelation());
	Dynamics1Level<fluid_dynamics::ExtendMultiPhaseIntegration1stHalfRiemannWithWall>
		air_pressure_relaxation(air_block_contact, air_water_complex, 2.0);
	Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfRiemannWithWall>
		air_density_relaxation(air_block_contact, air_water_complex);

	/*DampingWithRandomChoice<InteractionSplit<DampingPairwiseWithWall<Vecd, DampingPairwiseInner>>>
        fluid_damping(0.2, water_tank_complex_for_damping, "Velocity", viscous_dynamics);*/
	DampingWithRandomChoice<InteractionSplit<DampingPairwiseInner<Vec3d>>>
        fluid_damping(0.2, water_inner, "Velocity", viscous_dynamics);
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
	SimpleDynamics<NormalDirectionFromShapeAndOp> tank_normal_direction(tank, "InnerWall");
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
	Matd moment_of_inertia = Matd::Zero();
	for (int i = 0; i != Dimensions; ++i)
	{
		for (int j = 0; j != Dimensions; ++j)
		{
			ReduceDynamics<QuantityMomentOfInertia> compute_moment_of_inertia(tank, mass_center, i, j);
			moment_of_inertia(i, j) = compute_moment_of_inertia.exec();
		}
	}
	SimpleDynamics<Constrain3DSolidBodyRotation> constrain_rotation(tank, mass_center, moment_of_inertia);

	/** Update normal direction. */
	SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> tank_update_normal_direction(tank);

	//--------------------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//--------------------------------------------------------------------------------
	BodyRegionByCell probe_s1(water_block, makeShared<ProbeShape>("PorbeS1"));
	ReducedQuantityRecording<ReduceDynamics<fluid_dynamics::FreeSurfaceHeight>>
		probe_1(io_environment, probe_s1);

	BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
	/*ObservedQuantityRecording<Vecd> write_tank_move("Position", io_environment, tank_observer_contact);
	ObservedQuantityRecording<Vecd> write_tank_nom("NormalDirection", io_environment, tank_observer_contact);*/
	ReducedQuantityRecording<ReduceDynamics<solid_dynamics::TotalForceFromFluid>>
		write_viscous_force_on_tank(io_environment, viscous_force_on_tack, "TotalViscousForceOnTank");
	ReducedQuantityRecording<ReduceDynamics<solid_dynamics::TotalForceFromFluid>>
		write_total_force_on_tank(io_environment, fluid_force_on_tank_update, "TotalForceOnTank");

	//--------------------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//--------------------------------------------------------------------------------
	/** Initialize cell linked lists for all bodies. */
	sph_system.initializeSystemCellLinkedLists();
	/** Initialize configurations for all bodies. */
	sph_system.initializeSystemConfigurations();
	initial_density_condition.exec();
	/** Computing surface normal direction for the tank. */
	tank_corrected_configuration.exec();
	tank_normal_direction.exec();
	//--------------------------------------------------------------------------------
	//	Setup computing and initial conditions.
	//--------------------------------------------------------------------------------
	size_t number_of_iterations = sph_system.RestartStep();
	int screen_output_interval = 10;
	int restart_output_interval = screen_output_interval * 10;
	Real End_Time = 13;			                                      /** End time. */
	Real D_Time = 0.05;								/** time stamps for output. */
	Real Dt = 0.0;				   /** Default advection time step sizes for fluid. */
	Real dt = 0.0; 					/** Default acoustic time step sizes for fluid. */
	Real dt_a = 0.0;				  /** Default acoustic time step sizes for air. */

	//--------------------------------------------------------------------------------
	//	Statistics for CPU time.
	//--------------------------------------------------------------------------------
	TickCount t1 = TickCount::now();
	TickCount::interval_t interval;

	//--------------------------------------------------------------------------------
	//	First output before the main loop.
	//--------------------------------------------------------------------------------
	/** Computing linear reproducing configuration for the tank. */
	write_real_body_states.writeToFile(0);
	probe_1.writeToFile(0);
	//write_tank_move.writeToFile(0);
	//write_tank_nom.writeToFile(0);

	//--------------------------------------------------------------------------------
	//	Main loop starts here.
	//--------------------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real integration_time = 0.0;

		/** Integrate time (loop) until the next output time. */
		while (integration_time < D_Time)
		{
			/** Outer loop for dual-time criteria time-stepping. */
			initialize_a_water_step.exec();
			initialize_a_air_step.exec();

			Real Dt_f = water_advection_time_step.exec();
			Real Dt_a = air_advection_time_step.exec();
			Dt = SMIN(Dt_f, Dt_a);

			water_density_by_summation.exec();
			air_density_by_summation.exec();
			air_transport_correction.exec();
			water_viscous_acceleration.exec();
			air_viscous_acceleration.exec();

			/** FSI for viscous force. */
			viscous_force_on_tack.exec();

			/** Update normal direction on elastic body. */
			tank_update_normal_direction.exec();

			size_t inner_ite_dt = 0;
			size_t inner_ite_dt_s = 0;
			Real relaxation_time = 0.0;

			while (relaxation_time < Dt)
			{
				Real dt_f = water_acoustic_time_step.exec();
				dt_a = air_acoustic_time_step.exec();
				dt = SMIN(SMIN(dt_f, dt_a), Dt);

				if (GlobalStaticVariables::physical_time_ < 1.0)
                {
                    fluid_damping.exec(dt);
                }

				/** Fluid pressure relaxation. */
				water_pressure_relaxation.exec(dt);
				air_pressure_relaxation.exec(dt);
				/** FSI for pressure force. */
				fluid_force_on_tank_update.exec();
				/** Fluid density relaxation. */
				water_density_relaxation.exec(dt);
				air_density_relaxation.exec(dt);

				/*interpolation_observer_position.exec();*/

				/** Solid dynamics. */
				inner_ite_dt_s = 0;
				Real dt_s_sum = 0.0;
				tank_average_velocity_and_acceleration.initialize_displacement_.exec();
				while (dt_s_sum < dt)
				{
					Real dt_s = SMIN(tank_acoustic_time_step.exec(), dt - dt_s_sum);
					tank_stress_relaxation_1st_half.exec(dt_s);

					constrain_rotation.exec(dt_s);
					constrain_mass_center_1.exec(dt_s);

					if (GlobalStaticVariables::physical_time_ < 1.0)
					{
						tank_damping.exec(dt_s);
						constrain_rotation.exec();
						constrain_mass_center_1.exec();
					}

                    /*tank_damping.exec(dt_s);
                    constrain_rotation.exec();
					constrain_mass_center_1.exec();*/

					tank_stress_relaxation_2nd_half.exec(dt_s);
					dt_s_sum += dt_s;
					inner_ite_dt_s++;
				}
				tank_average_velocity_and_acceleration.update_averages_.exec(dt);

				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
				inner_ite_dt++;
			}

			/** Screen output, write body reduced values and restart files. */
			if (number_of_iterations % screen_output_interval == 0)
			{
				std::cout << std::fixed << std::setprecision(9)
					<< "N=" << number_of_iterations << "	Time = "
					<< GlobalStaticVariables::physical_time_
					<< "	Dt = " << Dt << "	Dt / dt = " << inner_ite_dt << "	dt / dt_s = " << inner_ite_dt_s << "\n";
			}
			number_of_iterations++;

			/** Update cell linked list and configuration. */
			water_block.updateCellLinkedListWithParticleSort(100);
			water_block_contact.updateConfiguration();
			water_air_complex.updateConfiguration();
			water_tank_complex_for_damping.updateConfiguration();

			air_block.updateCellLinkedListWithParticleSort(100);
			air_block_contact.updateConfiguration();
			air_water_complex.updateConfiguration();

			tank.updateCellLinkedList();
			tank_contacts.updateConfiguration();
			/*tank_observer_contact.updateConfiguration();*/
		}

		TickCount t2 = TickCount::now();
		compute_vorticity.exec();

		/** Write run-time observation into file. */
		write_real_body_states.writeToFile();
		probe_1.writeToFile();
		/*write_tank_move.writeToFile();
		write_tank_nom.writeToFile();*/
		write_viscous_force_on_tank.writeToFile(number_of_iterations);
		write_total_force_on_tank.writeToFile(number_of_iterations);

		TickCount t3 = TickCount::now();
		interval += t3 - t2;
	}
	TickCount t4 = TickCount::now();

	TimeInterval tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

	return 0;
}