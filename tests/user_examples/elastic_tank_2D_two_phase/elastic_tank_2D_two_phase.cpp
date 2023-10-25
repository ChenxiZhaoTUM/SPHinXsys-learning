/**
 * @file 	dambreak.cpp
 * @brief 	2D dambreak example.
 * @details This is the one of the basic test cases, also the first case for
 * 			understanding SPH method for fluid simulation.
 * @author 	Luhui Han, Chi Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //SPHinXsys Library.

#define PI (3.14159265358979323846)
using namespace SPH;   // Namespace cite here.

Real particle_spacing_ref = 0.007;	/**< Initial reference particle spacing. */

Real rho0_f = 1000.0;						 /**< Reference density of fluid. */
Real rho0_a = 1.226;						 /**< Reference density of air. */
Real rho0_s = 7890;
Real gravity_g = 9.81;					 /**< Gravity. */
Real U_max = 2.0 * sqrt(gravity_g * 0.5); /**< Characteristic velocity. */
Real c_f = 10.0 * U_max;				 /**< Reference sound speed. */
Real mu_water = 653.9e-6;
Real mu_air = 20.88e-6;

Real poisson = 0.27; 		/**< Poisson ratio.*/
Real Ae = 135.0e9; 			/**< Normalized Youngs Modulus. */
Real Youngs_modulus = Ae;
std::string water = "./input/water.dat";
std::string tank_inner = "./input/tank_inner.dat";
std::string tank_outer = "./input/tank_outer.dat";


Real f = 1.0;
Real a = 0.08;

class WaterBlock : public MultiPolygonShape
{
public:
	explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(water, ShapeBooleanOps::add);
		//multi_polygon_.addAPolygon(createWaterBlockShape(), ShapeBooleanOps::add);
	}
};

//----------------------------------------------------------------------
//	cases-dependent geometric shape for air block.
//----------------------------------------------------------------------
class AirBlock : public MultiPolygonShape
{
public:
	explicit AirBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(tank_inner, ShapeBooleanOps::add);
		multi_polygon_.addAPolygonFromFile(water, ShapeBooleanOps::sub);
		//multi_polygon_.addAPolygon(createWaterBlockShape(), ShapeBooleanOps::sub);
	}
};
//----------------------------------------------------------------------
//	Wall boundary shape definition.
//----------------------------------------------------------------------
class Tank : public MultiPolygonShape
{	
public:
	explicit Tank(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(tank_outer, ShapeBooleanOps::add);
		multi_polygon_.addAPolygonFromFile(tank_inner, ShapeBooleanOps::sub);
	}
};

class VariableGravity : public Gravity
{
	Real time_ = 0;
public:
	VariableGravity() : Gravity(Vecd(0.0, -gravity_g)) {};
	virtual Vecd InducedAcceleration(Vecd& position) override
	{
		time_ = GlobalStaticVariables::physical_time_;

		if (time_ < 0.5)
		{
			global_acceleration_[0] = 0.0;
		}
		else
		{
			global_acceleration_[0] = 4.0 * PI * PI * f * f * a * sin(2 * PI * f * (time_ - 0.5));
		}

		return global_acceleration_;
	}
};

typedef DataDelegateSimple<SolidParticles> SolidDataSimple;

template <typename VariableType>
class QuantityMomentOfMomentum : public QuantitySummation<VariableType>
{
protected:
	StdLargeVec<Vecd> &vel_;
	StdLargeVec<Vecd> &pos_;
	Vecd mass_center_;


public:
	explicit QuantityMomentOfMomentum(SPHBody &sph_body, Vecd mass_center= Vecd::Zero())
		: QuantitySummation<VariableType>(sph_body, "MassiveMeasure"),
		mass_center_(mass_center), vel_(this->particles_->vel_), pos_(this->particles_->pos_)
	{
		this->quantity_name_ = "Moment of Momentum";
	};
	virtual ~QuantityMomentOfMomentum() {};

	VariableType reduce(size_t index_i, Real dt = 0.0)
	{
		return ((pos_[index_i][0] - mass_center_[0]) * vel_[index_i][1] - (pos_[index_i][1] - mass_center_[1]) * vel_[index_i][0])* this->variable_[index_i];
	};
};

template <typename VariableType>
class QuantityMomentOfInertia : public QuantitySummation<VariableType>
{
protected:
	StdLargeVec<Vecd> &pos_;
	Vecd mass_center_;
	

public:
	explicit QuantityMomentOfInertia(SPHBody &sph_body, Vecd mass_center = Vecd::Zero())
		: QuantitySummation<VariableType>(sph_body, "MassiveMeasure"),
		pos_(this->particles_->pos_), mass_center_(mass_center)
	{
		this->quantity_name_ = "Moment of Inertia";
	};
	virtual ~QuantityMomentOfInertia() {};

	VariableType reduce(size_t index_i, Real dt = 0.0)
	{
		return (pos_[index_i] - mass_center_).norm() *(pos_[index_i] - mass_center_).norm() * this->variable_[index_i];
	};
};

class QuantityMassPosition : public QuantitySummation<Vecd>
{
protected:
	StdLargeVec<Real> &mass_;


public:
	explicit QuantityMassPosition(SPHBody &sph_body)
		: QuantitySummation<Vecd>(sph_body, "Position"),
		mass_(this->particles_->mass_)
	{
		this->quantity_name_ = "Mass*Position";
	};
	virtual ~QuantityMassPosition() {};

	Vecd reduce(size_t index_i, Real dt = 0.0)
	{
		return this->variable_[index_i] * mass_[index_i];
	};
};

class Constrain2DSolidBodyRotation : public LocalDynamics, public SolidDataSimple
{
private:
	Vecd mass_center_;
	Real moment_of_inertia_;
	Real angular_velocity_;
	ReduceDynamics<QuantityMomentOfMomentum<Real>> compute_total_moment_of_momentum_;
	StdLargeVec<Vecd> &vel_;
	StdLargeVec<Vecd> &pos_;

protected:
	virtual void setupDynamics(Real dt = 0.0) override
	{
		angular_velocity_ = compute_total_moment_of_momentum_.parallel_exec(dt) / moment_of_inertia_;
	}

public:
	explicit Constrain2DSolidBodyRotation(SPHBody &sph_body, Vecd mass_center = Vecd::Zero(), Real moment_of_inertia=Real(0.0))
		: LocalDynamics(sph_body), SolidDataSimple(sph_body),
		vel_(particles_->vel_), pos_(particles_->pos_), compute_total_moment_of_momentum_(sph_body, mass_center),
		mass_center_(mass_center), moment_of_inertia_(moment_of_inertia), angular_velocity_(Real(0.0)) {};
	virtual ~Constrain2DSolidBodyRotation() {};

	void update(size_t index_i, Real dt = 0.0)
	{
		Real x = pos_[index_i][0] - mass_center_[0];
		Real y = pos_[index_i][1] - mass_center_[1];
		Real local_radius = sqrt(pow(y, 2.0) + pow(x, 2.0));
		Real angular = atan2(y, x);
		Vecd linear_velocity_;

		linear_velocity_[1] = angular_velocity_ * local_radius * cos(angular);
		linear_velocity_[0] = -angular_velocity_ * local_radius * sin(angular);
		vel_[index_i] -= linear_velocity_;
	}
};

//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char* av[])
{
	/* Build up -- a SPHSystem -- */
	BoundingBox system_domain_bounds(Vec2d(-0.2, -0.090), Vec2d(0.2, 1.0));
	SPHSystem system(system_domain_bounds, particle_spacing_ref);
	// Tag for run particle relaxation for the initial body fitted distribution.
	system.setRunParticleRelaxation(true);
	// Tag for computation start with relaxed body fitted particles distribution.
	system.setReloadParticles(false);
	/* Tag for computation from restart files. 0: start with initial condition. */
	system.setRestartStep(0);
	//handle command line arguments
	system.handleCommandlineOptions(ac, av);
	/* Output environment. */
	IOEnvironment in_output(system);


	/*
	@Brief creating body, materials and particles for the cylinder.
	*/
	SolidBody tank(system, makeShared<Tank>("Tank"));
	tank.defineParticlesAndMaterial<ElasticSolidParticles, SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
	tank.defineBodyLevelSetShape()->writeLevelSet(in_output);
	(!system.RunParticleRelaxation() && system.ReloadParticles())
		? tank.generateParticles<ParticleGeneratorReload>(in_output, tank.getName())
		: tank.generateParticles<ParticleGeneratorLattice>();


	FluidBody water_block(system, makeShared<WaterBlock>("WaterBody"));
	water_block.defineParticlesAndMaterial<FluidParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_water);
	water_block.generateParticles<ParticleGeneratorLattice>();
	water_block.addBodyStateForRecording<Vecd>("Acceleration");
	water_block.addBodyStateForRecording<Real>("Pressure");

	FluidBody air_block(system, makeShared<AirBlock>("AirBody"));
	air_block.defineParticlesAndMaterial<FluidParticles, WeaklyCompressibleFluid>(rho0_a, c_f, mu_air);
	air_block.generateParticles<ParticleGeneratorLattice>();

	InnerRelation tank_inner(tank);
	//----------------------------------------------------------------------
	//	Run particle relaxation for body-fitted distribution if chosen.
	//----------------------------------------------------------------------
	if (system.RunParticleRelaxation())
	{
		//----------------------------------------------------------------------
		//	Methods used for particle relaxation.
		//----------------------------------------------------------------------
		/** Random reset the insert body particle position. */
		SimpleDynamics<RandomizeParticlePosition> random_tank_particles(tank);
		/** Write the body state to Vtp file. */
		BodyStatesRecordingToVtp write_tank_to_vtp(in_output, { &tank });
		/** Write the particle reload files. */
		ReloadParticleIO write_tank_particle_reload_files(in_output, tank, "Tank");
		/** A  Physics relaxation step. */
		relax_dynamics::RelaxationStepInner tank_relaxation_step_inner(tank_inner);
		//----------------------------------------------------------------------
		//	Particle relaxation starts here.
		//----------------------------------------------------------------------
		// random_tank_particles.parallel_exec(0.25);
		tank_relaxation_step_inner.SurfaceBounding().parallel_exec();
		write_tank_to_vtp.writeToFile(0);
		//----------------------------------------------------------------------
		//	Relax particles of the insert body.
		//----------------------------------------------------------------------
		int ite_p = 0;
		while (ite_p < 2000)
		{
			tank_relaxation_step_inner.parallel_exec();
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
	ContactRelation water_block_contact(water_block, { &tank });
	ContactRelation air_block_contact(air_block, { &tank });
	ComplexRelation water_air_complex(water_block, { &air_block });
	ComplexRelation air_water_complex(air_block, { &water_block });
	ContactRelation wall_boundary_water_contact(tank, RealBodyVector{ &water_block,&air_block });

	/*
	@Brief define simple data file input and outputs functions.
	*/
	BodyStatesRecordingToVtp 			write_real_body_states(in_output, system.real_bodies_);
	RestartIO							restart_io(in_output, system.real_bodies_);


	/** Initialize particle acceleration. */
	SimpleDynamics<NormalDirectionFromBodyShape> inner_normal_direction(tank);
	//InteractionDynamics<solid_dynamics::CorrectConfiguration> wall_boundary_corrected_configuration(tank_inner);

	//SimpleDynamics<TimeStepInitialization> initialize_a_water_step(water_block, makeShared<Gravity>(Vecd(0.0, -gravity_g)));
	//SimpleDynamics<TimeStepInitialization> initialize_a_air_step(air_block, makeShared<Gravity>(Vecd(0.0, -gravity_g)));

	SimpleDynamics<TimeStepInitialization> initialize_a_water_step(water_block, makeShared<VariableGravity>());
	SimpleDynamics<TimeStepInitialization> initialize_a_air_step(air_block, makeShared<VariableGravity>());

	/** Evaluation of density by summation approach. */
	InteractionWithUpdate<fluid_dynamics::DensitySummationFreeSurfaceComplex>
		update_water_density_by_summation(water_block_contact, water_air_complex.getInnerRelation());
	InteractionWithUpdate<fluid_dynamics::DensitySummationComplex>
		update_air_density_by_summation(air_block_contact, air_water_complex);
	InteractionDynamics<fluid_dynamics::TransportVelocityCorrectionComplex>
		air_transport_correction(air_block_contact, air_water_complex);
	InteractionDynamics<fluid_dynamics::ViscousAccelerationMultiPhaseWithWall> viscous_acceleration_water(water_block_contact, water_air_complex);
	InteractionDynamics<fluid_dynamics::ViscousAccelerationMultiPhaseWithWall> viscous_acceleration_air(air_block_contact, air_water_complex);
	/** Time step size without considering sound wave speed. */
	ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_water_advection_time_step_size(water_block, U_max);
	ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_air_advection_time_step_size(air_block, U_max);
	/** Time step size with considering sound wave speed. */
	ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_water_time_step_size(water_block);
	ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_air_time_step_size(air_block);
	/** Pressure relaxation for water by using position verlet time stepping. */
	Dynamics1Level<fluid_dynamics::Integration1stHalfRiemannWithWall>
		water_pressure_relaxation(water_block_contact, water_air_complex.getInnerRelation());
	Dynamics1Level<fluid_dynamics::Integration2ndHalfRiemannWithWall>
		water_density_relaxation(water_block_contact, water_air_complex.getInnerRelation());
	///** Extend Pressure relaxation is used for air. */
	//Dynamics1Level<fluid_dynamics::ExtendMultiPhaseIntegration1stHalfRiemannWithWall>
	//	vapor_pressure_relaxation(air_block_contact, air_water_complex, 2.0);
	//Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfRiemannWithWall>
	//	vapor_density_relaxation(air_block_contact, air_water_complex);

	//Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfRiemannWithWall>
	//	water_pressure_relaxation(water_block_contact, water_air_complex);
	//Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfRiemannWithWall>
	//	water_density_relaxation(water_block_contact, water_air_complex);
	/** Extend Pressure relaxation is used for air. */
	Dynamics1Level<fluid_dynamics::ExtendMultiPhaseIntegration1stHalfRiemannWithWall>
		vapor_pressure_relaxation(air_block_contact, air_water_complex);
	Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfRiemannWithWall>
		vapor_density_relaxation(air_block_contact, air_water_complex);
	//----------------------------------------------------------------------
	//	Algorithms of FSI.
	//----------------------------------------------------------------------
	InteractionDynamics<solid_dynamics::FluidViscousForceOnSolid> viscous_force_on_solid(wall_boundary_water_contact);
	InteractionDynamics<solid_dynamics::FluidForceOnSolidUpdate>
		fluid_force_on_solid_update(wall_boundary_water_contact, viscous_force_on_solid);
	solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(tank);
	//----------------------------------------------------------------------
	//	Algorithms of Elastic dynamics.
	//----------------------------------------------------------------------
	Dynamics1Level<solid_dynamics::Integration1stHalf> wall_stress_relaxation_first_half(tank_inner);
	Dynamics1Level<solid_dynamics::Integration2ndHalf> wall_stress_relaxation_second_half(tank_inner);
	ReduceDynamics<solid_dynamics::AcousticTimeStepSize> wall_computing_time_step_size(tank);

	SimpleDynamics<solid_dynamics::ConstrainSolidBodyMassCenter> constrain_mass_center_1(tank, Vecd(1.0, 1.0));
	ReduceDynamics<QuantitySummation<Real>> compute_total_mass_(tank, "MassiveMeasure");
	ReduceDynamics<QuantityMassPosition> compute_mass_position_(tank);
	Vecd mass_center = compute_mass_position_.parallel_exec() / compute_total_mass_.parallel_exec();
	Real moment_of_inertia = Real(0.0);
	ReduceDynamics<QuantityMomentOfInertia<Real>> compute_moment_of_inertia(tank, mass_center);
	moment_of_inertia = compute_moment_of_inertia.parallel_exec();;
	SimpleDynamics<Constrain2DSolidBodyRotation> constrain_rotation(tank, mass_center, moment_of_inertia);
	SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> wall_update_normal(tank);

	ReducedQuantityRecording<ReduceDynamics<solid_dynamics::TotalViscousForceOnSolid>>
		write_viscous_force_on_tank(in_output, tank);
	ReducedQuantityRecording<ReduceDynamics<solid_dynamics::TotalForceOnSolid>>
		write_total_force_on_tank(in_output, tank);

	/**
 * @brief Pre-simulation.
 */
 /** initialize cell linked lists for all bodies. */
	system.initializeSystemCellLinkedLists();
	/** initialize configurations for all bodies. */
	system.initializeSystemConfigurations();
	/** computing surface normal direction for the tank. */
	inner_normal_direction.parallel_exec();
	/** computing linear reproducing configuration for the tank. */
	//wall_boundary_corrected_configuration.parallel_exec();

	write_real_body_states.writeToFile(0);

	if (system.RestartStep() != 0)
	{
		GlobalStaticVariables::physical_time_ = restart_io.readRestartFiles(system.RestartStep());
		water_block.updateCellLinkedList();
		air_block.updateCellLinkedList();
		tank.updateCellLinkedList();
		water_air_complex.updateConfiguration();
		water_block_contact.updateConfiguration();
		air_water_complex.updateConfiguration();
		air_block_contact.updateConfiguration();
		wall_boundary_water_contact.updateConfiguration();
	}

	size_t number_of_iterations = system.RestartStep();
	int screen_output_interval = 100;
	int restart_output_interval = screen_output_interval * 10;
	Real End_Time = 20.0;			/**< End time. */
	Real D_Time = 0.05;	/**< time stamps for output. */
	Real dt = 0.0; 					/**< Default acoustic time step sizes for fluid. */

	/** Statistics for computing time. */
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
	while (GlobalStaticVariables::physical_time_ < End_Time)
	{
		Real integration_time = 0.0;
		/** Integrate time (loop) until the next output time. */
		while (integration_time < D_Time)
		{
			/** Acceleration due to viscous force and gravity. */

			initialize_a_water_step.parallel_exec();
			initialize_a_air_step.parallel_exec();

			Real Dt_f = get_water_advection_time_step_size.parallel_exec();
			Real Dt_a = get_air_advection_time_step_size.parallel_exec();
			Real Dt = SMIN(Dt_f, Dt_a);

			update_water_density_by_summation.parallel_exec();
			update_air_density_by_summation.parallel_exec();
			air_transport_correction.parallel_exec();

			viscous_acceleration_air.parallel_exec();
			viscous_acceleration_water.parallel_exec();
			viscous_force_on_solid.parallel_exec();
			wall_update_normal.parallel_exec();

			/** Dynamics including pressure relaxation. */
			Real relaxation_time = 0.0;
			while (relaxation_time < Dt)
			{
				Real dt_f = get_water_time_step_size.parallel_exec();
				Real dt_a = get_air_time_step_size.parallel_exec();
				dt = SMIN(SMIN(dt_f, dt_a), Dt);

				water_pressure_relaxation.parallel_exec(dt);
				vapor_pressure_relaxation.parallel_exec(dt);
				fluid_force_on_solid_update.parallel_exec();
				water_density_relaxation.parallel_exec(dt);
				vapor_density_relaxation.parallel_exec(dt);

				size_t inner_ite_dt_s = 0;
				Real dt_s_sum = 0.0;
				average_velocity_and_acceleration.initialize_displacement_.parallel_exec();
				while (dt_s_sum < dt)
				{
					Real dt_s = SMIN(wall_computing_time_step_size.parallel_exec(), dt - dt_s_sum);
					wall_stress_relaxation_first_half.parallel_exec(dt_s);
					constrain_rotation.parallel_exec(dt_s);
					constrain_mass_center_1.parallel_exec(dt_s);
					wall_stress_relaxation_second_half.parallel_exec(dt_s);

					dt_s_sum += dt_s;
					inner_ite_dt_s++;

					write_real_body_states.writeToFile();
					write_viscous_force_on_tank.writeToFile(number_of_iterations);
					write_total_force_on_tank.writeToFile(number_of_iterations);
				}
				average_velocity_and_acceleration.update_averages_.parallel_exec(dt);


				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
			}

			if (number_of_iterations % screen_output_interval == 0)
			{
				std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
					<< GlobalStaticVariables::physical_time_
					<< "	Dt = " << Dt << "	dt = " << dt << "\n";

				if (number_of_iterations % restart_output_interval == 0)
					restart_io.writeToFile(number_of_iterations);
			}

			number_of_iterations++;

			/** Update cell linked list and configuration. */

			water_block.updateCellLinkedListWithParticleSort(100);
			water_block_contact.updateConfiguration();
			water_air_complex.updateConfiguration();
			tank.updateCellLinkedList();
			air_block.updateCellLinkedListWithParticleSort(100);
			air_block_contact.updateConfiguration();
			air_water_complex.updateConfiguration();
			wall_boundary_water_contact.updateConfiguration();

		}
		tick_count t2 = tick_count::now();
		/** write run-time observation into file */
		/*write_real_body_states.writeToFile();
		write_viscous_force_on_tank.writeToFile(number_of_iterations);
		write_total_force_on_tank.writeToFile(number_of_iterations);*/

		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}

	tick_count t4 = tick_count::now();
	tick_count::interval_t tt;
	tt = t4 - t1 - interval;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
	return 0;
};
