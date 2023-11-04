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

//----------------------------------------------------------------------
//	LNG Tank
//----------------------------------------------------------------------
Real resolution_ref = 0.0045;              /**< Global reference resolution. */
std::string water = "./input/water_profile.dat";
std::string tank_inner = "./input/tank_inner_profile.dat";
std::string tank_outer = "./input/tank_outer_0.018_profile.dat";
Vecd translation(0, 0.12);
BoundingBox system_domain_bounds(Vecd(-0.6, -0.2), Vecd(0.6, 0.4));

//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1000.0;                                         /**< Density. */
Real rho0_a = 1.226;						 /**< Reference density of air. */
Real gravity_g = 9.81;					                      /**< Gravity. */
Real U_max = 2.0 * sqrt(gravity_g * 0.0612); /**< Characteristic velocity. */
Real c_f = 10.0 * U_max;				 /**< Reference sound speed. */
Real mu_water = 653.9e-6;
Real mu_air = 20.88e-6;
Real viscous_dynamics = rho0_f * U_max * 0.64; /**< Dynamics viscosity. */

//----------------------------------------------------------------------
//	Global parameters on the solid properties
//----------------------------------------------------------------------
Real rho0_s = 2800.0; /**< Reference density.*/
Real poisson = 0.33; /**< Poisson ratio.*/
Real Ae = 70.0e9; /**< Normalized Youngs Modulus. */
//Real Ae = 135.0e9;
Real Youngs_modulus = Ae;
Real physical_viscosity = 1.3e4;
// Real physical_viscosity = sqrt(rho0_s * Youngs_modulus) * 0.03 * 0.03 / 0.24 / 4;

//----------------------------------------------------------------------
//	Define case dependent geometries
//----------------------------------------------------------------------
class WaterBlock : public MultiPolygonShape
{
public:
	explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(water, ShapeBooleanOps::add, translation);
	}
};

class AirBlock : public MultiPolygonShape
{
public:
	explicit AirBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(tank_inner, ShapeBooleanOps::add, translation);
		multi_polygon_.addAPolygonFromFile(water, ShapeBooleanOps::sub, translation);
	}
};

class WallBoundary : public MultiPolygonShape
{	
public:
	explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(tank_outer, ShapeBooleanOps::add, translation);
		multi_polygon_.addAPolygonFromFile(tank_inner, ShapeBooleanOps::sub, translation);
	}
};

class InitialDensity
    : public fluid_dynamics::FluidInitialCondition
{
  public:
      InitialDensity(SPHBody &sph_body)
        : fluid_dynamics::FluidInitialCondition(sph_body),
          fluid_particles_(dynamic_cast<BaseParticles *>(&sph_body.getBaseParticles())),
          p_(*fluid_particles_->getVariableByName<Real>("Pressure")), rho_(fluid_particles_->rho_){};

    void update(size_t index_i, Real dt)
    {
        p_[index_i] = rho0_f * gravity_g * (0.0612 - pos_[index_i][1]);
        rho_[index_i] = p_[index_i] / pow(c_f, 2) + rho0_f;
    }

  protected:
    BaseParticles *fluid_particles_;
    StdLargeVec<Real> &p_, &rho_;
};


Real h = 1.3 * resolution_ref;
MultiPolygon createWaveProbeShape()
{
    std::vector<Vecd> pnts;
    pnts.push_back(Vecd(-0.198 - h, 0.0));
    pnts.push_back(Vecd(-0.198 - h, 0.106));
    pnts.push_back(Vecd(-0.198 + h, 0.106));
    pnts.push_back(Vecd(-0.198 + h, 0.0));
    pnts.push_back(Vecd(-0.198 - h, 0.0));

    MultiPolygon multi_polygon;
    multi_polygon.addAPolygon(pnts, ShapeBooleanOps::add);
    return multi_polygon;
}

//----------------------------------------------------------------------
//	Define external excitation 02.
//----------------------------------------------------------------------
Real omega =  2 * PI * 0.5496;
Real Theta0 = - 3.0 * PI / 180.0;

class VariableGravitySecond : public Gravity
{
	Real time_ = 0;

public:
	VariableGravitySecond() : Gravity(Vecd(0.0, 0.0)) {};
	virtual Vecd InducedAcceleration(Vecd& position) override
	{
		time_= GlobalStaticVariables::physical_time_;
		Real Theta = Theta0 * sin(omega * (time_ - 1.0));
		Real ThetaV = Theta0 * omega * cos(omega * (time_ - 1.0));

		Real alpha = std::atan2(position[1], position[0]);
		Real distance = std::sqrt(pow(position[0], 2) + pow(position[1], 2));
		Real Vx = Theta * distance * std::sin(alpha);
		Real Vy = Theta * distance * std::cos(alpha);

		if (time_ < 1.0)
		{
			global_acceleration_[0] = 0.0;
			global_acceleration_[1] = -gravity_g;
		}
		else
		{
			global_acceleration_[0] = -gravity_g * sin(Theta) - ThetaV * ThetaV * position[0] + 2 * ThetaV * Vy;
			global_acceleration_[1] = -gravity_g * cos(Theta) + ThetaV * ThetaV * position[1] - 2 * ThetaV * Vx;
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
		angular_velocity_ = compute_total_moment_of_momentum_.exec(dt) / moment_of_inertia_;
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
	SolidBody tank(sph_system, makeShared<WallBoundary>("Tank"));
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
	water_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_water);
	water_block.generateParticles<ParticleGeneratorLattice>();
	water_block.addBodyStateForRecording<Real>("Pressure");
	water_block.addBodyStateForRecording<Vecd>("Acceleration");

	FluidBody air_block(sph_system, makeShared<AirBlock>("AirBody"));
	air_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_a, c_f, mu_air);
	air_block.generateParticles<ParticleGeneratorLattice>();
	air_block.addBodyStateForRecording<Real>("Pressure");

	//--------------------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//--------------------------------------------------------------------------------
	InnerRelation tank_inner(tank);

	ContactRelation water_block_contact(water_block, { &tank });
	ContactRelation air_block_contact(air_block, { &tank });
	ContactRelation tank_contacts(tank, RealBodyVector{ &water_block, &air_block });

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
	//	Algorithms of fluid dynamics.
	//--------------------------------------------------------------------------------
	SimpleDynamics<InitialDensity> initial_density_condition(water_block);

	/** Time step initialization of fluid body. */
	SimpleDynamics<TimeStepInitialization> initialize_a_water_step(water_block, makeShared<VariableGravitySecond>());
	SimpleDynamics<TimeStepInitialization> initialize_a_air_step(air_block, makeShared<VariableGravitySecond>());

	InteractionWithUpdate<fluid_dynamics::DensitySummationFreeSurfaceComplex>
		water_density_by_summation(water_block_contact, water_air_complex.getInnerRelation());
	InteractionWithUpdate<fluid_dynamics::DensitySummationComplex>
		air_density_by_summation(air_block_contact, air_water_complex);
	InteractionDynamics<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>>
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
	Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfRiemannWithWall>
		water_pressure_relaxation(water_block_contact, water_air_complex);
	Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfRiemannWithWall>
		water_density_relaxation(water_block_contact, water_air_complex);

	/*Dynamics1Level<fluid_dynamics::Integration1stHalfRiemannWithWall>
		water_pressure_relaxation(water_block_contact, water_air_complex.getInnerRelation());
	Dynamics1Level<fluid_dynamics::Integration2ndHalfRiemannWithWall>
		water_density_relaxation(water_block_contact, water_air_complex.getInnerRelation());*/
	Dynamics1Level<fluid_dynamics::ExtendMultiPhaseIntegration1stHalfRiemannWithWall>
		air_pressure_relaxation(air_block_contact, air_water_complex, 2.0);
	Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfRiemannWithWall>
		air_density_relaxation(air_block_contact, air_water_complex);

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
	SimpleDynamics<NormalDirectionFromBodyShape> tank_normal_direction(tank);
	InteractionWithUpdate<KernelCorrectionMatrixInner> tank_corrected_configuration(tank_inner);
	/** Time step size of elastic body. */
	ReduceDynamics<solid_dynamics::AcousticTimeStepSize> tank_acoustic_time_step(tank);
	/** Stress relaxation for the elastic body. */
	Dynamics1Level<solid_dynamics::Integration1stHalfPK2> tank_stress_relaxation_1st_half(tank_inner);
	Dynamics1Level<solid_dynamics::Integration2ndHalf> tank_stress_relaxation_2nd_half(tank_inner);
	DampingWithRandomChoice<InteractionSplit<DampingBySplittingInner<Vecd>>>
		tank_damping(0.2, tank_inner, "Velocity", physical_viscosity);

	/** Exert constrain on tank. */
	SimpleDynamics<solid_dynamics::ConstrainSolidBodyMassCenter> constrain_mass_center_1(tank, Vecd(1.0, 1.0));
	ReduceDynamics<QuantitySummation<Real>> compute_total_mass_(tank, "MassiveMeasure");
	ReduceDynamics<QuantityMassPosition> compute_mass_position_(tank);
	Vecd mass_center = compute_mass_position_.exec() / compute_total_mass_.exec();
	Real moment_of_inertia = Real(0.0);
	ReduceDynamics<QuantityMomentOfInertia<Real>> compute_moment_of_inertia(tank, mass_center);
	moment_of_inertia = compute_moment_of_inertia.exec();
	SimpleDynamics<Constrain2DSolidBodyRotation> constrain_rotation(tank, mass_center, moment_of_inertia);

	/** Update normal direction. */
	SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> tank_update_normal_direction(tank);

	//--------------------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//--------------------------------------------------------------------------------
	BodyRegionByCell probe_s1(water_block, makeShared<MultiPolygonShape>(createWaveProbeShape(), "PorbeS1"));
	ReducedQuantityRecording<UpperFrontInAxisDirection<BodyPartByCell>>
		probe_1(io_environment, probe_s1, "FreeSurfaceHeight");

	BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
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