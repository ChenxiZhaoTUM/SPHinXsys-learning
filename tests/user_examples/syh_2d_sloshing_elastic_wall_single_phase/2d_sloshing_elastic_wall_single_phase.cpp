/**
 * @file 	fsi2.cpp
 * @brief 	This is the benchmark test of fluid-structure interaction.
 * @details We consider a flow-induced vibration of an elastic beam behind a cylinder in 2D.
 *			The case can be found in Chi Zhang, Massoud Rezavand, Xiangyu Hu,
 *			Dual-criteria time stepping for weakly compressible smoothed particle hydrodynamics.
 *			Journal of Computation Physics 404 (2020) 109135.
 * @author 	Xiangyu Hu, Chi Zhang and Luhui Han
 */
#include "sphinxsys.h"
using namespace SPH;

//----------------------------------------------------------------------
//	Vertical Tank
//----------------------------------------------------------------------
Real resolution_ref = 0.007;              /**< Global reference resolution. */
std::string water = "./input/water.dat";
std::string tank_inner = "./input/tank_inner.dat";
std::string tank_outer = "./input/tank_outer.dat";
BoundingBox system_domain_bounds(Vec2d(-0.2, -0.090), Vec2d(0.2, 1.0));

//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1000.0;                                         /**< Density. */
Real gravity_g = 9.81;					                      /**< Gravity. */
Real U_max = 2.0 * sqrt(gravity_g * 0.5); /**< Characteristic velocity. */
Real c_f = 10.0 * U_max;				 /**< Reference sound speed. */
Real mu_water = 653.9e-6;
Real viscous_dynamics = rho0_f * U_max * 0.28; /**< Dynamics viscosity. */

//----------------------------------------------------------------------
//	Global parameters on the solid properties
//----------------------------------------------------------------------
Real rho0_s = 7890; /**< Reference density.*/
Real poisson = 0.27; /**< Poisson ratio.*/
Real Ae = 135.0e9;    /**< Normalized Youngs Modulus. */
Real Youngs_modulus = Ae;
Real physical_viscosity = 1.0e4;
// Real physical_viscosity = sqrt(rho0_s * Youngs_modulus) * 0.02 * 0.02 / 0.3 / 4;
//----------------------------------------------------------------------
//	Define case dependent geometries
//----------------------------------------------------------------------
class WaterBlock : public MultiPolygonShape
{
public:
	explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(water, ShapeBooleanOps::add);
	}
};

class WallBoundary : public MultiPolygonShape
{	
public:
	explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygonFromFile(tank_outer, ShapeBooleanOps::add);
		multi_polygon_.addAPolygonFromFile(tank_inner, ShapeBooleanOps::sub);
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

int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem with global controls.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    // Tag for run particle relaxation for the initial body fitted distribution.
	sph_system.setRunParticleRelaxation(false);
	// Tag for computation start with relaxed body fitted particles distribution.
	sph_system.setReloadParticles(true);

#ifdef BOOST_AVAILABLE
    sph_system.handleCommandlineOptions(ac, av); // handle command line arguments
#endif
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_water);
    water_block.generateParticles<ParticleGeneratorLattice>();
    water_block.addBodyStateForRecording<Vecd>("Acceleration");
	water_block.addBodyStateForRecording<Real>("Pressure");

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("Tank"));
    wall_boundary.defineParticlesAndMaterial<ElasticSolidParticles, SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
	{
		wall_boundary.generateParticles<ParticleGeneratorReload>(io_environment, wall_boundary.getName());
	}
	else
	{
		wall_boundary.defineBodyLevelSetShape()->writeLevelSet(io_environment);
		wall_boundary.generateParticles<ParticleGeneratorLattice>();
	}

    InnerRelation tank_inner(wall_boundary);
	//----------------------------------------------------------------------
	//	Run particle relaxation for body-fitted distribution if chosen.
	//----------------------------------------------------------------------
	if (sph_system.RunParticleRelaxation())
	{
		//----------------------------------------------------------------------
		//	Methods used for particle relaxation.
		//----------------------------------------------------------------------
		/** Random reset the insert body particle position. */
		SimpleDynamics<RandomizeParticlePosition> random_tank_particles(wall_boundary);
		/** Write the body state to Vtp file. */
		BodyStatesRecordingToVtp write_tank_to_vtp(io_environment, { &wall_boundary });
		/** Write the particle reload files. */
		ReloadParticleIO write_tank_particle_reload_files(io_environment, wall_boundary, "Tank");
		/** A  Physics relaxation step. */
		relax_dynamics::RelaxationStepInner tank_relaxation_step_inner(tank_inner);
		//----------------------------------------------------------------------
		//	Particle relaxation starts here.
		//----------------------------------------------------------------------
		random_tank_particles.exec(0.25);
		tank_relaxation_step_inner.SurfaceBounding().exec();
		write_tank_to_vtp.writeToFile(0);
		//----------------------------------------------------------------------
		//	Relax particles of the insert body.
		//----------------------------------------------------------------------
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

    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //----------------------------------------------------------------------
    ComplexRelation water_block_complex(water_block, {&wall_boundary});
    ContactRelation wall_water_contact(wall_boundary, {&water_block});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    /** Initialize particle acceleration. */
    SimpleDynamics<TimeStepInitialization> initialize_a_fluid_step(water_block, makeShared<Gravity>(Vecd(0.0, -gravity_g)));
    /** Evaluation of density by summation approach. */
    //InteractionWithUpdate<fluid_dynamics::DensitySummationComplex> update_density_by_summation(water_block_complex);
    InteractionWithUpdate<fluid_dynamics::DensitySummationFreeSurfaceComplex> update_density_by_summation(water_block_complex);
    /** Time step size without considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_max);
    /** Time step size with considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    /** Pressure relaxation using verlet time stepping. */
    /** Here, we do not use Riemann solver for pressure as the flow is viscous. */
    Dynamics1Level<fluid_dynamics::Integration1stHalfRiemannWithWall> pressure_relaxation(water_block_complex);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfRiemannWithWall> density_relaxation(water_block_complex);

    InteractionDynamics<fluid_dynamics::ViscousAccelerationWithWall> viscous_acceleration(water_block_complex);
    /** Computing vorticity in the flow. */
    InteractionDynamics<fluid_dynamics::VorticityInner> compute_vorticity(water_block_complex.getInnerRelation());

    //----------------------------------------------------------------------
    //	Algorithms of FSI.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
    /** Corrected configuration for the elastic insert body. */
    InteractionWithUpdate<CorrectedConfigurationInner> tank_corrected_configuration(tank_inner);
    /** Compute the force exerted on solid body due to fluid pressure and viscosity. */
    InteractionDynamics<solid_dynamics::ViscousForceFromFluid> viscous_force_on_solid(wall_water_contact);
    InteractionDynamics<solid_dynamics::AllForceAccelerationFromFluid>
        fluid_force_on_solid_update(wall_water_contact, viscous_force_on_solid);
    /** Compute the average velocity of the insert body. */
    solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(wall_boundary);

    //----------------------------------------------------------------------
    //	Algorithms of solid dynamics.
    //----------------------------------------------------------------------
    /** Compute time step size of elastic solid. */
    ReduceDynamics<solid_dynamics::AcousticTimeStepSize> tank_computing_time_step_size(wall_boundary);
    /** Stress relaxation for the inserted body. */
    Dynamics1Level<solid_dynamics::Integration1stHalfPK2> tank_stress_relaxation_first_half(tank_inner);
    Dynamics1Level<solid_dynamics::Integration2ndHalf> tank_stress_relaxation_second_half(tank_inner);
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseWithWall<Vec2d, DampingPairwiseInner>>>
        fluid_damping(0.2, water_block_complex, "Velocity", viscous_dynamics);
    DampingWithRandomChoice<InteractionSplit<DampingBySplittingInner<Vecd>>>
        tank_damping(0.2, tank_inner, "Velocity", physical_viscosity);

    SimpleDynamics<solid_dynamics::ConstrainSolidBodyMassCenter> constrain_mass_center_1(wall_boundary, Vecd(1.0, 1.0));
	ReduceDynamics<QuantitySummation<Real>> compute_total_mass_(wall_boundary, "MassiveMeasure");
	ReduceDynamics<QuantityMassPosition> compute_mass_position_(wall_boundary);
	Vecd mass_center = compute_mass_position_.exec() / compute_total_mass_.exec();
	Real moment_of_inertia = Real(0.0);
	ReduceDynamics<QuantityMomentOfInertia<Real>> compute_moment_of_inertia(wall_boundary, mass_center);
	moment_of_inertia = compute_moment_of_inertia.exec();;
	SimpleDynamics<Constrain2DSolidBodyRotation> constrain_rotation(wall_boundary, mass_center, moment_of_inertia);

    /** Update norm .*/
    SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> tank_update_normal(wall_boundary);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
    ReducedQuantityRecording<ReduceDynamics<solid_dynamics::TotalForceFromFluid>>
        write_total_viscous_force_on_tank(io_environment, viscous_force_on_solid, "TotalViscousForceOnSolid");
    ReducedQuantityRecording<ReduceDynamics<TotalMechanicalEnergy>>
        write_kinetic_energy(io_environment, wall_boundary);

    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    /** initialize cell linked lists for all bodies. */
    sph_system.initializeSystemCellLinkedLists();
    /** initialize configurations for all bodies. */
    sph_system.initializeSystemConfigurations();
    /** computing surface normal direction for the wall. */
    wall_boundary_normal_direction.exec();
    /** computing linear reproducing configuration for the insert body. */
    tank_corrected_configuration.exec();
    //----------------------------------------------------------------------
    //	Setup computing and initial conditions.
    //----------------------------------------------------------------------
    size_t number_of_iterations = 0;
    int screen_output_interval = 100;
    //Real end_time = 200.0;
    //Real output_interval = end_time / 200.0;
    Real end_time = 50.0;
    Real output_interval = 0.05;
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    write_real_body_states.writeToFile();
    write_kinetic_energy.writeToFile(0);
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < output_interval)
        {
            initialize_a_fluid_step.exec();
            Real Dt = get_fluid_advection_time_step_size.exec();
            update_density_by_summation.exec();
            viscous_acceleration.exec();

            /** FSI for viscous force. */
            viscous_force_on_solid.exec();
            /** Update normal direction on elastic body.*/
            tank_update_normal.exec();
            size_t inner_ite_dt = 0;
            size_t inner_ite_dt_s = 0;
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                Real dt = SMIN(get_fluid_time_step_size.exec(), Dt);

                fluid_damping.exec(dt);

                /** Fluid pressure relaxation */
                pressure_relaxation.exec(dt);
                /** FSI for pressure force. */
                fluid_force_on_solid_update.exec();
                /** Fluid density relaxation */
                density_relaxation.exec(dt);

                /** Solid dynamics. */
                inner_ite_dt_s = 0;
                Real dt_s_sum = 0.0;
                average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    Real dt_s = SMIN(tank_computing_time_step_size.exec(), dt - dt_s_sum);
                    tank_stress_relaxation_first_half.exec(dt_s);
                    
                    constrain_rotation.exec();
					constrain_mass_center_1.exec();

                    tank_damping.exec(dt_s);

                    constrain_rotation.exec();
					constrain_mass_center_1.exec();

                    tank_stress_relaxation_second_half.exec(dt_s);
                    dt_s_sum += dt_s;
                    inner_ite_dt_s++;
                }
                average_velocity_and_acceleration.update_averages_.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
                inner_ite_dt++;
            }

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << GlobalStaticVariables::physical_time_
                          << "	Dt = " << Dt << "	Dt / dt = " << inner_ite_dt << "	dt / dt_s = " << inner_ite_dt_s << "\n";
            }
            number_of_iterations++;

            water_block.updateCellLinkedListWithParticleSort(100);
            water_block_complex.updateConfiguration();
            /** one need update configuration after periodic condition. */
            wall_boundary.updateCellLinkedList();
            wall_water_contact.updateConfiguration();
        }

        TickCount t2 = TickCount::now();
        /** write run-time observation into file */
        compute_vorticity.exec();
        write_real_body_states.writeToFile();
        write_total_viscous_force_on_tank.writeToFile(number_of_iterations);
        write_kinetic_energy.writeToFile(number_of_iterations);

        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
    return 0;
}
