/**
 * @file 	T_shaped_pipe.cpp
 * @brief 	This is the benchmark test of multi-inlet and multi-outlet.
 * @details We consider a flow with one inlet and two outlets in a T-shaped pipe in 2D.
 * @author 	Xiangyu Hu, Shuoguo Zhang
 */

#include "sphinxsys.h"
#include "bidirectional_buffer.h"
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "pressure_boundary.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Set the file path to the data file.
//----------------------------------------------------------------------
//std::string water_path = "./input/Yshape_water_for_code.dat";
std::string wall_inner_path = "./input/Yshape_inner_boundary_for_code.dat";
std::string wall_outer_path = "./input/Yshape_outer_boundary_for_code.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vecd translation(0.0, 0.0);
Real length_scale = pow(10, -2);
BoundingBox system_domain_bounds(Vec2d(-3.0 * length_scale, -12.0 * length_scale), Vecd(44.0 * length_scale, 16.0 * length_scale));
Real resolution_ref = 0.5 * length_scale;           /**< Initial reference particle spacing. */
Real BW = resolution_ref * 4;         /**< Reference size of the emitter. */
Real DH = 7.0 * length_scale;
Real DL = 25.0 * length_scale;
Real thickness = 0.5 * length_scale;
Real level_set_refinement_ratio = resolution_ref / (0.1 * thickness);
//-------------------------------------------------------
//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real Inlet_pressure = 0.2;
Real Outlet_pressure = 0.1;
Real rho0_f = 1060; /**< Reference density of fluid. */
Real U_f = 0.5;                                                      /**< Characteristic velocity. */
Real mu_f = 0.00355; /**< Dynamics viscosity. */
Real c_f = 10.0 * U_f * SMAX(Real(1), DH / (Real(1.66 * length_scale) + Real(2.57 * length_scale)));

Real rho0_s = 1120;                /** Normalized density. */
Real Youngs_modulus = 1.08e6;    /** Normalized Youngs Modulus. */
Real poisson = 0.49;               /** Poisson ratio. */
//----------------------------------------------------------------------
//	Define case dependent body shapes.
//----------------------------------------------------------------------
class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygonFromFile(wall_inner_path, ShapeBooleanOps::add, translation, length_scale);
    }
};

class ShellShape : public MultiPolygonShape
{
  public:
    explicit ShellShape(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygonFromFile(wall_outer_path, ShapeBooleanOps::add, translation, length_scale);
        multi_polygon_.addAPolygonFromFile(wall_inner_path, ShapeBooleanOps::sub, translation, length_scale);
    }
};
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
struct InflowVelocity
{
    Real u_ref_, t_ref_;
    AlignedBoxShape &aligned_box_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ref_(U_f), t_ref_(2.0),
            aligned_box_(boundary_condition.getAlignedBox()){}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = Vecd::Zero();
        
        Real run_time = GlobalStaticVariables::physical_time_;
        Real u_ave = run_time < t_ref_ ? 0.5 * u_ref_ * (1.0 - cos(Pi * run_time / t_ref_)) : u_ref_;
        target_velocity[0] = u_ave;
        target_velocity[1] = 0.0;

        return target_velocity;
    }
};

/**
 * @brief 	Pressure boundary definition.
 */
struct LeftInflowPressure
{
    template <class BoundaryConditionType>
    LeftInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real &p_)
    {
        return p_;
    }
};

struct UpOutflowPressure
{
    template <class BoundaryConditionType>
    UpOutflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real &p_)
    {
        /*constant pressure*/
        Real pressure = Outlet_pressure;
        return pressure;
    }
};

struct DownOutflowPressure
{
    template <class BoundaryConditionType>
    DownOutflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real &p_)
    {
        /*constant pressure*/
        Real pressure = Outlet_pressure;
        return pressure;
    }
};
//----------------------------------------------------------------------
//	Define constrain class for tank translation and rotation.
//----------------------------------------------------------------------
template <typename VariableType, class DynamicsIdentifier>
class QuantityMomentOfMomentum : public QuantitySummation<VariableType, DynamicsIdentifier>
{
protected:
	StdLargeVec<Vecd> &vel_;
	StdLargeVec<Vecd>& pos_;
	Vecd mass_center_;

public:
	explicit QuantityMomentOfMomentum(SPHBody& sph_body, Vecd mass_center)
		: QuantitySummation<VariableType, DynamicsIdentifier>(sph_body, "Mass"),
		mass_center_(mass_center), 
        vel_(*this->particles_->template getVariableDataByName<Vecd>("Velocity")), 
        pos_(*this->particles_->template getVariableDataByName<Vecd>("Position"))
	{
		this->quantity_name_ = "Moment of Momentum";
	};
	virtual ~QuantityMomentOfMomentum() {};

	VariableType reduce(size_t index_i, Real dt = 0.0)
	{
		return ((pos_[index_i][0] - mass_center_[0]) * vel_[index_i][1] - (pos_[index_i][1] - mass_center_[1]) * vel_[index_i][0])* this->variable_[index_i];
	};
};

template <typename VariableType, class DynamicsIdentifier>
class QuantityMomentOfInertia : public QuantitySummation<VariableType, DynamicsIdentifier>
{
protected:
	StdLargeVec<Vecd>& pos_;
	Vecd mass_center_;

public:
	explicit QuantityMomentOfInertia(SPHBody& sph_body, Vecd mass_center = Vecd::Zero())
		: QuantitySummation<VariableType, DynamicsIdentifier>(sph_body, "Mass"),
		pos_(*this->particles_->template getVariableDataByName<Vecd>("Position")), 
        mass_center_(mass_center)
	{
		this->quantity_name_ = "Moment of Inertia";
	};
	virtual ~QuantityMomentOfInertia() {};

	Real reduce(size_t index_i, Real dt = 0.0)
	{
		return (pos_[index_i] - mass_center_).norm() *(pos_[index_i] - mass_center_).norm() * this->variable_[index_i];
	};
};

class QuantityMassPosition : public QuantitySummation<Vecd>
{
protected:
	StdLargeVec<Real>& mass_;


public:
	explicit QuantityMassPosition(SPHBody& sph_body)
		: QuantitySummation<Vecd>(sph_body, "Position"),
		mass_(*particles_->getVariableDataByName<Real>("Mass"))
	{
		this->quantity_name_ = "Mass*Position";
	};
	virtual ~QuantityMassPosition() {};

	Vecd reduce(size_t index_i, Real dt = 0.0)
	{
		return this->variable_[index_i] * mass_[index_i];
	};
};

class Constrain2DSolidBodyRotation : public LocalDynamics, public DataDelegateSimple
{
private:
	Vecd mass_center_;
	Real moment_of_inertia_;
	Real angular_velocity_;
	Vecd linear_velocity_;
	ReduceDynamics<QuantityMomentOfMomentum<Real, SPHBody>> compute_total_moment_of_momentum_;
	StdLargeVec<Vecd>& vel_;
	StdLargeVec<Vecd>& pos_;

protected:
	virtual void setupDynamics(Real dt = 0.0) override
	{
        angular_velocity_ = compute_total_moment_of_momentum_.exec(dt) / moment_of_inertia_;
	}

public:
	explicit Constrain2DSolidBodyRotation(SPHBody& sph_body, Vecd mass_center, Real moment_of_inertia)
		: LocalDynamics(sph_body), DataDelegateSimple(sph_body),
		vel_(*particles_->getVariableDataByName<Vecd>("Velocity")), 
        pos_(*particles_->getVariableDataByName<Vecd>("Position")), 
        compute_total_moment_of_momentum_(sph_body, mass_center),
		mass_center_(mass_center), moment_of_inertia_(moment_of_inertia) {}

	virtual ~Constrain2DSolidBodyRotation() {};

	void update(size_t index_i, Real dt = 0.0)
	{
		Real x = pos_[index_i][0] - mass_center_[0];
		Real y = pos_[index_i][1] - mass_center_[1];
		Real local_radius = sqrt(pow(y, 2.0) + pow(x, 2.0));
		Real angular = atan2(y, x);

		linear_velocity_[1] = angular_velocity_ * local_radius * cos(angular);
		linear_velocity_[0] = -angular_velocity_ * local_radius * sin(angular);
		vel_[index_i] -= linear_velocity_;
	}
};
//-----------------------------------------------------------------------------------------------------------
//	Main program starts here.
//-----------------------------------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem with global controls.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(true);       // Tag for computation with save particles distribution
#ifdef BOOST_AVAILABLE
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment(); // handle command line arguments
#endif
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);

    SolidBody shell_body(sph_system, makeShared<ShellShape>("ShellBody"));
    shell_body.defineBodyLevelSetShape(level_set_refinement_ratio)->writeLevelSet(sph_system);
    shell_body.defineMaterial<SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? shell_body.generateParticles<SurfaceParticles, Reload>(shell_body.getName())
        : shell_body.generateParticles<SurfaceParticles, Lattice>(thickness);

    // test buffer location
    /*Vec2d test_half = Vec2d(0.5 * 3.0, 0.5 * BW);
    Vec2d test_translation = Vec2d(32.716, 13.854);
    Real test_rotation = -0.8506;
    SolidBody test_up(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(test_rotation), Vec2d(test_translation)), test_half, "TestBody"));
    test_up.defineParticlesAndMaterial<SolidParticles, Solid>();
    test_up.generateParticles<Lattice>();*/
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation imported_model_inner(shell_body);
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_imported_model_particles(shell_body);
        /** A  Physics relaxation step. */
        ShellRelaxationStep relaxation_step_inner(imported_model_inner);
        ShellNormalDirectionPrediction shell_normal_prediction(imported_model_inner, thickness);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_imported_model_to_vtp({shell_body});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(shell_body);
        //----------------------------------------------------------------------
        //	Physics relaxation starts here.
        //----------------------------------------------------------------------
        relaxation_step_inner.MidSurfaceBounding().exec();
        write_imported_model_to_vtp.writeToFile(0.0);
        shell_body.updateCellLinkedList();
        //----------------------------------------------------------------------
        // From here the time stepping begins.
        //----------------------------------------------------------------------
        int ite = 0;
        int relax_step = 1000;
        while (ite < relax_step)
        {
            relaxation_step_inner.exec();
            ite++;
            if (ite % 200 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
                write_imported_model_to_vtp.writeToFile(ite);
            }
        }

        std::cout << "The physics relaxation process of wall particles finish !" << std::endl;
        write_particle_reload_files.writeToFile(0);

        return 0;
    }
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------  
    InnerRelation water_block_inner(water_block);
    InnerRelation shell_inner(shell_body);
    ContactRelationFromShellToFluid water_shell_contact(water_block, {&shell_body}, {false});
    ContactRelationFromFluidToShell shell_water_contact(shell_body, {&water_block}, {false});
    ShellInnerRelationWithContactKernel shell_curvature_inner(shell_body, water_block);
    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    // which is only used for update configuration.
    //----------------------------------------------------------------------
    ComplexRelation water_block_complex(water_block_inner, {&water_shell_contact});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    // shell dynamics
    InteractionDynamics<thin_structure_dynamics::ShellCorrectConfiguration> shell_corrected_configuration(shell_inner);
    Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationFirstHalf> shell_stress_relaxation_first(shell_inner, 3, true);
    Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationSecondHalf> shell_stress_relaxation_second(shell_inner);
    ReduceDynamics<thin_structure_dynamics::ShellAcousticTimeStepSize> shell_time_step_size(shell_body);
    SimpleDynamics<thin_structure_dynamics::AverageShellCurvature> shell_average_curvature(shell_curvature_inner);
    SimpleDynamics<thin_structure_dynamics::UpdateShellNormalDirection> shell_update_normal(shell_body);

    /** Exert constrain on shell. */
	SimpleDynamics<solid_dynamics::ConstrainSolidBodyMassCenter> constrain_mass_center(shell_body);
	ReduceDynamics<QuantitySummation<Real>> compute_total_mass_(shell_body, "Mass");
	ReduceDynamics<QuantityMassPosition> compute_mass_position_(shell_body);
	Vecd mass_center = compute_mass_position_.exec() / compute_total_mass_.exec();
	Real moment_of_inertia = Real(0.0);
    ReduceDynamics<QuantityMomentOfInertia<Real, SPHBody>> compute_moment_of_inertia(shell_body, mass_center);
	moment_of_inertia = compute_moment_of_inertia.exec();;
	SimpleDynamics<Constrain2DSolidBodyRotation> constrain_rotation(shell_body, mass_center, moment_of_inertia);


    // fluid dynamics
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_shell_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_shell_contact);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_shell_contact);

    Vec2d left_bidirectional_halfsize = Vec2d(0.5 * BW, 0.5 * 7.0 * length_scale);
    Vec2d left_bidirectional_translation = Vec2d(0.5 * BW, 0.0);
    BodyAlignedBoxByCell left_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec2d(left_bidirectional_translation)), left_bidirectional_halfsize));
    fluid_dynamics::NonPrescribedPressureBidirectionalBuffer left_emitter_inflow_injection(left_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell left_disposer(
        water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(Pi), Vec2d(left_bidirectional_translation)), left_bidirectional_halfsize));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_disposer_outflow_deletion(left_disposer);

    Vec2d right_up_bidirectional_halfsize = Vec2d(0.6 * BW, 0.5 * 3.0 * length_scale);
    Vec2d right_up_bidirectional_translation = Vec2d(33.0 * length_scale, 14.2 * length_scale);
    Real right_up_disposer_rotation = 0.722;
    Real right_up_emitter_rotation = right_up_disposer_rotation + Pi;
    BodyAlignedBoxByCell right_up_emitter(
        water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(right_up_emitter_rotation), Vec2d(right_up_bidirectional_translation)), right_up_bidirectional_halfsize));
    fluid_dynamics::BidirectionalBuffer<UpOutflowPressure> right_up_emitter_inflow_injection(right_up_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_up_disposer(
        water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(right_up_disposer_rotation), Vec2d(right_up_bidirectional_translation)), right_up_bidirectional_halfsize));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> right_up_disposer_outflow_deletion(right_up_disposer);

    Vec2d right_down_bidirectional_halfsize = Vec2d(0.6 * BW, 0.5 * 4.0 * length_scale);
    Vec2d right_down_bidirectional_translation = Vec2d(42.0 * length_scale, -9.7 * length_scale);
    Real right_down_disposer_rotation = -0.317;
    Real right_down_emitter_rotation = right_down_disposer_rotation + Pi;
    BodyAlignedBoxByCell right_down_emitter(
        water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(right_down_emitter_rotation), Vec2d(right_down_bidirectional_translation)), right_down_bidirectional_halfsize));
    fluid_dynamics::BidirectionalBuffer<DownOutflowPressure> right_down_emitter_inflow_injection(right_down_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_down_disposer(
        water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(right_down_disposer_rotation), Vec2d(right_down_bidirectional_translation)), right_down_bidirectional_halfsize));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> right_down_disposer_outflow_deletion(right_down_disposer);
    
    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_shell_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<LeftInflowPressure>> left_inflow_pressure_condition(left_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<UpOutflowPressure>> up_inflow_pressure_condition(right_up_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<DownOutflowPressure>> down_inflow_pressure_condition(right_down_emitter);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_emitter);


    // FSI
    InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_on_shell(shell_water_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_on_shell(shell_water_contact);
    solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(shell_body);


    RealBody buffer_body_in(
        sph_system, makeShared<AlignedBoxShape>(xAxis,Transform(Vec2d(left_bidirectional_translation)), left_bidirectional_halfsize, "BufferBodyIn"));
    buffer_body_in.generateParticles<BaseParticles, Lattice>();
    RealBody buffer_body_out01(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(right_up_emitter_rotation), Vec2d(right_up_bidirectional_translation)), right_up_bidirectional_halfsize, "BufferBodyOut01"));
    buffer_body_out01.generateParticles<BaseParticles, Lattice>();
    RealBody buffer_body_out02(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(right_down_disposer_rotation), Vec2d(right_down_bidirectional_translation)), right_down_bidirectional_halfsize, "BufferBodyOut02"));
    buffer_body_out02.generateParticles<BaseParticles, Lattice>();


    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    body_states_recording.addToWrite<Vecd>(shell_body, "PressureForceFromFluid");
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    shell_corrected_configuration.exec();
    shell_average_curvature.exec();
    water_block_complex.updateConfiguration();
    shell_water_contact.updateConfiguration();
    boundary_indicator.exec();
    left_emitter_inflow_injection.tag_buffer_particles.exec();
    right_up_emitter_inflow_injection.tag_buffer_particles.exec();
    right_down_emitter_inflow_injection.tag_buffer_particles.exec();
    //----------------------------------------------------------------------
    //	Setup computing and initial conditions.
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 2.5;   /**< End time. */
    Real Output_Time = 0.1; /**< Time stamps for output of body states. */
    Real dt = 0.0;          /**< Default acoustic time step sizes. */
    Real dt_s = 0.0; /**< Default acoustic time step sizes for solid. */
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
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
    //----------------------------------------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < Output_Time)
        {
            time_instance = TickCount::now();
            Real Dt = get_fluid_advection_time_step_size.exec();
            //std::cout << "Dt = " << Dt << std::endl;
            update_fluid_density.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();
            interval_computing_time_step += TickCount::now() - time_instance;

            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);
                //std::cout << "dt = " << dt << std::endl;

                pressure_relaxation.exec(dt);
                kernel_summation.exec();
                left_inflow_pressure_condition.exec(dt);
                up_inflow_pressure_condition.exec(dt);
                down_inflow_pressure_condition.exec(dt);
                inflow_velocity_condition.exec();
                density_relaxation.exec(dt);

                Real dt_s_sum = 0.0;
                average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    dt_s = 0.5 * shell_time_step_size.exec(); // why 0.5?
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;
                    shell_stress_relaxation_first.exec(dt_s);

                    constrain_rotation.exec(dt_s);
					constrain_mass_center.exec(dt_s);

                    shell_stress_relaxation_second.exec(dt_s);
                    dt_s_sum += dt_s;

                    body_states_recording.writeToFile();
                }
                average_velocity_and_acceleration.update_averages_.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
            }
            interval_computing_pressure_relaxation += TickCount::now() - time_instance;

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << GlobalStaticVariables::physical_time_
                          << "	Dt = " << Dt << "	dt = " << dt << "	dt_s = " << dt_s << "\n";

                /*body_states_recording.writeToFile();*/
            }
            number_of_iterations++;

            time_instance = TickCount::now();

            left_emitter_inflow_injection.injection.exec();
            right_up_emitter_inflow_injection.injection.exec();
            right_down_emitter_inflow_injection.injection.exec();
            left_disposer_outflow_deletion.exec();
            right_up_disposer_outflow_deletion.exec();
            right_down_disposer_outflow_deletion.exec();

            water_block.updateCellLinkedListWithParticleSort(100);
            shell_update_normal.exec();
            shell_body.updateCellLinkedList();
            shell_curvature_inner.updateConfiguration();
            shell_average_curvature.exec();
            shell_water_contact.updateConfiguration();
            water_block_complex.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;
            boundary_indicator.exec();

            left_emitter_inflow_injection.tag_buffer_particles.exec();
            right_up_emitter_inflow_injection.tag_buffer_particles.exec();
            right_down_emitter_inflow_injection.tag_buffer_particles.exec();
        }
        TickCount t2 = TickCount::now();
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