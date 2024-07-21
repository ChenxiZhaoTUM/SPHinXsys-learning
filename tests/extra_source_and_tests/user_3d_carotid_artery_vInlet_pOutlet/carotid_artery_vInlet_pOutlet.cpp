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
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_blood_file = "./input/carotid_fluid_geo.stl";
std::string full_path_to_wall_file = "./input/carotid_wall_geo.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real length_scale = pow(10, -3);
//Real length_scale = 1.0;
Vec3d domain_lower_bound(-7.0 * length_scale, -4.0 * length_scale, -35.0 * length_scale);
Vec3d domain_upper_bound(20.0 * length_scale, 12.0 * length_scale, 30.0 * length_scale);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real resolution_ref = 0.15 * length_scale;
//----------------------------------------------------------------------
//	Buffer location.
//----------------------------------------------------------------------
struct RotationResult
{
    Vec3d axis;
    Real angle;
};

RotationResult RotationCalculator(Vecd target_normal, Vecd standard_direction) 
{
    target_normal.normalize();

    Vec3d axis = standard_direction.cross(target_normal);
    Real angle = std::acos(standard_direction.dot(target_normal));

    if (axis.norm() < 1e-6)
    {
        if (standard_direction.dot(target_normal) < 0)
        {
            axis = Vec3d(1, 0, 0);
            angle = M_PI;
        }
        else
        {
            axis = Vec3d(0, 0, 1);
            angle = 0;
        }
    }
    else
    {
        axis.normalize();
    }

    return {axis, angle};
}

// inlet R=2.9293, (1.5611, 5.8559, -30.8885), (-0.1034, 0.0458, -0.9935)
Real DW_in = 2.9293 * 2 * length_scale;
Vec3d inlet_half = Vec3d(3.2, 3.2, 0.35) * length_scale;
Vec3d inlet_translation = Vec3d(1.5921, 5.8422, -30.5904) * length_scale;
Vec3d inlet_normal(0.1034, -0.0458, 0.9935);
Vec3d inlet_standard_direction(0, 0, 1);
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, inlet_standard_direction);
Rotation3d inlet_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);
Rotation3d inlet_desposer_rotation(inlet_rotation_result.angle + Pi, inlet_rotation_result.axis);

//outlet1 R=1.9416, (-2.6975, -0.4330, 21.7855), (-0.3160, -0.0009, 0.9488)
Real DW_01 = 1.9416 * 2 * length_scale;
Vec3d outlet_01_half = Vec3d(0.35, 2.4, 2.4) * length_scale;
Vec3d outlet_01_translation = Vec3d(-2.6027, -0.4327, 21.5009) * length_scale;
Vec3d outlet_01_normal(-0.3160, -0.0009, 0.9488);
Vec3d outlet_01_standard_direction(1, 0, 0);
RotationResult outlet_01_rotation_result = RotationCalculator(outlet_01_normal, outlet_01_standard_direction);
Rotation3d outlet_01_rotation(outlet_01_rotation_result.angle, outlet_01_rotation_result.axis);
Rotation3d outlet_emitter_01_rotation(outlet_01_rotation_result.angle + Pi, outlet_01_rotation_result.axis);

//outlet2 R=1.2760, (9.0465, 1.02552, 18.6363), (-0.0417, 0.0701, 0.9967)
Real DW_02 = 1.2760 * 2 * length_scale;
Vec3d outlet_02_half = Vec3d(0.35, 1.5, 1.5) * length_scale;
Vec3d outlet_02_translation = Vec3d(9.0590, 1.0045, 18.3373) * length_scale;
Vec3d outlet_02_normal(-0.0417, 0.0701, 0.9967);
Vec3d outlet_02_standard_direction(1, 0, 0);
RotationResult outlet_02_rotation_result = RotationCalculator(outlet_02_normal, outlet_02_standard_direction);
Rotation3d outlet_02_rotation(outlet_02_rotation_result.angle, outlet_02_rotation_result.axis);
Rotation3d outlet_emitter_02_rotation(outlet_02_rotation_result.angle + Pi, outlet_02_rotation_result.axis);

//Real BW = resolution_ref * 6;         /**< Reference size of the emitter. */
//Real DL_sponge = resolution_ref * 20; /**< Reference size of the emitter buffer to impose inflow condition. */
//-------------------------------------------------------
//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1060; /**< Reference density of fluid. */
Real U_f = 0.5;    /**< Characteristic velocity. */
/** Reference sound speed needs to consider the flow speed in the narrow channels. */
Real c_f = 10.0 * U_f * SMAX(Real(1), DW_in / (DW_01 + DW_02));
Real mu_f = 0.00355; /**< Dynamics viscosity. */
//Real Outlet_pressure = 1.33 * pow(10, 4);
Real Inlet_pressure = 0.2;
Real Outlet_pressure = 0.1;
//----------------------------------------------------------------------
//	Define case dependent body shapes.
//----------------------------------------------------------------------
class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeSTL>(full_path_to_blood_file, translation, length_scale);
    }
};

class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeSTL>(full_path_to_wall_file, translation, length_scale);
    }
};
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------

struct InflowVelocity
{
    Real u_ref_, t_ref_, interval_;
    AlignedBoxShape &aligned_box_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ref_(0.1), t_ref_(0.218), interval_(0.5),
        aligned_box_(boundary_condition.getAlignedBox()){}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = velocity;
        Real run_time = GlobalStaticVariables::physical_time_;
        int n = static_cast<int>(run_time / interval_);
        Real t_in_cycle = run_time - n * interval_;

        target_velocity[2] = t_in_cycle < t_ref_ ? 0.5 * sin(4 * Pi * (run_time + 0.0160236)) : u_ref_;
        return target_velocity;
    }
};

class TimeDependentAcceleration : public Gravity
{
    Real t_ref_, du_ave_dt_, interval_;

  public:
    explicit TimeDependentAcceleration(Vecd gravity_vector)
        : Gravity(gravity_vector), t_ref_(0.218), du_ave_dt_(0), interval_(0.5) {}

    virtual Vecd InducedAcceleration(const Vecd &position) override
    {
        Real run_time = GlobalStaticVariables::physical_time_;
        int n = static_cast<int>(run_time / interval_);
        Real t_in_cycle = run_time - n * interval_;

        du_ave_dt_ = 0.5 * 4 * Pi * cos(4 * Pi * run_time);

        return t_in_cycle < t_ref_ ? Vecd(0.0, 0.0, du_ave_dt_) : global_acceleration_;
    }
};

//----------------------------------------------------------------------
//	Pressure boundary definition.
//----------------------------------------------------------------------
struct LeftInflowPressure
{
    template <class BoundaryConditionType>
    LeftInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real &p_)
    {
        return p_;
    }
};

struct RightInflowPressure
{
    template <class BoundaryConditionType>
    RightInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real &p_)
    {
        /*constant pressure*/
        Real pressure = Outlet_pressure;
        return pressure;
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
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape()->cleanLevelSet();
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    //water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
    : water_block.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineAdaptationRatios(1.15, 2.0);
    wall_boundary.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
    wall_boundary.defineMaterial<Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_boundary.generateParticles<BaseParticles, Reload>(wall_boundary.getName())
        : wall_boundary.generateParticles<BaseParticles, Lattice>();
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation wall_inner(wall_boundary);
        InnerRelation blood_inner(water_block);
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_particles(wall_boundary);
        SimpleDynamics<RandomizeParticlePosition> random_blood_particles(water_block);
        RelaxationStepInner relaxation_step_inner(wall_inner);
        RelaxationStepInner relaxation_step_inner_blood(blood_inner);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_wall_state_to_vtp(wall_boundary);
        BodyStatesRecordingToVtp write_blood_state_to_vtp(water_block);
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(wall_boundary);
        ReloadParticleIO write_blood_particle_reload_files(water_block);
        //----------------------------------------------------------------------
        //	Physics relaxation starts here.
        //----------------------------------------------------------------------
        random_particles.exec(0.25);
        random_blood_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_inner_blood.SurfaceBounding().exec();
        write_wall_state_to_vtp.writeToFile(0.0);
        write_blood_state_to_vtp.writeToFile(0.0);
        //----------------------------------------------------------------------
        // From here the time stepping begins.
        //----------------------------------------------------------------------
        int ite = 0;
        int relax_step = 1000;
        while (ite < relax_step)
        {
            relaxation_step_inner.exec();
            ite++;
            if (ite % 100 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
                //write_wall_state_to_vtp.writeToFile(ite);
                //write_blood_state_to_vtp.writeToFile(ite);
            }
        }

        std::cout << "The physics relaxation process of wall particles finish !" << std::endl;
        write_wall_state_to_vtp.writeToFile(ite);
        write_blood_state_to_vtp.writeToFile(ite);
        write_particle_reload_files.writeToFile(0);
        write_blood_particle_reload_files.writeToFile(0);

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
    ContactRelation water_wall_contact(water_block, {&wall_boundary});
    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    // which is only used for update configuration.
    //----------------------------------------------------------------------
    ComplexRelation water_block_complex(water_block_inner, water_wall_contact);
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    TimeDependentAcceleration time_dependent_acceleration(Vecd::Zero());
    SimpleDynamics<GravityForce> apply_gravity_force(water_block, time_dependent_acceleration);
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_wall_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_wall_contact);

    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_wall_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_wall_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_wall_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_wall_contact);

    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);

    BodyAlignedBoxByCell left_emitter(water_block, makeShared<AlignedBoxShape>(zAxis, Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half));
    fluid_dynamics::NonPrescribedPressureBidirectionalBuffer left_emitter_inflow_injection(left_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_emitter_01(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half));
    fluid_dynamics::BidirectionalBuffer<RightInflowPressure> right_emitter_inflow_injection_01(right_emitter_01, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_emitter_02(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half));
    fluid_dynamics::BidirectionalBuffer<RightInflowPressure> right_emitter_inflow_injection_02(right_emitter_02, in_outlet_particle_buffer);

    BodyAlignedBoxByCell left_disposer(water_block, makeShared<AlignedBoxShape>(zAxis, Transform(Rotation3d(inlet_desposer_rotation),Vec3d(inlet_translation)), inlet_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_disposer_outflow_deletion(left_disposer);
    BodyAlignedBoxByCell right_disposer_01(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> right_disposer_outflow_deletion_01(right_disposer_01);
    BodyAlignedBoxByCell right_disposer_02(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> right_disposer_outflow_deletion_02(right_disposer_02);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_wall_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<LeftInflowPressure>> left_inflow_pressure_condition(left_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightInflowPressure>> right_inflow_pressure_condition_01(right_emitter_01);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightInflowPressure>> right_inflow_pressure_condition_02(right_emitter_02);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_emitter);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    boundary_indicator.exec();
    left_emitter_inflow_injection.tag_buffer_particles.exec();
    right_emitter_inflow_injection_01.tag_buffer_particles.exec();
    right_emitter_inflow_injection_02.tag_buffer_particles.exec();
    wall_boundary_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 2.5;   /**< End time. */
    Real Output_Time = 0.1; /**< Time stamps for output of body states. */
    Real dt = 0.0;          /**< Default acoustic time step sizes. */
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
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < Output_Time)
        {
            time_instance = TickCount::now();
            apply_gravity_force.exec();
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
                right_inflow_pressure_condition_01.exec(dt);
                right_inflow_pressure_condition_02.exec(dt);
                inflow_velocity_condition.exec();
                density_relaxation.exec(dt);
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

                body_states_recording.writeToFile();
            }
            number_of_iterations++;

            time_instance = TickCount::now();

            left_emitter_inflow_injection.injection.exec();
            right_emitter_inflow_injection_01.injection.exec();
            right_emitter_inflow_injection_02.injection.exec();
            left_disposer_outflow_deletion.exec();
            right_disposer_outflow_deletion_01.exec();
            right_disposer_outflow_deletion_02.exec();
            water_block.updateCellLinkedListWithParticleSort(100);
            water_block_complex.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;
            boundary_indicator.exec();
            left_emitter_inflow_injection.tag_buffer_particles.exec();
            right_emitter_inflow_injection_01.tag_buffer_particles.exec();
            right_emitter_inflow_injection_02.tag_buffer_particles.exec();
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
