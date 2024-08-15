/**
 * @file 	T_shaped_pipe.cpp
 * @brief 	This is the benchmark test of multi-inlet and multi-outlet.
 * @details We consider a flow with one inlet and two outlets in a T-shaped pipe in 2D.
 * @author 	Xiangyu Hu, Shuoguo Zhang
 */

#include "sphinxsys.h"
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
Vec3d domain_lower_bound(-7.0 * length_scale, -4.0 * length_scale, -35.0 * length_scale);
Vec3d domain_upper_bound(20.0 * length_scale, 12.0 * length_scale, 30.0 * length_scale);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 0.15 * length_scale;
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

Vecd standard_direction(1, 0, 0);

// inlet R=2.9293, (1.5611, 5.8559, -30.8885), (-0.1034, 0.0458, -0.9935)
Real DW_in = 2.9293 * 2 * length_scale;
Vecd inlet_buffer_half = Vecd(2.0 * dp_0, 3.2 * length_scale, 3.2 * length_scale);
Vecd inlet_normal(0.1034, -0.0458, 0.9935);
Vecd inlet_emitter_translation = Vecd(1.5611, 5.8559, -30.8885) * length_scale + inlet_normal * 2.0 * length_scale;
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, standard_direction);
Rotation3d inlet_emitter_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);

//outlet1 R=1.9416, (-2.6975, -0.4330, 21.7855), (-0.3160, -0.0009, 0.9488)
Real DW_01 = 1.9416 * 2 * length_scale;
Vecd outlet_01_buffer_half = Vecd(2.0 * dp_0, 2.4 * length_scale, 2.4 * length_scale);
Vecd outlet_01_normal(-0.3160, -0.0009, 0.9488);
Vecd outlet_01_disposer_translation = Vecd(-2.6975, -0.4330, 21.7855) * length_scale - outlet_01_normal * 2.0 * length_scale;
RotationResult outlet_01_rotation_result = RotationCalculator(outlet_01_normal, standard_direction);
Rotation3d outlet_01_disposer_rotation(outlet_01_rotation_result.angle, outlet_01_rotation_result.axis);

//outlet2 R=1.2760, (9.0465, 1.02552, 18.6363), (-0.0417, 0.0701, 0.9967)
Real DW_02 = 1.2760 * 2 * length_scale;
Vecd outlet_02_buffer_half = Vecd(2.0 * dp_0, 1.5 * length_scale, 1.5 * length_scale);
Vecd outlet_02_normal(-0.0417, 0.0701, 0.9967);
Vecd outlet_02_disposer_translation = Vecd(9.0465, 1.02552, 18.6363) * length_scale - outlet_02_normal * 2.0 * length_scale;
RotationResult outlet_02_rotation_result = RotationCalculator(outlet_02_normal, standard_direction);
Rotation3d outlet_02_disposer_rotation(outlet_02_rotation_result.angle, outlet_02_rotation_result.axis);

//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1.0; /**< Reference density of fluid. */
Real U_f = 1.0;    /**< Characteristic velocity. */
/** Reference sound speed needs to consider the flow speed in the narrow channels. */
Real c_f = 10.0 * U_f * SMAX(Real(1), DW_in / (DW_01 + DW_02));
Real Re = 100.0;                    /**< Reynolds number. */
Real mu_f = rho0_f * U_f * DW_in / Re; /**< Dynamics viscosity. */
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
    Real u_ref_, t_ref_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ref_(U_f), t_ref_(2.0) {}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = velocity;
        Real run_time = GlobalStaticVariables::physical_time_;
        target_velocity[0] = run_time < t_ref_ ? 0.5 * u_ref_ * (1.0 - cos(Pi * run_time / t_ref_)) : u_ref_;
        return target_velocity;
    }
};

class TimeDependentAcceleration : public Gravity
{
    Real t_ref_, u_ref_, du_ave_dt_;

  public:
    explicit TimeDependentAcceleration(Vecd gravity_vector)
        : Gravity(gravity_vector), t_ref_(2.0), u_ref_(U_f), du_ave_dt_(0) {}

    virtual Vecd InducedAcceleration(const Vecd &position) override
    {
        Real run_time_ = GlobalStaticVariables::physical_time_;
        du_ave_dt_ = 0.5 * u_ref_ * (Pi / t_ref_) * sin(Pi * run_time_ / t_ref_);

        return run_time_ < t_ref_ ? Vecd(du_ave_dt_, 0.0, 0.0) : global_acceleration_;
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
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(true);       // Tag for computation with save particles distribution
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape()->cleanLevelSet();
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> inlet_particle_buffer(0.5);
    water_block.generateParticlesWithReserve<BaseParticles, Lattice>(inlet_particle_buffer);

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
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_particles(wall_boundary);
        RelaxationStepInner relaxation_step_inner(wall_inner);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_wall_state_to_vtp(wall_boundary);
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(wall_boundary);
        //----------------------------------------------------------------------
        //	Physics relaxation starts here.
        //----------------------------------------------------------------------
        random_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        write_wall_state_to_vtp.writeToFile(0.0);
        //----------------------------------------------------------------------
        // From here the time stepping begins.
        //----------------------------------------------------------------------
        int ite = 0;
        int relax_step = 2000;
        while (ite < relax_step)
        {
            relaxation_step_inner.exec();
            ite++;
            if (ite % 500 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
                write_wall_state_to_vtp.writeToFile(ite);
            }
        }

        std::cout << "The physics relaxation process of wall particles finish !" << std::endl;
        write_wall_state_to_vtp.writeToFile(ite);
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
    // initial acceleration
    TimeDependentAcceleration time_dependent_acceleration(Vecd::Zero());
    SimpleDynamics<GravityForce> apply_initial_force(water_block, time_dependent_acceleration);
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> inlet_outlet_surface_particle_indicator(water_block_inner, water_wall_contact);

    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_wall_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallNoRiemann> density_relaxation(water_block_inner, water_wall_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_force(water_block_inner, water_wall_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_wall_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationFreeStreamComplex> update_density_by_summation(water_block_inner, water_wall_contact);
    
    // time step
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);

    // add emitter and disposer
    BodyAlignedBoxByParticle emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_emitter_translation)), inlet_buffer_half));
    SimpleDynamics<fluid_dynamics::EmitterInflowInjection> emitter_inflow_injection(emitter, inlet_particle_buffer);

    BodyAlignedBoxByCell inlet_flow_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_emitter_translation)), inlet_buffer_half));
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_condition(inlet_flow_buffer);
   
    BodyAlignedBoxByCell disposer_up(
        water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_01_disposer_rotation), Vec3d(outlet_01_disposer_translation)), outlet_01_buffer_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> disposer_up_outflow_deletion(disposer_up);

    BodyAlignedBoxByCell disposer_down(
        water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_02_disposer_rotation), Vec3d(outlet_02_disposer_translation)), outlet_02_buffer_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> disposer_down_outflow_deletion(disposer_down);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_body_states(sph_system);
    write_body_states.addToWrite<Real>(water_block, "Pressure");
    write_body_states.addToWrite<int>(water_block, "Indicator");
    write_body_states.addToWrite<Real>(water_block, "Density");
    ReducedQuantityRecording<TotalKineticEnergy> write_water_kinetic_energy(water_block);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    wall_boundary_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Setup computing and initial conditions.
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    Real end_time = 100.0;
    Real output_interval = end_time / 200.0; /**< Time stamps for output of body states. */
    Real dt = 0.0;                           /**< Default acoustic time step sizes. */
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    write_body_states.writeToFile();
    //----------------------------------------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < output_interval)
        {
            apply_initial_force.exec();
            Real Dt = get_fluid_advection_time_step_size.exec();
            inlet_outlet_surface_particle_indicator.exec();
            update_density_by_summation.exec();
            viscous_force.exec();
            transport_velocity_correction.exec();

            /** Dynamics including pressure relaxation. */
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt - relaxation_time);
                pressure_relaxation.exec(dt);
                inflow_condition.exec();
                density_relaxation.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
            }

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << GlobalStaticVariables::physical_time_
                          << "	Dt = " << Dt << "	dt = " << dt << "\n";
                write_water_kinetic_energy.writeToFile(number_of_iterations);
            }
            number_of_iterations++;

            /** inflow injection*/
            emitter_inflow_injection.exec();
            disposer_up_outflow_deletion.exec();
            disposer_down_outflow_deletion.exec();

            /** Update cell linked list and configuration. */
            water_block.updateCellLinkedListWithParticleSort(100);
            water_block_complex.updateConfiguration();
        }

        TickCount t2 = TickCount::now();
        write_body_states.writeToFile();
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds()
              << " seconds." << std::endl;

    return 0;
}
