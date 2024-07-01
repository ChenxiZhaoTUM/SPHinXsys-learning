/**
 * @file 	carotid_artery.cpp
 * @brief 	
 * @details 
 * @author 	
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
Real length_scale = 1.0;
Vec3d domain_lower_bound(-7.0 * length_scale, -4.0 * length_scale, -35.0 * length_scale);
Vec3d domain_upper_bound(20.0 * length_scale, 12.0 * length_scale, 30.0 * length_scale);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real resolution_ref = 0.15;
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
Real DW_in = 2.9293 * 2;
Vec3d inlet_half = Vec3d(3.2, 3.2, 0.3);
Vec3d inlet_translation = Vec3d(1.5921, 5.8422, -30.5904);
Vec3d inlet_normal(0.1034, -0.0458, 0.9935);
Vec3d inlet_standard_direction(0, 0, 1);
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, inlet_standard_direction);
Rotation3d inlet_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);

//outlet1 R=1.9416, (-2.6975, -0.4330, 21.7855), (-0.3160, -0.0009, 0.9488)
Real DW_01 = 1.9416 * 2;
Vec3d outlet_01_half = Vec3d(2.2, 2.2, 0.3);
Vec3d outlet_01_translation = Vec3d(-2.6027, -0.4327, 21.5009);
Vec3d outlet_01_normal(-0.3160, -0.0009, 0.9488);
Vec3d outlet_01_standard_direction(0, 0, 1);
RotationResult outlet_01_rotation_result = RotationCalculator(outlet_01_normal, outlet_01_standard_direction);
Rotation3d outlet_01_rotation(outlet_01_rotation_result.angle, outlet_01_rotation_result.axis);

//outlet2 R=1.2760, (9.0465, 1.02552, 18.6363), (-0.0417, 0.0701, 0.9967)
Real DW_02 = 1.2760 * 2;
Vec3d outlet_02_half = Vec3d(1.5, 1.5, 0.3);
Vec3d outlet_02_translation = Vec3d(9.0590, 1.0045, 18.3373);
Vec3d outlet_02_normal(-0.0417, 0.0701, 0.9967);
Vec3d outlet_02_standard_direction(0, 0, 1);
RotationResult outlet_02_rotation_result = RotationCalculator(outlet_02_normal, outlet_02_standard_direction);
Rotation3d outlet_02_rotation(outlet_02_rotation_result.angle, outlet_02_rotation_result.axis);

//Real BW = resolution_ref * 6;         /**< Reference size of the emitter. */
//Real DL_sponge = resolution_ref * 20; /**< Reference size of the emitter buffer to impose inflow condition. */
//-------------------------------------------------------
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
class Blood : public ComplexShape
{
  public:
    explicit Blood(const std::string &shape_name) : ComplexShape(shape_name)
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
    AlignedBoxShape &aligned_box_;
    Vecd halfsize_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ref_(U_f), t_ref_(2.0),
          aligned_box_(boundary_condition.getAlignedBox()),
          halfsize_(aligned_box_.HalfSize()) {}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = velocity;
        Real run_time = GlobalStaticVariables::physical_time_;
        Real u_ave = run_time < t_ref_ ? 0.5 * u_ref_ * (1.0 - cos(Pi * run_time / t_ref_)) : u_ref_;
        target_velocity[2] = 1.5 * u_ave;
        return target_velocity;
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
    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineAdaptationRatios(1.15, 2.0);
    wall_boundary.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
    wall_boundary.defineParticlesAndMaterial<SolidParticles, Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_boundary.generateParticles<Reload>(wall_boundary.getName())
        : wall_boundary.generateParticles<Lattice>();
    
    FluidBody blood_block(sph_system, makeShared<Blood>("Blood"));
    blood_block.defineBodyLevelSetShape()->cleanLevelSet();
    blood_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    
    ParticleBuffer<ReserveSizeFactor> inlet_particle_buffer(0.5);
     (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? blood_block.generateParticlesWithReserve<Reload>(inlet_particle_buffer, blood_block.getName())
        : blood_block.generateParticles<Lattice>();
    //blood_block.generateParticlesWithReserve<Lattice>(inlet_particle_buffer);
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation wall_inner(wall_boundary);
        InnerRelation blood_inner(blood_block);
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_particles(wall_boundary);
        SimpleDynamics<RandomizeParticlePosition> random_blood_particles(blood_block);
        RelaxationStepInner relaxation_step_inner(wall_inner);
        RelaxationStepInner relaxation_step_inner_blood(blood_inner);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_wall_state_to_vtp(wall_boundary);
        BodyStatesRecordingToVtp write_blood_state_to_vtp(blood_block);
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(wall_boundary);
        ReloadParticleIO write_blood_particle_reload_files(blood_block);
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
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------

    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation blood_block_inner(blood_block);
    ContactRelation blood_wall_contact(blood_block, {&wall_boundary});
    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    // which is only used for update configuration.
    //----------------------------------------------------------------------
    ComplexRelation blood_block_complex(blood_block_inner, blood_wall_contact);
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> inlet_outlet_surface_particle_indicator(blood_block_inner, blood_wall_contact);

    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(blood_block_inner, blood_wall_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallNoRiemann> density_relaxation(blood_block_inner, blood_wall_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_force(blood_block_inner, blood_wall_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(blood_block_inner, blood_wall_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationFreeStreamComplex> update_density_by_summation(blood_block_inner, blood_wall_contact);
    blood_block.addBodyStateForRecording<Real>("Pressure"); // output for debug
    blood_block.addBodyStateForRecording<int>("Indicator"); // output for debug
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(blood_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(blood_block);
    

    BodyAlignedBoxByParticle emitter(blood_block, makeShared<AlignedBoxShape>(Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half));
    SimpleDynamics<fluid_dynamics::EmitterInflowInjection> emitter_inflow_injection(emitter, inlet_particle_buffer, zAxis);

    BodyAlignedBoxByCell inlet_flow_buffer(blood_block, makeShared<AlignedBoxShape>(Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half));
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_condition(inlet_flow_buffer);
   
    BodyAlignedBoxByCell disposer_left(
        blood_block, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> disposer_left_outflow_deletion(disposer_left, zAxis);

    BodyAlignedBoxByCell disposer_right(
        blood_block, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> disposer_right_outflow_deletion(disposer_right, zAxis);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_body_states(sph_system.real_bodies_);
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
            }
            number_of_iterations++;

            /** inflow injection*/
            emitter_inflow_injection.exec();
            disposer_left_outflow_deletion.exec();
            disposer_right_outflow_deletion.exec();

            /** Update cell linked list and configuration. */
            blood_block.updateCellLinkedListWithParticleSort(100);
            blood_block_complex.updateConfiguration();
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
