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
std::string full_path_to_blood_file = "./input/blood4-repaired.stl";
//std::string full_path_to_wall_file = "./input/bb11.stl";
Real length_scale = 1.0;
Vec3d translation(0.0, 0.0, 0.0);
Vec3d domain_lower_bound(-6.0 * length_scale, -4.0 * length_scale, -32.5 * length_scale);
Vec3d domain_upper_bound(12.0 * length_scale, 10.0 * length_scale, 23.5 * length_scale);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real resolution_ref = 0.3 / 3;

Vecd C_inlet(2.26, 5.30, -30.97);  /**< The center location of inlet. */
Real DW = 5.81;  /**< The diameter of main blood vessel. */
Vecd normal_inlet(-0.18, 0.58, 0.80);
Vecd C_outlet1(-2.29, -0.27, 21.82);
Real DW1 = 3.2;  /**< The diameter of first branch. */
Vecd normal_outlet1(0.53, -0.81, 0.26);
Vecd C_outlet2(8.47, 1.51, 18.62);
Real DW2 = 2.0;  /**< The diameter of second branch. */
Vecd normal_outlet2(0.0, 0.0, 1.0);

Real BW = resolution_ref * 4;         /**< Reference size of the emitter. */
Real DL_sponge = resolution_ref * 20; /**< Reference size of the emitter buffer to impose inflow condition. */
//-------------------------------------------------------
//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1.0; /**< Reference density of fluid. */
Real U_f = 1.0;    /**< Characteristic velocity. */
/** Reference sound speed needs to consider the flow speed in the narrow channels. */
Real c_f = 10.0 * U_f * SMAX(Real(1), DW / (DW1 + DW2));
Real Re = 100.0;                    /**< Reynolds number. */
Real mu_f = rho0_f * U_f * DW / Re; /**< Dynamics viscosity. */
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
        /*add<TriangleMeshShapeSTL>(full_path_to_wall_file, translation, length_scale);*/

        add<ExtrudeShape<TriangleMeshShapeSTL>>(4.0 * resolution_ref, full_path_to_blood_file, translation, length_scale);
        subtract<TriangleMeshShapeSTL>(full_path_to_blood_file, translation, length_scale);
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
        //target_velocity[0] = 1.5 * u_ave * SMAX(0.0, 1.0 - position[1] * position[1] / halfsize_[1] / halfsize_[1]);
        
        target_velocity[2] = 1.5 * u_ave * SMAX(0.0, 1.0 - position[1] * position[1] / halfsize_[1] / halfsize_[1]);
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
    sph_system.setRunParticleRelaxation(true); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(false);       // Tag for computation with save particles distribution
#ifdef BOOST_AVAILABLE
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment(); // handle command line arguments
#endif
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------
    FluidBody blood_block(sph_system, makeShared<Blood>("Blood"));
    blood_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    
    ParticleBuffer<ReserveSizeFactor> inlet_particle_buffer(0.5);
    blood_block.generateParticlesWithReserve<Lattice>(inlet_particle_buffer);

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
    wall_boundary.defineParticlesAndMaterial<SolidParticles, Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_boundary.generateParticles<Reload>(wall_boundary.getName())
        : wall_boundary.generateParticles<Lattice>();

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
        BodyStatesRecordingToVtp write_wall_state_to_vtp({wall_boundary});
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
        int relax_step = 1000;
        while (ite < relax_step)
        {
            relaxation_step_inner.exec();
            ite++;
            if (ite % 100 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
                write_wall_state_to_vtp.writeToFile(ite);
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
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(blood_block_inner, blood_wall_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallNoRiemann> density_relaxation(blood_block_inner, blood_wall_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_force(blood_block_inner, blood_wall_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(blood_block_inner, blood_wall_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> inlet_outlet_surface_particle_indicator(blood_block_inner, blood_wall_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationFreeStreamComplex> update_density_by_summation(blood_block_inner, blood_wall_contact);
    blood_block.addBodyStateForRecording<Real>("Pressure"); // output for debug
    blood_block.addBodyStateForRecording<int>("Indicator"); // output for debug
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(blood_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(blood_block);
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);

    Vecd emitter_halfsize = Vecd(0.5 * DW, 0.5 * DW, 0.5 * BW);
    Vecd emitter_translation = C_inlet - DL_sponge * normal_inlet + emitter_halfsize;
    BodyAlignedBoxByParticle emitter(blood_block, makeShared<AlignedBoxShape>(Transform(Vecd(emitter_translation)), emitter_halfsize));
    SimpleDynamics<fluid_dynamics::EmitterInflowInjection> emitter_inflow_injection(emitter, inlet_particle_buffer, zAxis);

    Vecd inlet_flow_buffer_halfsize = Vecd(0.5 * DW, 0.5 * DW, 0.5 * DL_sponge);
    Vecd inlet_flow_buffer_translation = C_inlet - DL_sponge * normal_inlet + inlet_flow_buffer_halfsize;;
    BodyAlignedBoxByCell inlet_flow_buffer(blood_block, makeShared<AlignedBoxShape>(Transform(Vecd(inlet_flow_buffer_translation)), inlet_flow_buffer_halfsize));
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_condition(inlet_flow_buffer);
   
    Vecd disposer_left_halfsize = Vecd(0.5 * DW1 * 1.1, 0.5 * DW1 * 1.1, 0.5 * BW);
    Vecd disposer_left_translation = C_outlet1 + disposer_left_halfsize;
    BodyAlignedBoxByCell disposer_left(
        blood_block, makeShared<AlignedBoxShape>(Transform(Vecd(disposer_left_translation)), disposer_left_halfsize));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> disposer_left_outflow_deletion(disposer_left, zAxis);

    Vecd disposer_right_halfsize = Vecd(0.5 * DW2 * 1.1, 0.5 * DW2 * 1.1, 0.5 * BW);
    Vecd disposer_right_translation = C_outlet2 + disposer_right_halfsize;
    BodyAlignedBoxByCell disposer_right(
        blood_block, makeShared<AlignedBoxShape>(Transform(Vecd(disposer_right_translation)), disposer_right_halfsize));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> disposer_right_outflow_deletion(disposer_right, zAxis);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_body_states(sph_system.real_bodies_);
    RegressionTestDynamicTimeWarping<ReducedQuantityRecording<TotalKineticEnergy>>
        write_blood_kinetic_energy(blood_block);
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
                write_blood_kinetic_energy.writeToFile(number_of_iterations);
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

    /*if (sph_system.GenerateRegressionData())
    {
        write_blood_kinetic_energy.generateDataBase(1.0e-3);
    }
    else
    {
        write_blood_kinetic_energy.testResult();
    }*/

    return 0;
}
