/**
 * @file 	pulsatile_Aorta_flow.cpp
 * @brief 	3D pulsatile Aorta flow example
 * @details This is the one of the basic test cases for pressure boundary condition and bidirectional buffer.
 * @author 	Shuoguo Zhang and Xiangyu Hu
 */
/**
 * @brief 	SPHinXsys Library.
 */
#include "sphinxsys.h" 
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "pressure_boundary.h"
#include "bidirectional_buffer.h"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "windkessel_bc.h"
/**
 * @brief Namespace cite here.
 */
using namespace SPH;

std::string full_path_to_stl_file_wall = "./input/wall.stl";
std::string full_path_to_stl_file_fluid = "./input/blood.stl";

/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vecd(-6.0E-2, -4.0E-2, -2.0E-2), Vecd(3.0E-2, 10.0E-2, 15.0E-2));

Real Inlet_pressure = 0.0;
Real Outlet_pressure = 0.0;
Real rho0_f = 1060.0;                   
Real mu_f = 0.00355;
Real U_f = 3.0;
Real c_f = 10.0*U_f;

Real resolution_ref = 0.06E-2;
Real scaling = 1.0E-2;
Vecd translation(0.0, 0.0, 0.0);
StdVec<Vecd> observer_location = {Vecd(-1.24E-2, 4.41E-2, 5.18E-2)};

Vecd buffer_halfsize_inlet = Vecd(2.1 * resolution_ref, 3.0E-2, 3.0E-2);
Vecd buffer_translation_inlet = Vecd(-0.9768E-2, 4.6112E-2, 3.0052E-2);
Vecd normal_vector_inlet = Vecd(0.1000, 0.1665, 0.9810);
Vecd vector_inlet = Vecd(0, -0.9859, 0.1674);

Vecd buffer_halfsize_1 = Vecd(1.5 * resolution_ref, 1.0E-2, 1.0E-2);
Vecd buffer_translation_1 = Vecd(-1.2562E-2, 4.4252E-2, 10.0148E-2);
Vecd normal_vector_1 = Vecd(0.6420, 0.4110, 0.6472);
Vecd vector_1 = Vecd(0, -0.8442, 0.5360);

Vecd buffer_halfsize_2 = Vecd(1.6 * resolution_ref, 1.0E-2, 1.0E-2);
Vecd buffer_translation_2 = Vecd(-2.6303E-2, 3.0594E-2, 10.6919E-2);
Vecd normal_vector_2 = Vecd(-0.0988, 0.0485, 0.9939);
Vecd vector_2 = Vecd(0, -0.9988, 0.0488);

Vecd buffer_halfsize_3 = Vecd(1.6 * resolution_ref, 0.4E-2, 0.4E-2);
Vecd buffer_translation_3 = Vecd(-2.8585E-2, 1.8357E-2, 9.8034E-2);
Vecd normal_vector_3 = Vecd(-0.1471, -0.1813, 0.9724);
Vecd vector_3 = Vecd(0, -0.9831, -0.1833);

Vecd buffer_halfsize_4 = Vecd(1.1 * resolution_ref, 1.0E-2, 1.0E-2);
Vecd buffer_translation_4 = Vecd(-1.0946E-2, 1.0386E-2, 9.5016E-2);
Vecd normal_vector_4 = Vecd(0.5675, 0.4280, 0.7034);
Vecd vector_4 = Vecd(0, -0.8543, 0.5197);
  
Vecd buffer_halfsize_5 = Vecd(1.3 * resolution_ref, 2.0E-2, 2.0E-2);
Vecd buffer_translation_5 = Vecd(-1.6791E-2, -0.8069E-2, 0.5017E-2);
Vecd normal_vector_5 = Vecd(0.0327, -0.0729, 0.9968);
Vecd vector_5 = Vecd(0, -0.9973, -0.0729);

class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        /** Geometry definition. */
        add<TriangleMeshShapeSTL>(full_path_to_stl_file_fluid, translation, scaling);
        subtract<TriangleMeshShapeSTL>(full_path_to_stl_file_wall, translation, scaling);
    }
};

class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        /** Geometry definition. */
        add<TriangleMeshShapeSTL>(full_path_to_stl_file_wall, translation, scaling);
    }
};

struct InflowVelocity
{
    Real u_ave;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ave(0.0) {}

    Vecd operator()(Vecd &position, Vecd &velocity, Real current_time)
    {
        Vecd target_velocity = velocity;

        u_ave = 0.1502;
        Real a[8] = {0.1743, 0.0008, -0.0899, -0.0824, -0.0699, -0.0414, -0.011, 0.0044};
        Real b[8] = {0.1234, 0.1041, 0.0126, -0.005, 0.0289, 0.0415, 0.0246, 0.0122};
        Real w = 2 * Pi / 0.812;
        for (size_t i = 0; i < 8; i++)
        {
            u_ave = SMAX(u_ave + a[i] * cos(w * (i + 1) * current_time) + b[i] * sin(w * (i + 1) * current_time),
                        0.0);
        }
            
        target_velocity[0] = u_ave;
        target_velocity[1] = 0.0;
        target_velocity[2] = 0.0;

        return target_velocity;
    }
};

/**
 * @brief 	Main program starts here.
 */
int main(int ac, char *av[])
{
    /**
     * @brief Build up -- a SPHSystem --
     */
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.setGenerateRegressionData(false);
    /** Tag for run particle relaxation for the initial body fitted distribution. */
    sph_system.setRunParticleRelaxation(false);
    /** Tag for computation start with relaxed body fitted particles distribution. */
    sph_system.setReloadParticles(true);
    /** handle command line arguments. */
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    /**
     * @brief Material property, particles and body creation of fluid.
     */
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    //water_block.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
    water_block.defineBodyLevelSetShape()->correctLevelSetSign();
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
        : water_block.generateParticles<BaseParticles, Lattice>();
    /**
     * @brief 	Particle and body creation of wall boundary.
     */
    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("Wall"));
    wall_boundary.defineAdaptationRatios(1.15, 2.0);
    //wall_boundary.defineBodyLevelSetShape()->writeLevelSet(sph_system);
    wall_boundary.defineBodyLevelSetShape();
    wall_boundary.defineMaterial<Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_boundary.generateParticles<BaseParticles, Reload>(wall_boundary.getName())
        : wall_boundary.generateParticles<BaseParticles, Lattice>();

    ObserverBody velocity_observer(sph_system, "VelocityObserver");
    velocity_observer.generateParticles<ObserverParticles>(observer_location);
    /** topology */
    InnerRelation water_block_inner(water_block);
    ContactRelation water_block_contact(water_block, {&wall_boundary});
    ContactRelation velocity_observer_contact(velocity_observer, {&water_block});
    ComplexRelation water_block_complex(water_block_inner, water_block_contact);
    if (sph_system.RunParticleRelaxation())
    {
        /** body topology only for particle relaxation */
        InnerRelation water_block_inner(water_block);
        InnerRelation wall_boundary_inner(wall_boundary);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** Random reset the insert body particle position. */
        SimpleDynamics<RandomizeParticlePosition> water_block_random_inserted_body_particles(water_block);
        SimpleDynamics<RandomizeParticlePosition> wall_boundary_random_inserted_body_particles(wall_boundary);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp water_block_write_inserted_body_to_vtp(water_block);
        BodyStatesRecordingToVtp wall_boundary_write_inserted_body_to_vtp(wall_boundary);
        /** Write the particle reload files. */
        ReloadParticleIO water_block_write_particle_reload_files({&water_block});
        ReloadParticleIO wall_boundary_write_particle_reload_files({&wall_boundary});
        /** A  Physics relaxation step. */
        RelaxationStepInner water_block_relaxation_step_inner(water_block_inner);
        RelaxationStepInner wall_boundary_relaxation_step_inner(wall_boundary_inner);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        water_block_random_inserted_body_particles.exec(0.25);
        wall_boundary_random_inserted_body_particles.exec(0.25);
        water_block_relaxation_step_inner.SurfaceBounding().exec();
        wall_boundary_relaxation_step_inner.SurfaceBounding().exec();
        water_block_write_inserted_body_to_vtp.writeToFile(0);
        wall_boundary_write_inserted_body_to_vtp.writeToFile(0);
        //----------------------------------------------------------------------
        //	Relax particles of the insert body.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 3000)
        {
            water_block_relaxation_step_inner.exec();
            wall_boundary_relaxation_step_inner.exec();
            ite_p += 1;
            if (ite_p % 500 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
                water_block_write_inserted_body_to_vtp.writeToFile(ite_p);
                wall_boundary_write_inserted_body_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of inserted body finish !" << std::endl;
        /** Output results. */
        water_block_write_particle_reload_files.writeToFile(0);
        wall_boundary_write_particle_reload_files.writeToFile(0);
        return 0;
    }
    /**
     * @brief 	Define all numerical methods which are used in this case.
     */
    /**
     * @brief 	Methods used for time stepping.
     */
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_block_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex>
        free_stream_surface_indicator(water_block_inner, water_block_contact);
    /** Pressure relaxation algorithm without Riemann solver for viscous flows. */
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_block_contact);
    /** Pressure relaxation algorithm by using position verlet time stepping. */
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_block_contact);
    /* Time step size without considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step_size(water_block, U_f);
    /** Time step size with considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step_size(water_block);
    /** Computing viscous acceleration. */
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_block_contact);
    /** Impose transport velocity. */
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>>
        transport_velocity_correction(water_block_inner, water_block_contact);
    
    BodyAlignedBoxByCell inflow_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform
    (Rotation3d(std::acos(Eigen::Vector3d::UnitX().dot(normal_vector_inlet)), vector_inlet),Vecd(buffer_translation_inlet)),buffer_halfsize_inlet));
    fluid_dynamics::BidirectionalBuffer<fluid_dynamics::NonPrescribedPressure> inflow_injection(inflow_emitter, in_outlet_particle_buffer);
    
    BodyAlignedBoxByCell outflow_emitter_1(water_block, makeShared<AlignedBoxShape>(xAxis, Transform
    (Rotation3d(std::acos(Eigen::Vector3d::UnitX().dot(-normal_vector_1)), -vector_1),(buffer_translation_1)),buffer_halfsize_1));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_1(outflow_emitter_1, in_outlet_particle_buffer);

    BodyAlignedBoxByCell outflow_emitter_2(water_block, makeShared<AlignedBoxShape>(xAxis, Transform
    (Rotation3d(std::acos(Eigen::Vector3d::UnitX().dot(-normal_vector_2)), -vector_2), (buffer_translation_2)), buffer_halfsize_2));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_2(outflow_emitter_2, in_outlet_particle_buffer);

    BodyAlignedBoxByCell outflow_emitter_3(water_block, makeShared<AlignedBoxShape>(xAxis, Transform
    (Rotation3d(std::acos(Eigen::Vector3d::UnitX().dot(-normal_vector_3)), -vector_3), (buffer_translation_3)), buffer_halfsize_3));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_3(outflow_emitter_3, in_outlet_particle_buffer);

    BodyAlignedBoxByCell outflow_emitter_4(water_block, makeShared<AlignedBoxShape>(xAxis, Transform
    (Rotation3d(std::acos(Eigen::Vector3d::UnitX().dot(-normal_vector_4)), -vector_4), (buffer_translation_4)), buffer_halfsize_4));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_4(outflow_emitter_4, in_outlet_particle_buffer);

    BodyAlignedBoxByCell outflow_emitter_5(water_block, makeShared<AlignedBoxShape>(xAxis, Transform
    (Rotation3d(std::acos(Eigen::Vector3d::UnitX().dot(normal_vector_5)), vector_5), (buffer_translation_5)), buffer_halfsize_5));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_5(outflow_emitter_5, in_outlet_particle_buffer);
    
    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_block_contact);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> emitter_buffer_inflow_condition(inflow_emitter);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition1(outflow_emitter_1);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition2(outflow_emitter_2);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition3(outflow_emitter_3);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition4(outflow_emitter_4);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition5(outflow_emitter_5);
    /**
     * @brief Output.
     */
    //----------------------------------------------------------------------
    //	Define the configuration related particles dynamics.
    //----------------------------------------------------------------------
    ParticleSorting particle_sorting(water_block);
    /** Output the body states. */
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "PositionDivergence");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");

    RegressionTestDynamicTimeWarping<ObservedQuantityRecording<Vecd>>
        write_centerline_velocity("Velocity", velocity_observer_contact);
    /**
     * @brief Setup geometry and initial conditions.
     */
    sph_system.initializeSystemCellLinkedLists(); 
    sph_system.initializeSystemConfigurations();
    free_stream_surface_indicator.exec();
    inflow_injection.tag_buffer_particles.exec();
    outflow_injection_1.tag_buffer_particles.exec();
    outflow_injection_2.tag_buffer_particles.exec();
    outflow_injection_3.tag_buffer_particles.exec();
    outflow_injection_4.tag_buffer_particles.exec();
    outflow_injection_5.tag_buffer_particles.exec();

    wall_boundary_normal_direction.exec();
    
    /**
     * @brief 	Basic parameters.
     */
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    size_t number_of_iterations = 0.0;
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 2.0;   /**< End time. */
    Real Output_Time = 0.01; /**< Time stamps for output of body states. */
    Real dt = 0.0;          /**< Default acoustic time step sizes. */
    /** statistics for computing CPU time. */
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    TimeInterval interval_computing_time_step;
    TimeInterval interval_computing_pressure_relaxation;
    TimeInterval interval_updating_configuration;
    TickCount time_instance;
    //Real accumulated_time = 0.004;
    Real accumulated_time = 0.0001;
    int updateP_n = 0;

    /** Output the start states of bodies. */
    body_states_recording.writeToFile(0);
    write_centerline_velocity.writeToFile(number_of_iterations);

    // why -average_Q??
    //outflow_pressure_condition1.getTargetPressure()->setWindkesselParams(4.0E8, 3.0E-10, 4.05E9, accumulated_time, 2.8E-6);
    //outflow_pressure_condition2.getTargetPressure()->setWindkesselParams(5.35E8, 2.25E-10, 5.41E9, accumulated_time, 2.09E-6);
    //outflow_pressure_condition3.getTargetPressure()->setWindkesselParams(2.1E8, 5.73E-10, 2.12E9, accumulated_time, 5.33E-6);
    //outflow_pressure_condition4.getTargetPressure()->setWindkesselParams(1.45E8, 8.31E-10, 1.46E9, accumulated_time, 7.74E-6);
    //outflow_pressure_condition5.getTargetPressure()->setWindkesselParams(1.76E7, 6.83E-9, 1.78E8, accumulated_time, 6.36E-5);


    outflow_pressure_condition1.getTargetPressure()->setWindkesselParams(4.0E8, 3.0E-10, 4.05E9, accumulated_time, 0);
    outflow_pressure_condition2.getTargetPressure()->setWindkesselParams(5.35E8, 2.25E-10, 5.41E9, accumulated_time, 0);
    outflow_pressure_condition3.getTargetPressure()->setWindkesselParams(2.1E8, 5.73E-10, 2.12E9, accumulated_time, 0);
    outflow_pressure_condition4.getTargetPressure()->setWindkesselParams(1.45E8, 8.31E-10, 1.46E9, accumulated_time, 0);
    outflow_pressure_condition5.getTargetPressure()->setWindkesselParams(1.76E7, 6.83E-9, 1.78E8, accumulated_time, 0);
    /**
     * @brief 	Main loop starts here.
    */
    while (physical_time < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < Output_Time)
        {  
            time_instance = TickCount::now();
            Real Dt = get_fluid_advection_time_step_size.exec();          
            update_fluid_density.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();
            interval_computing_time_step += TickCount::now() - time_instance;
            /** Dynamics including pressure relaxation. */
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);
                pressure_relaxation.exec(dt);
                kernel_summation.exec();

                emitter_buffer_inflow_condition.exec();  

                // windkessel model implementation

                if (physical_time >= updateP_n * accumulated_time)
                {
                    outflow_pressure_condition1.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition2.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition3.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition4.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition5.getTargetPressure()->updateNextPressure();
                   
                    ++updateP_n;
                }

                outflow_pressure_condition1.exec(dt); 
                outflow_pressure_condition2.exec(dt);
                outflow_pressure_condition3.exec(dt); 
                outflow_pressure_condition4.exec(dt); 
                outflow_pressure_condition5.exec(dt);    

                density_relaxation.exec(dt);
                
                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
            }
            interval_computing_pressure_relaxation += TickCount::now() - time_instance;
            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9)
                          << "N=" << number_of_iterations
                          << "	Time = " << physical_time
                          << "	Dt = " << Dt << "	dt = " << dt << "\n";

                if (number_of_iterations % observation_sample_interval == 0 && number_of_iterations != sph_system.RestartStep())
                {
                    write_centerline_velocity.writeToFile(number_of_iterations);
                }
            }
            number_of_iterations++;
            /** Update cell linked list and configuration. */
            time_instance = TickCount::now();
            /** Water block configuration and periodic condition. */
            inflow_injection.injection.exec();
            outflow_injection_1.injection.exec();
            outflow_injection_2.injection.exec();
            outflow_injection_3.injection.exec();
            outflow_injection_4.injection.exec();
            outflow_injection_5.injection.exec();

            inflow_injection.deletion.exec();
            outflow_injection_1.deletion.exec();
            outflow_injection_2.deletion.exec();
            outflow_injection_3.deletion.exec();
            outflow_injection_4.deletion.exec();
            outflow_injection_5.deletion.exec();

            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                particle_sorting.exec();
            }

            water_block.updateCellLinkedList();
            water_block_contact.updateConfiguration();
            water_block_complex.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;
            free_stream_surface_indicator.exec();

            inflow_injection.tag_buffer_particles.exec();
            outflow_injection_1.tag_buffer_particles.exec();
            outflow_injection_2.tag_buffer_particles.exec();
            outflow_injection_3.tag_buffer_particles.exec();
            outflow_injection_4.tag_buffer_particles.exec();
            outflow_injection_5.tag_buffer_particles.exec();
        }
        TickCount t2 = TickCount::now();
        body_states_recording.writeToFile();  
        velocity_observer_contact.updateConfiguration();
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