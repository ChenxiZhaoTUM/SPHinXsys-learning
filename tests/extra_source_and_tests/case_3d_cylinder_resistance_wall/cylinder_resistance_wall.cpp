/**
 * @file 	case_3d_cylinder_VIPO_shell.cpp
 * @brief 
 * @details
 * @author 
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
Real scale = 0.001;
Real diameter = 6.35 * scale;
Real fluid_radius = 0.5 * diameter;
Real full_length = 10 * fluid_radius;
//----------------------------------------------------------------------
//	Geometry parameters for shell.
//----------------------------------------------------------------------
int number_of_particles = 10;
Real resolution_ref = diameter / number_of_particles;
Real resolution_wall = 0.5 * resolution_ref;
Real wall_thickness = resolution_wall * 4.0;
int SimTK_resolution = 20;
Vec3d translation_fluid(full_length * 0.5, 0., 0.);

//----------------------------------------------------------------------
//	Geometry parameters for boundary condition.
//----------------------------------------------------------------------
Vec3d emitter_halfsize(resolution_ref * 2, fluid_radius, fluid_radius);
Vec3d emitter_translation(resolution_ref * 2, 0., 0.);
Vec3d disposer_halfsize(resolution_ref * 2, fluid_radius * 1.1, fluid_radius * 1.1);
Vec3d disposer_translation(full_length - disposer_halfsize[0], 0., 0.);

//----------------------------------------------------------------------
//	Domain bounds of the system.
//----------------------------------------------------------------------
BoundingBox system_domain_bounds(Vec3d(0, -0.5 * diameter, -0.5 * diameter) - Vec3d(wall_thickness, wall_thickness, wall_thickness),
                                 Vec3d(full_length, 0.5 * diameter, 0.5 * diameter) + Vec3d(wall_thickness, wall_thickness, wall_thickness));
//----------------------------------------------------------------------
//	Material parameters.
//----------------------------------------------------------------------
Real Outlet_pressure = 0;
Real rho0_f = 1060.0; /**< Reference density of fluid. */
Real mu_f = 0.00355;   /**< Viscosity. */
/**< Characteristic velocity. Average velocity */
Real U_f = 0.5;
Real U_max = 2.0 * U_f;  // parabolic inflow, Thus U_max = 2*U_f
Real c_f = 10.0 * U_max; /**< Reference sound speed. */

StdVec<Vecd> createAxialObservationPoints(
    double full_length, Vec3d translation = Vec3d(0.0, 0.0, 0.0))
{
    StdVec<Vecd> observation_points;
    int nx = 51;
    for (int i = 0; i < nx; i++)
    {
        double x = full_length / (nx - 1) * i;
        Vec3d point_coordinate(x, 0.0, 0.0);
        observation_points.emplace_back(point_coordinate + translation);
    }
    return observation_points;
};

StdVec<Vecd> createRadialObservationPoints(
    double full_length, double diameter, int number_of_particles,
    Vec3d translation = Vec3d(0.0, 0.0, 0.0))
{
    StdVec<Vecd> observation_points;
    int n = number_of_particles + 1;
    double x = full_length / 2.0;
    for (int i = 0; i < n - 1; i++) // we leave out the point close to the boundary as the
                                    // interpolation there is incorrect
                                    // TODO: fix the interpolation
    {
        double z = diameter / 2.0 * i / double(n);
        observation_points.emplace_back(Vec3d(x, 0.0, z) + translation);
        observation_points.emplace_back(Vec3d(x, 0.0, -z) + translation);
    }
    return observation_points;
};
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
struct InflowVelocity
{
    Real u_ave;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ave(0.0) {}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = velocity;
        Real run_time = GlobalStaticVariables::physical_time_;

        u_ave = 0.2339;
        Real a[8] = {-0.0176, -0.0657, -0.0280, 0.0068, 0.0075, 0.0115, 0.0040, 0.0035};
        Real b[8] = {0.1205, 0.0171, -0.0384, -0.0152, -0.0122, 0.0002, 0.0033, 0.0060};
        Real w = 2 * Pi / 1;
        for (size_t i = 0; i < 8; i++)
        {
            u_ave = SMAX(u_ave + a[i] * cos(w * (i + 1) * run_time) + b[i] * sin(w * (i + 1) * run_time),
                         0.0);
        }

        target_velocity[0] = SMAX(2.0 * u_ave * (1.0 - (position[1] * position[1] + position[2] * position[2]) / fluid_radius / fluid_radius),
                                  1.0e-2);
        target_velocity[1] = 0.0;
        target_velocity[2] = 0.0;

        return target_velocity;
    }
};

int main(int ac, char *av[])
{
    
    //----------------------------------------------------------------------
    //  Define water shape
    //----------------------------------------------------------------------
    auto water_block_shape = makeShared<ComplexShape>("WaterBody");
    water_block_shape->add<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), fluid_radius,
                                                      full_length * 0.5, SimTK_resolution,
                                                      translation_fluid);
    auto wall_shape = makeShared<ComplexShape>("WallBody");
    wall_shape->add<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), fluid_radius + wall_thickness,
                                               full_length * 0.5 + resolution_wall, SimTK_resolution,
                                                      translation_fluid);
    wall_shape->subtract<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), fluid_radius,
                                               full_length * 0.7, SimTK_resolution,
                                               translation_fluid);
    //----------------------------------------------------------------------
    //  Build up -- a SPHSystem --
    //----------------------------------------------------------------------
    SPHSystem system(system_domain_bounds, resolution_ref);
    system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    system.setReloadParticles(true);       // Tag for computation with save particles distribution
#ifdef BOOST_AVAILABLE
    system.handleCommandlineOptions(ac, av); // handle command line arguments
#endif
    IOEnvironment io_environment(system);
    //----------------------------------------------------------------------
    //	Creating bodies with corresponding materials and particles.
    //----------------------------------------------------------------------
    FluidBody water_block(system, water_block_shape);
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    water_block.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(system);
    (!system.RunParticleRelaxation() && system.ReloadParticles())
        ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
        : water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);

    SolidBody wall_boundary(system, wall_shape);
    wall_boundary.defineAdaptation<SPH::SPHAdaptation>(1.15, resolution_ref / resolution_wall);
    wall_boundary.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(system);
    //wall_boundary.defineMaterial<LinearElasticSolid>(1, 1e3, 0.45);
    wall_boundary.defineMaterial<Solid>();
    (!system.RunParticleRelaxation() && system.ReloadParticles())
        ? wall_boundary.generateParticles<BaseParticles, Reload>("WallBody")
        : wall_boundary.generateParticles<BaseParticles, Lattice>();

    ObserverBody observer_axial(system, "fluid_observer_axial");
    observer_axial.generateParticles<ObserverParticles>(createAxialObservationPoints(full_length));
    ObserverBody observer_radial(system, "fluid_observer_radial");
    observer_radial.generateParticles<ObserverParticles>(createRadialObservationPoints(full_length, diameter, number_of_particles));
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (system.RunParticleRelaxation() && !system.ReloadParticles())
    {
        InnerRelation water_block_inner(water_block);
        InnerRelation wall_boundary_inner(wall_boundary);
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
        SimpleDynamics<RandomizeParticlePosition> random_particles(wall_boundary);
        RelaxationStepInner relaxation_step_water_inner(water_block_inner);
        RelaxationStepInner relaxation_step_inner(wall_boundary_inner);
        //----------------------------------------------------------------------
        //	Relaxation output
        //----------------------------------------------------------------------
        BodyStatesRecordingToVtp write_body_state_to_vtp(system);
        ReloadParticleIO write_particle_reload_files({ &water_block, &wall_boundary });
        //----------------------------------------------------------------------
        //	Physics relaxation starts here.
        //----------------------------------------------------------------------
        random_water_particles.exec(0.25);
        random_particles.exec(0.25);
        relaxation_step_water_inner.SurfaceBounding().exec();
        relaxation_step_inner.SurfaceBounding().exec();
        write_body_state_to_vtp.writeToFile(0.0);
        //----------------------------------------------------------------------
        // From here the time stepping begins.
        //----------------------------------------------------------------------
        int ite = 0;
        int relax_step = 1000;
        while (ite < relax_step)
        {
            relaxation_step_water_inner.exec();
            relaxation_step_inner.exec();
            ite++;
            if (ite % 250 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
                write_body_state_to_vtp.writeToFile(ite);
            }
        }
        write_particle_reload_files.writeToFile(0);
        std::cout << "The physics relaxation process of imported model finish !" << std::endl;
        return 0;
    }
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //----------------------------------------------------------------------
    InnerRelation water_block_inner(water_block);
    InnerRelation wall_boundary_inner(wall_boundary);
    ContactRelation water_block_contact(water_block, {&wall_boundary});
    ContactRelation observer_contact_axial(observer_axial, {&water_block});
    ContactRelation observer_contact_radial(observer_radial, {&water_block});
    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    // which is only used for update configuration.
    //----------------------------------------------------------------------
    ComplexRelation water_block_complex(water_block_inner, water_block_contact);

    //----------------------------------------------------------------------
    // Define the numerical methods used in the simulation.
    // Note that there may be data dependence on the sequence of constructions.
    // Generally, the geometric models or simple objects without data dependencies,
    // such as gravity, should be initiated first.
    // Then the major physical particle dynamics model should be introduced.
    // Finally, the auxillary models such as time step estimator, initial condition,
    // boundary condition and other constraints should be defined.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
    InteractionWithUpdate<LinearGradientCorrectionMatrixInner> wall_corrected_configuration(wall_boundary_inner);
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_block_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_block_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_block_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_block_contact);

    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_max);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    //----------------------------------------------------------------------
    //	Boundary conditions.
    //----------------------------------------------------------------------
    BodyAlignedBoxByCell left_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vec3d(emitter_translation)), emitter_halfsize));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_disposer_outflow_deletion(left_disposer);
    BodyAlignedBoxByCell right_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(disposer_translation)), disposer_halfsize));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> right_disposer_outflow_deletion(right_disposer, "outlet");

    BodyAlignedBoxByCell left_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(emitter_translation)), emitter_halfsize));
    fluid_dynamics::BidirectionalBuffer<fluid_dynamics::NonPrescribedPressure> left_bidirection_buffer(left_buffer, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vec3d(disposer_translation)), disposer_halfsize));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer right_bidirection_buffer(right_buffer, "outlet", in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_block_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<fluid_dynamics::NonPrescribedPressure>> left_pressure_condition(left_buffer);
    SimpleDynamics<fluid_dynamics::ResistanceBoundaryCondition> right_pressure_condition(right_buffer, "outlet");
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_buffer);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(system);
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    body_states_recording.addToWrite<Vecd>(wall_boundary, "NormalDirection");
    ObservedQuantityRecording<Vec3d> write_fluid_velocity_axial("Velocity", observer_contact_axial);
    ObservedQuantityRecording<Vec3d> write_fluid_velocity_radial("Velocity", observer_contact_radial);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    system.initializeSystemCellLinkedLists();
    system.initializeSystemConfigurations();
    boundary_indicator.exec();
    left_bidirection_buffer.tag_buffer_particles.exec();
    right_bidirection_buffer.tag_buffer_particles.exec();
    wall_boundary_normal_direction.exec();
    wall_corrected_configuration.exec();
    water_block_complex.updateConfiguration();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = 0;
    int screen_output_interval = 100;
    Real end_time = 2.0;               /**< End time. */
    Real Output_Time = end_time / 100; /**< Time stamps for output of body states. */
    Real dt = 0.0;                     /**< Default acoustic time step sizes. */

    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    TimeInterval interval_computing_time_step;
    TimeInterval interval_computing_pressure_relaxation;
    TimeInterval interval_updating_configuration;
    TickCount time_instance;
    Real accumulated_time = 0.02;
    int updateP_n = 0;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    body_states_recording.writeToFile(0);
    right_pressure_condition.getTargetPressure()->setWindkesselParams(2.05E8, accumulated_time, 0);
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < Output_Time)
        {
            /** Acceleration due to viscous force and gravity. */
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
                dt = SMIN(get_fluid_time_step_size.exec(),
                          Dt - relaxation_time);
                pressure_relaxation.exec(dt);

                // boundary condition implementation
                kernel_summation.exec();
                //left_pressure_condition.exec(dt);
                inflow_velocity_condition.exec();
                
                if (GlobalStaticVariables::physical_time_ >= updateP_n * accumulated_time)
                {
                    right_pressure_condition.getTargetPressure()->updateNextPressure();

                    ++updateP_n;
                }
                right_pressure_condition.exec(dt);
                
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
            }
            number_of_iterations++;

            time_instance = TickCount::now();
            /** Water block configuration and periodic condition. */
            left_bidirection_buffer.injection.exec();
            right_bidirection_buffer.injection.exec();
            left_disposer_outflow_deletion.exec();
            right_disposer_outflow_deletion.exec();

            water_block.updateCellLinkedList();
            water_block_complex.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;
            boundary_indicator.exec();

            left_bidirection_buffer.tag_buffer_particles.exec();
            right_bidirection_buffer.tag_buffer_particles.exec();
        }
        TickCount t2 = TickCount::now();
        body_states_recording.writeToFile();
        TickCount t3 = TickCount::now();
        interval += t3 - t2;

        /** Update observer and write output of observer. */
        observer_contact_axial.updateConfiguration();
        observer_contact_radial.updateConfiguration();
        write_fluid_velocity_axial.writeToFile(number_of_iterations);
        write_fluid_velocity_radial.writeToFile(number_of_iterations);
    }
    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds()
              << " seconds." << std::endl;
    std::cout << std::fixed << std::setprecision(9)
              << "interval_computing_time_step ="
              << interval_computing_time_step.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9)
              << "interval_computing_pressure_relaxation = "
              << interval_computing_pressure_relaxation.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9)
              << "interval_updating_configuration = "
              << interval_updating_configuration.seconds() << "\n";
    return 0;
}