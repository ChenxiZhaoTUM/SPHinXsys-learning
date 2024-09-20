/**
 * @file 	poiseuille_flow_shell.cpp
 * @brief 	3D poiseuille flow interaction with shell example
 * @details This is the one of the basic test cases for validating fluid-rigid shell interaction
 * @author  Weiyi Kong
 */
#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real scale = 0.001;
Real diameter = 6.35 * scale;
Real fluid_radius = 0.5 * diameter;
Real full_length = 10 * fluid_radius;
int number_of_particles = 10;
Real resolution_ref = diameter / number_of_particles;
int SimTK_resolution = 10;
//----------------------------------------------------------------------
//	Geometry parameters for shell.
//----------------------------------------------------------------------
Real wall_thickness = resolution_ref * 4.0;
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
BoundingBox system_domain_bounds(Vec3d(0, -0.5 * diameter, -0.5 * diameter) -
                                            Vec3d(wall_thickness, wall_thickness,
                                                    wall_thickness),
                                        Vec3d(full_length, 0.5 * diameter, 0.5 * diameter) +
                                            Vec3d(wall_thickness, wall_thickness,
                                                    wall_thickness));
//----------------------------------------------------------------------
//  Define body shapes
//----------------------------------------------------------------------
class WaterBlock : public ComplexShape
{
  public:
      explicit WaterBlock(const std::string& shape_name) : ComplexShape(shape_name)
      {
          //add<TransformShape<GeometricShapeCylinder>>(Transform(Vec3d(translation_fluid)), fluid_radius, full_length * 0.5);
          add<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), fluid_radius, full_length * 0.5, SimTK_resolution, translation_fluid);
      }
};

class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        /*add<TransformShape<GeometricShapeCylinder>>(Transform(Vec3d(translation_fluid)), fluid_radius + wall_thickness, full_length);
        subtract<TransformShape<GeometricShapeCylinder>>(Transform(Vec3d(translation_fluid)), fluid_radius, full_length);*/

        add<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), fluid_radius + wall_thickness, full_length * 0.5, SimTK_resolution, translation_fluid);
        subtract<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), fluid_radius, full_length * 0.5, SimTK_resolution, translation_fluid);

    }
};
//----------------------------------------------------------------------
//	Material parameters.
//----------------------------------------------------------------------
Real rho0_f = 1050.0; /**< Reference density of fluid. */
Real mu_f = 3.6e-3;   /**< Viscosity. */
Real Re = 100;
/**< Characteristic velocity. Average velocity */
Real U_f = Re * mu_f / rho0_f / diameter;
Real U_max = 2.0 * U_f;  // parabolic inflow, Thus U_max = 2*U_f
Real c_f = 10.0 * U_max; /**< Reference sound speed. */

Real rho0_s = 1120;                /** Normalized density. */
Real Youngs_modulus = 1.08e6;    /** Normalized Youngs Modulus. */
Real poisson = 0.49;               /** Poisson ratio. */


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
    Real u_ref_, t_ref_;
    AlignedBoxShape &aligned_box_;
    Vec3d halfsize_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ref_(U_f), t_ref_(1.0),
          aligned_box_(boundary_condition.getAlignedBox()),
          halfsize_(aligned_box_.HalfSize()) {}

    Vec3d operator()(Vec3d &position, Vec3d &velocity)
    {
        Vec3d target_velocity = Vec3d(0, 0, 0);
        target_velocity[0] = SMAX(2.0 * U_f *
                                      (1.0 - (position[1] * position[1] + position[2] * position[2]) /
                                                 fluid_radius / fluid_radius),
                                  0.0);

        return target_velocity;
    }
};

class BoundaryGeometry : public BodyPartByParticle
{
  public:
    BoundaryGeometry(SPHBody &body, const std::string &body_part_name, Real constrain_len)
        : BodyPartByParticle(body, body_part_name), constrain_len_(constrain_len)
    {
        TaggingParticleMethod tagging_particle_method = std::bind(&BoundaryGeometry::tagManually, this, _1);
        tagParticles(tagging_particle_method);
    };
    virtual ~BoundaryGeometry(){};

  private:
      Real constrain_len_;

    void tagManually(size_t index_i)
    {
        if (base_particles_.ParticlePositions()[index_i][1] < constrain_len_ 
            || base_particles_.ParticlePositions()[index_i][1] > full_length - constrain_len_)
        {
            body_part_particles_.push_back(index_i);
        }
    };
};

int main(int ac, char *av[])
{
    //std::cout << "U_f = " << U_f << std::endl; //U_f = 0.0539933

    //----------------------------------------------------------------------
    //  Build up -- a SPHSystem --
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(true);       // Tag for computation with save particles distribution
#ifdef BOOST_AVAILABLE
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment(); // handle command line arguments
#endif
    //----------------------------------------------------------------------
    //	Creating bodies with corresponding materials and particles.
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> inlet_particle_buffer(0.5);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(inlet_particle_buffer, water_block.getName())
        : water_block.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
    //wall_boundary.defineMaterial<LinearElasticSolid>(1, 1e3, 0.45);
    wall_boundary.defineMaterial<SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
    //wall_boundary.defineMaterial<Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_boundary.generateParticles<BaseParticles, Reload>(wall_boundary.getName())
        : wall_boundary.generateParticles<BaseParticles, Lattice>();

    ObserverBody observer_axial(sph_system, "fluid_observer_axial");
    observer_axial.generateParticles<ObserverParticles>(createAxialObservationPoints(full_length));
    ObserverBody observer_radial(sph_system, "fluid_observer_radial");
    observer_radial.generateParticles<ObserverParticles>(createRadialObservationPoints(full_length, diameter, number_of_particles));
    
    if (sph_system.RunParticleRelaxation())
    {
        /** body topology only for particle relaxation */
        InnerRelation wall_boundary_inner(wall_boundary);
        InnerRelation water_block_inner(water_block);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** Random reset the insert body particle position. */
        SimpleDynamics<RandomizeParticlePosition> random_body_particles(wall_boundary);
        SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_body_to_vtp({sph_system});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files({ &wall_boundary, &water_block });
        /** A  Physics relaxation step. */
        RelaxationStepInner relaxation_step_inner(wall_boundary_inner);
        RelaxationStepInner relaxation_step_water_inner(water_block_inner);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_body_particles.exec(0.25);
        random_water_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_water_inner.SurfaceBounding().exec();
        write_body_to_vtp.writeToFile(0);

        int ite_p = 0;
        while (ite_p < 5000)
        {
            relaxation_step_inner.exec();
            relaxation_step_water_inner.exec();
            ite_p += 1;
            if (ite_p % 500 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
                write_body_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of the cylinder finish !" << std::endl;

        /** Output results. */
        write_particle_reload_files.writeToFile(0);
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
    ContactRelation wall_water_contact(wall_boundary, {&water_block});
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
    // solid dynamics
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
    InteractionWithUpdate<LinearGradientCorrectionMatrixInner> wall_corrected_configuration(wall_boundary_inner);
    Dynamics1Level<solid_dynamics::Integration1stHalfPK2> wall_stress_relaxation_first_half(wall_boundary_inner);
    Dynamics1Level<solid_dynamics::Integration2ndHalf> wall_stress_relaxation_second_half(wall_boundary_inner);
    ReduceDynamics<solid_dynamics::AcousticTimeStepSize> wall_computing_time_step_size(wall_boundary);
    SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> wall_update_normal(wall_boundary);
    /** Exert constrain on wall. */
    BoundaryGeometry boundary_geometry(wall_boundary, "BoundaryGeometry", resolution_ref * 4);
    SimpleDynamics<FixBodyPartConstraint> constrain_holder(boundary_geometry);

    // fluid dynamics
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> inlet_outlet_surface_particle_indicator(water_block_inner, water_block_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_block_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallNoRiemann> density_relaxation(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationFreeStreamComplex> update_density_by_summation(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_block_contact);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_max);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    //	Boundary conditions. Inflow & Outflow in Y-direction
    BodyAlignedBoxByParticle emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(emitter_translation)), emitter_halfsize));
    SimpleDynamics<fluid_dynamics::EmitterInflowInjection> emitter_inflow_injection(emitter, inlet_particle_buffer);
    BodyAlignedBoxByCell emitter_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(emitter_translation)), emitter_halfsize));
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> emitter_buffer_inflow_condition(emitter_buffer);
    BodyAlignedBoxByCell disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(disposer_translation)), disposer_halfsize));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> disposer_outflow_deletion(disposer);

    // FSI
    InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_on_wall(wall_water_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_on_wall(wall_water_contact);
    solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(wall_boundary);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<Vecd>(wall_boundary, "NormalDirection");
    body_states_recording.addToWrite<Vecd>(wall_boundary, "PressureForceFromFluid");
    ObservedQuantityRecording<Vec3d> write_fluid_velocity_axial("Velocity", observer_contact_axial);
    ObservedQuantityRecording<Vec3d> write_fluid_velocity_radial("Velocity", observer_contact_radial);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    wall_boundary_normal_direction.exec();
    wall_corrected_configuration.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    Real end_time = 2.0;               /**< End time. */
    Real Output_Time = end_time / 100; /**< Time stamps for output of body states. */
    Real dt = 0.0;                     /**< Default acoustic time step sizes. */
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
    body_states_recording.writeToFile(0);

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
            inlet_outlet_surface_particle_indicator.exec();
            update_density_by_summation.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();
            /** FSI for viscous force. */
            viscous_force_on_wall.exec();

            interval_computing_time_step += TickCount::now() - time_instance;
            /** Dynamics including pressure relaxation. */
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(),
                          Dt - relaxation_time);
                /** Fluid pressure relaxation */
                pressure_relaxation.exec(dt);
                /** FSI for pressure force. */
                pressure_force_on_wall.exec();
                /** Fluid density relaxation */
                density_relaxation.exec(dt);

                Real dt_s_sum = 0.0;
                average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    dt_s = wall_computing_time_step_size.exec();
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;
                    wall_stress_relaxation_first_half.exec(dt_s);

                    constrain_holder.exec(dt_s);

                    wall_stress_relaxation_second_half.exec(dt_s);
                    dt_s_sum += dt_s;

                    //body_states_recording.writeToFile();
                }
                average_velocity_and_acceleration.update_averages_.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
                emitter_buffer_inflow_condition.exec();
            }
            interval_computing_pressure_relaxation +=
                TickCount::now() - time_instance;
            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << GlobalStaticVariables::physical_time_
                          << "	Dt = " << Dt << "	dt = " << dt << "	dt_s = " << dt_s << "\n";
            }
            number_of_iterations++;

            time_instance = TickCount::now();
            /** Water block configuration and periodic condition. */
            emitter_inflow_injection.exec();
            disposer_outflow_deletion.exec();

            /** Update cell linked list and configuration. */
            water_block.updateCellLinkedListWithParticleSort(100);
            wall_update_normal.exec();
            wall_boundary.updateCellLinkedList();
            water_block_complex.updateConfiguration();
            wall_water_contact.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;
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
