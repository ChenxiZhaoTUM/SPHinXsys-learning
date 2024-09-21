/**
 * @file 	poiseuille_flow_shell.cpp
 * @brief 	3D poiseuille flow interaction with shell example
 * @details This is the one of the basic test cases for validating fluid-rigid shell interaction
 * @author  Weiyi Kong
 */
#include "sphinxsys.h"
#include <gtest/gtest.h>
using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_stl_file_fluid = "./input/water.stl";

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
Real resolution_shell = 0.5 * resolution_ref;
Real shell_thickness = 0.5 * resolution_shell;
Vec3d translation_fluid(full_length * 0.5, 0., 0.);
//----------------------------------------------------------------------
//	Geometry parameters for boundary condition.
//----------------------------------------------------------------------
Vec3d emitter_halfsize(resolution_ref * 3, fluid_radius, fluid_radius);
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
Vec3d translation(0., 0., 0.);
class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        /** Geometry definition. */
        add<TriangleMeshShapeSTL>(full_path_to_stl_file_fluid, translation, scale);
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
//----------------------------------------------------------------------
//	Material parameters of the shell
//----------------------------------------------------------------------
Real rho0_s = 1120;                /** Normalized density. */
Real Youngs_modulus = 1.08e5;    /** Normalized Youngs Modulus. */
Real poisson = 0.3;               /** Poisson ratio. */
Real physical_viscosity = 0.25 * sqrt(rho0_s * Youngs_modulus) * full_length * scale;

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

class ShellBoundary;
template <>
class ParticleGenerator<SurfaceParticles, ShellBoundary> : public ParticleGenerator<SurfaceParticles>
{
    Real resolution_shell_;
    Real wall_thickness_;
    Real shell_thickness_;

  public:
    explicit ParticleGenerator(SPHBody &sph_body, SurfaceParticles &surface_particles,
                               Real resolution_shell, Real wall_thickness, Real shell_thickness)
        : ParticleGenerator<SurfaceParticles>(sph_body, surface_particles),
          resolution_shell_(resolution_shell),
          wall_thickness_(wall_thickness), shell_thickness_(shell_thickness){};
    void prepareGeometricData() override
    {
        const Real radius_mid_surface = fluid_radius + resolution_shell_ * 0.5;
        const auto particle_number_mid_surface =
            int(2.0 * radius_mid_surface * Pi / resolution_shell_);
        const auto particle_number_height =
            int((full_length + 2.0 * wall_thickness_) / resolution_shell_);
        for (int i = 0; i < particle_number_mid_surface; i++)
        {
            for (int j = 0; j < particle_number_height; j++)
            {
                Real theta = (i + 0.5) * 2 * Pi / (Real)particle_number_mid_surface;
                Real x = -wall_thickness_ + (full_length + 2 * wall_thickness_) * j / (Real)particle_number_height + 0.5 * resolution_shell_;
                Real y = radius_mid_surface * cos(theta);
                Real z = radius_mid_surface * sin(theta);
                addPositionAndVolumetricMeasure(Vec3d(x, y, z),
                                                resolution_shell_ * resolution_shell_);
                Vec3d n_0 = Vec3d(0.0, y / radius_mid_surface, z / radius_mid_surface);
                addSurfaceProperties(n_0, shell_thickness_);
            }
        }
    }
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
        if (base_particles_.ParticlePositions()[index_i][0] < constrain_len_ || base_particles_.ParticlePositions()[index_i][0] > full_length - constrain_len_)
        {
            body_part_particles_.push_back(index_i);
        }
    };
};

//----------------------------------------------------------------------
//	Test case function
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //  Build up -- a SPHSystem --
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.setGenerateRegressionData(false);
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

    SolidBody shell_boundary(sph_system, makeShared<DefaultShape>("Shell"));
    shell_boundary.defineAdaptation<SPH::SPHAdaptation>(1.15, resolution_ref / resolution_shell);
    shell_boundary.defineMaterial<SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
    //shell_boundary.defineMaterial<LinearElasticSolid>(1, 1e3, 0.45);
    //shell_boundary.defineMaterial<Solid>();
    shell_boundary.generateParticles<SurfaceParticles, ShellBoundary>(resolution_shell, wall_thickness, shell_thickness);

    ObserverBody observer_axial(sph_system, "fluid_observer_axial");
    observer_axial.generateParticles<ObserverParticles>(createAxialObservationPoints(full_length));
    ObserverBody observer_radial(sph_system, "fluid_observer_radial");
    observer_radial.generateParticles<ObserverParticles>(createRadialObservationPoints(full_length, diameter, number_of_particles));

    if (sph_system.RunParticleRelaxation())
    {
        /** body topology only for particle relaxation */
        InnerRelation water_block_inner(water_block);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** Random reset the insert body particle position. */
        SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_body_to_vtp({sph_system});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files({&water_block });
        /** A  Physics relaxation step. */
        RelaxationStepInner relaxation_step_water_inner(water_block_inner);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_water_particles.exec(0.25);
        relaxation_step_water_inner.SurfaceBounding().exec();
        write_body_to_vtp.writeToFile(0);

        int ite_p = 0;
        while (ite_p < 5000)
        {
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
    InnerRelation shell_boundary_inner(shell_boundary);
    ShellInnerRelationWithContactKernel shell_curvature_inner(shell_boundary, water_block);
    // shell normal should point from fluid to shell
    // normal corrector set to false if shell normal is already pointing from fluid to shell
    ContactRelationFromShellToFluid water_shell_contact(water_block, {&shell_boundary}, {false});
    ContactRelationFromFluidToShell shell_water_contact(shell_boundary, {&water_block}, {false});
    ContactRelation observer_contact_axial(observer_axial, {&water_block});
    ContactRelation observer_contact_radial(observer_radial, {&water_block});
    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    // which is only used for update configuration.
    //----------------------------------------------------------------------
    ComplexRelation water_block_complex(water_block_inner, water_shell_contact);

    //----------------------------------------------------------------------
    // Define the numerical methods used in the simulation.
    // Note that there may be data dependence on the sequence of constructions.
    // Generally, the geometric models or simple objects without data dependencies,
    // such as gravity, should be initiated first.
    // Then the major physical particle dynamics model should be introduced.
    // Finally, the auxillary models such as time step estimator, initial condition,
    // boundary condition and other constraints should be defined.
    //----------------------------------------------------------------------
    // shell dynamics
    InteractionDynamics<thin_structure_dynamics::ShellCorrectConfiguration> shell_corrected_configuration(shell_boundary_inner);
    Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationFirstHalf> shell_stress_relaxation_first(shell_boundary_inner, 3, true);
    Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationSecondHalf> shell_stress_relaxation_second(shell_boundary_inner);
    ReduceDynamics<thin_structure_dynamics::ShellAcousticTimeStepSize> shell_time_step_size(shell_boundary);
    SimpleDynamics<thin_structure_dynamics::AverageShellCurvature> shell_curvature(shell_curvature_inner);
    SimpleDynamics<thin_structure_dynamics::UpdateShellNormalDirection> shell_update_normal(shell_boundary);
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseInner<Vec3d, FixedDampingRate>>>
        shell_velocity_damping(0.2, shell_boundary_inner, "Velocity", physical_viscosity);
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseInner<Vec3d, FixedDampingRate>>>
        shell_rotation_damping(0.2, shell_boundary_inner, "AngularVelocity", physical_viscosity);
 
    /** Exert constrain on shell. */
    BoundaryGeometry boundary_geometry(shell_boundary, "BoundaryGeometry", resolution_ref * 4);
    //SimpleDynamics<thin_structure_dynamics::ConstrainShellBodyRegion> constrain_holder(boundary_geometry);
    SimpleDynamics<FixBodyPartConstraint> constrain_holder(boundary_geometry);
    /*SimpleDynamics<solid_dynamics::SpringConstrain> constrain_holder(boundary_geometry, 0.2);
    SimpleDynamics<solid_dynamics::ConstrainSolidBodyMassCenter> constrain_mass_center(shell_boundary);*/


    // fluid dynamics
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallNoRiemann> density_relaxation(water_block_inner, water_shell_contact);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationFreeStreamComplex> update_density_by_summation(water_block_inner, water_shell_contact);
    //----------------------------------------------------------------------
    //	Boundary conditions. Inflow & Outflow in Y-direction
    //----------------------------------------------------------------------
    BodyAlignedBoxByParticle emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(emitter_translation)), emitter_halfsize));
    SimpleDynamics<fluid_dynamics::EmitterInflowInjection> emitter_inflow_injection(emitter, inlet_particle_buffer);
    BodyAlignedBoxByCell emitter_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(emitter_translation)), emitter_halfsize));
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> emitter_buffer_inflow_condition(emitter_buffer);
    BodyAlignedBoxByCell disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(disposer_translation)), disposer_halfsize));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> disposer_outflow_deletion(disposer);
    
    
    // FSI
    InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_on_shell(shell_water_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_on_shell(shell_water_contact);
    solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(shell_boundary);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<Vecd>(shell_boundary, "NormalDirection");
    body_states_recording.addToWrite<Vecd>(shell_boundary, "PressureForceFromFluid");
    body_states_recording.addToWrite<Real>(shell_boundary, "Average1stPrincipleCurvature");
    body_states_recording.addToWrite<Real>(shell_boundary, "Average2ndPrincipleCurvature");
    ObservedQuantityRecording<Vec3d> write_fluid_velocity_axial("Velocity", observer_contact_axial);
    ObservedQuantityRecording<Vec3d> write_fluid_velocity_radial("Velocity", observer_contact_radial);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    shell_corrected_configuration.exec();
    shell_curvature.exec();
    //constrain_holder.exec();
    water_block_complex.updateConfiguration();
    shell_water_contact.updateConfiguration();

    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    Real end_time = 2.0;               /**< End time. */
    Real Output_Time = end_time / 100; /**< Time stamps for output of body states. */
    Real dt = 0.0;                     /**< Default acoustic time step sizes. */
    Real dt_s = 0.0;                   /**< Default acoustic time step sizes for solid. */
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
            //std::cout << "Dt = " << Dt << std::endl;
            boundary_indicator.exec();
            update_density_by_summation.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();

            /** FSI for viscous force. */
            viscous_force_on_shell.exec();

            interval_computing_time_step += TickCount::now() - time_instance;
            /** Dynamics including pressure relaxation. */
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(),
                          Dt - relaxation_time);
                //std::cout << "dt = " << dt << std::endl;
                pressure_relaxation.exec(dt);

                /** FSI for pressure force. */
                pressure_force_on_shell.exec();

                //emitter_buffer_inflow_condition.exec();
                density_relaxation.exec(dt);

                Real dt_s_sum = 0.0;
                average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    dt_s = shell_time_step_size.exec();
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;
                    //std::cout << "dt_s = " << dt_s << std::endl;
                    shell_stress_relaxation_first.exec(dt_s);

                    constrain_holder.exec();
                    shell_velocity_damping.exec(dt);
                    shell_rotation_damping.exec(dt);
                    constrain_holder.exec();

                    shell_stress_relaxation_second.exec(dt_s);
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
            shell_update_normal.exec();
            shell_boundary.updateCellLinkedList();
            shell_curvature_inner.updateConfiguration();
            shell_curvature.exec();
            shell_water_contact.updateConfiguration();
            water_block_complex.updateConfiguration();
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
