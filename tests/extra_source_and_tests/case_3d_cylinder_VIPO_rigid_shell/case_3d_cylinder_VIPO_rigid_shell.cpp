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
Real resolution_shell = 0.5 * resolution_ref;
Real shell_thickness = 0.5 * resolution_shell;
Real inflow_length = resolution_ref * 10.0; // Inflow region
//Real wall_thickness = resolution_ref * 4.0;
Real wall_thickness = 0;
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
BoundingBox system_domain_bounds(Vec3d(0, -0.5 * diameter, -0.5 * diameter) - Vec3d(wall_thickness, shell_thickness, shell_thickness),
                                 Vec3d(full_length, 0.5 * diameter, 0.5 * diameter) + Vec3d(wall_thickness, shell_thickness, shell_thickness));
//----------------------------------------------------------------------
//	Material parameters.
//----------------------------------------------------------------------
Real Outlet_pressure = 0;
Real rho0_f = 1060.0; /**< Reference density of fluid. */
Real mu_f = 0.00355;   /**< Viscosity. */
Real Re = 100;
/**< Characteristic velocity. Average velocity */
Real U_f = Re * mu_f / rho0_f / diameter;
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
        Real radius_mid_surface = fluid_radius + resolution_shell_ * 0.5;
        auto particle_number_mid_surface =
            int(2.0 * radius_mid_surface * Pi / resolution_shell_);
        auto particle_number_height =
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

    Vec3d operator()(Vec3d &position, Vec3d &velocity, Real current_time)
    {
        Vec3d target_velocity = Vec3d(0, 0, 0);
        target_velocity[0] = SMAX(2.0 * U_f * (1.0 - (position[1] * position[1] + position[2] * position[2]) / fluid_radius / fluid_radius),
                                  0.0);
        return target_velocity;
    }
};
//----------------------------------------------------------------------
//	Pressure boundary condition.
//----------------------------------------------------------------------
struct LeftInflowPressure
{
    template <class BoundaryConditionType>
    LeftInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real p, Real physical_time)
    {
        return p;
    }
};

struct RightOutflowPressure
{
    template <class BoundaryConditionType>
    RightOutflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real p, Real physical_time)
    {
        /*constant pressure*/
        Real pressure = Outlet_pressure;
        return pressure;
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

    //----------------------------------------------------------------------
    //  Build up -- a SPHSystem --
    //----------------------------------------------------------------------
    SPHSystem system(system_domain_bounds, resolution_ref);
    IOEnvironment io_environment(system);

    //----------------------------------------------------------------------
    //	Creating bodies with corresponding materials and particles.
    //----------------------------------------------------------------------
    FluidBody water_block(system, water_block_shape);
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);

    SolidBody shell_boundary(system, makeShared<DefaultShape>("Shell"));
    shell_boundary.defineAdaptation<SPH::SPHAdaptation>(1.15, resolution_ref / resolution_shell);
    shell_boundary.defineMaterial<LinearElasticSolid>(1, 1e3, 0.45);
    shell_boundary.generateParticles<SurfaceParticles, ShellBoundary>(resolution_shell, wall_thickness, shell_thickness);

    ObserverBody observer_axial(system, "fluid_observer_axial");
    observer_axial.generateParticles<ObserverParticles>(createAxialObservationPoints(full_length));
    ObserverBody observer_radial(system, "fluid_observer_radial");
    observer_radial.generateParticles<ObserverParticles>(createRadialObservationPoints(full_length, diameter, number_of_particles));
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //----------------------------------------------------------------------
    InnerRelation water_block_inner(water_block);
    InnerRelation shell_boundary_inner(shell_boundary);
    ShellInnerRelationWithContactKernel wall_curvature_inner(shell_boundary, water_block);
    // shell normal should point from fluid to shell
    // normal corrector set to false if shell normal is already pointing from fluid to shell
    ContactRelationFromShellToFluid water_shell_contact(water_block, {&shell_boundary}, {false});
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
    InteractionDynamics<thin_structure_dynamics::ShellCorrectConfiguration> wall_corrected_configuration(shell_boundary_inner);
    SimpleDynamics<thin_structure_dynamics::AverageShellCurvature> shell_curvature(wall_curvature_inner);

    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_shell_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_shell_contact);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step_size(water_block, U_max);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step_size(water_block);
    //----------------------------------------------------------------------
    //	Boundary conditions.
    //----------------------------------------------------------------------
    BodyAlignedBoxByCell left_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(emitter_translation)), emitter_halfsize));
    fluid_dynamics::BidirectionalBuffer<LeftInflowPressure> left_bidirection_buffer(left_buffer, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vec3d(disposer_translation)), disposer_halfsize));
    fluid_dynamics::BidirectionalBuffer<RightOutflowPressure> right_bidirection_buffer(right_buffer, in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_shell_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<LeftInflowPressure>> left_pressure_condition(left_buffer);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightOutflowPressure>> right_pressure_condition(right_buffer);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_buffer);
    //----------------------------------------------------------------------
    //	Define the configuration related particles dynamics.
    //----------------------------------------------------------------------
    ParticleSorting particle_sorting(water_block);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(system);
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    body_states_recording.addToWrite<Vecd>(shell_boundary, "NormalDirection");
    body_states_recording.addToWrite<Real>(shell_boundary, "Average1stPrincipleCurvature");
    body_states_recording.addToWrite<Real>(shell_boundary, "Average2ndPrincipleCurvature");
    ObservedQuantityRecording<Vec3d> write_fluid_velocity_axial("Velocity", observer_contact_axial);
    ObservedQuantityRecording<Vec3d> write_fluid_velocity_radial("Velocity", observer_contact_radial);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    system.initializeSystemCellLinkedLists();
    system.initializeSystemConfigurations();
    wall_corrected_configuration.exec();
    shell_curvature.exec();
    water_block_complex.updateConfiguration();
    boundary_indicator.exec();
    left_bidirection_buffer.tag_buffer_particles.exec();
    right_bidirection_buffer.tag_buffer_particles.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *system.getSystemVariableDataByName<Real>("PhysicalTime");
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

    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    body_states_recording.writeToFile(0);

    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (physical_time < end_time)
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
                left_pressure_condition.exec(dt);
                right_pressure_condition.exec(dt);
                inflow_velocity_condition.exec();
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
            }
            number_of_iterations++;

            time_instance = TickCount::now();
            /** Water block configuration and periodic condition. */
            left_bidirection_buffer.injection.exec();
            right_bidirection_buffer.injection.exec();
            
            left_bidirection_buffer.deletion.exec();
            right_bidirection_buffer.deletion.exec();

            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                particle_sorting.exec();
            }
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
