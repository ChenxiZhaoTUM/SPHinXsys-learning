/**
 * @file 	case_3d_cylinder_Windkessel_shell.cpp
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
#include "windkessel_bc.h"
#include "hemodynamic_indices.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real scale = 0.001;
Real diameter = 6.35 * scale;
Real fluid_radius = 0.5 * diameter;
Real full_length = 10 * fluid_radius;
//----------------------------------------------------------------------
//	Geometry parameters for wall.
//----------------------------------------------------------------------
int number_of_particles = 30;
Real resolution_ref = diameter / number_of_particles;
//Real resolution_shell = 0.5 * resolution_ref;
Real resolution_shell = resolution_ref;
Real wall_thickness = resolution_ref * 4.0;
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
Real rho0_f = 1000.0; /**< Reference density of fluid. */
Real U_max = 1.0;
Real c_f = 10.0 * U_max; /**< Reference sound speed. */
Real mu_f = 0.004;

Real rho0_s = 1200;           /** Normalized density. */
Real Youngs_modulus = 5.0e7; /** Normalized Youngs Modulus. */
Real poisson = 0.3;          /** Poisson ratio. */
Real physical_viscosity = diameter/full_length/4 * sqrt(rho0_s*Youngs_modulus) * diameter;
//Real physical_viscosity = 200;
//----------------------------------------------------------------------
//	Shell particle generation
//----------------------------------------------------------------------
class ShellBoundary;
template <>
class ParticleGenerator<SurfaceParticles, ShellBoundary> : public ParticleGenerator<SurfaceParticles>
{
    Real resolution_shell_;
    Real shell_thickness_;

  public:
    explicit ParticleGenerator(SPHBody &sph_body, SurfaceParticles &surface_particles,
                               Real resolution_shell, Real shell_thickness)
        : ParticleGenerator<SurfaceParticles>(sph_body, surface_particles),
          resolution_shell_(resolution_shell),
          shell_thickness_(shell_thickness){};
    void prepareGeometricData() override
    {
        Real radius_mid_surface = fluid_radius + resolution_shell_ * 0.5;
        auto particle_number_mid_surface =
            int(2.0 * radius_mid_surface * Pi / resolution_shell_);
        auto particle_number_height =
            int(full_length / resolution_shell_);
        for (int i = 0; i < particle_number_mid_surface; i++)
        {
            for (int j = 0; j < particle_number_height; j++)
            {
                Real theta = (i + 0.5) * 2 * Pi / (Real)particle_number_mid_surface;
                
                Real x = full_length  * j / (Real)particle_number_height + 0.5 * resolution_shell_;
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
    Real u_ave;
    AlignedBoxShape &aligned_box_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ave(0.0), aligned_box_(boundary_condition.getAlignedBox()) {}

    Vecd operator()(Vecd &position, Vecd &velocity, Real current_time)
    {
        Vecd target_velocity = velocity;

        u_ave = 0.2339;
        Real a[8] = {-0.0176, -0.0657, -0.0280, 0.0068, 0.0075, 0.0115, 0.0040, 0.0035};
        Real b[8] = {0.1205, 0.0171, -0.0384, -0.0152, -0.0122, 0.0002, 0.0033, 0.0060};
        Real w = 2 * Pi / 1;
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
//----------------------------------------------------------------------
//	Observation points.
//----------------------------------------------------------------------
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
    double x = full_length / 2.0;
    double R = diameter / 2.0;

    for (int i = 0; i <= number_of_particles; ++i)
    {
        double z = -R + (2.0 * R) * i / double(number_of_particles);
        observation_points.emplace_back(Vec3d(x, 0.0, z) + translation);
    }

    return observation_points;
};

StdVec<Vecd> createWallAxialObservationPoints(
    double full_length, Vec3d translation = Vec3d(0.0, 0.0, 0.0))
{
    StdVec<Vecd> observation_points;
    int nx = 51;
    for (int i = 0; i < nx; i++)
    {
        double x = full_length / (nx - 1) * i;
        Vec3d point_coordinate(x, -fluid_radius - 0.5 * resolution_shell, 0.0);
        observation_points.emplace_back(point_coordinate + translation);
    }
    return observation_points;
};

StdVec<Vecd> displacement_observation_location = {
    Vecd(5.0 * scale, fluid_radius + 0.5 * resolution_shell, 0.0), 
    Vecd(10.0 * scale, fluid_radius + 0.5 * resolution_shell, 0.0), 
    Vecd(15.0 * scale, fluid_radius + 0.5 * resolution_shell, 0.0),
    Vecd(20.0 * scale, fluid_radius + 0.5 * resolution_shell, 0.0),
    Vecd(25.0 * scale, fluid_radius + 0.5 * resolution_shell, 0.0)
};

//----------------------------------------------------------------------
//	Boundary constrain
//----------------------------------------------------------------------
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
        if (base_particles_.ParticlePositions()[index_i][0] < constrain_len_ 
            || base_particles_.ParticlePositions()[index_i][0] > full_length - constrain_len_)
        {
            body_part_particles_.push_back(index_i);
        }
    };
};

//----------------------------------------------------------------------
//	Main code.
//----------------------------------------------------------------------
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
    water_block.defineClosure<WeaklyCompressibleFluid, Viscosity>(ConstructArgs(rho0_f, c_f), mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    water_block.defineBodyLevelSetShape(2.0)->correctLevelSetSign();
    (!system.RunParticleRelaxation() && system.ReloadParticles())
        ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
        : water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);

    SolidBody shell_boundary(system, makeShared<DefaultShape>("Shell"));
    shell_boundary.defineAdaptation<SPH::SPHAdaptation>(1.15, resolution_ref / resolution_shell);
    shell_boundary.defineMaterial<NeoHookeanSolid>(rho0_s, Youngs_modulus, poisson);
    shell_boundary.generateParticles<SurfaceParticles, ShellBoundary>(resolution_shell, wall_thickness);

    ObserverBody fluid_axial_observer(system, "fluid_observer_axial");
    fluid_axial_observer.generateParticles<ObserverParticles>(createAxialObservationPoints(full_length));
    ObserverBody fluid_radial_observer(system, "fluid_observer_radial");
    fluid_radial_observer.generateParticles<ObserverParticles>(createRadialObservationPoints(full_length, diameter, 50));
    ObserverBody wall_axial_observer(system, "wall_observer_axial");
    wall_axial_observer.generateParticles<ObserverParticles>(createWallAxialObservationPoints(full_length));
    ObserverBody wall_displacement_observer(system, "wall_observer_displacement");
    wall_displacement_observer.generateParticles<ObserverParticles>(displacement_observation_location);
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (system.RunParticleRelaxation() && !system.ReloadParticles())
    {
        InnerRelation water_block_inner(water_block);
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
        RelaxationStepInner relaxation_step_water_inner(water_block_inner);
        //----------------------------------------------------------------------
        //	Relaxation output
        //----------------------------------------------------------------------
        BodyStatesRecordingToVtp write_body_state_to_vtp(system);
        ReloadParticleIO write_particle_reload_files({ &water_block});
        //----------------------------------------------------------------------
        //	Physics relaxation starts here.
        //----------------------------------------------------------------------
        random_water_particles.exec(0.25);
        relaxation_step_water_inner.SurfaceBounding().exec();
        write_body_state_to_vtp.writeToFile(0.0);
        //----------------------------------------------------------------------
        // From here the time stepping begins.
        //----------------------------------------------------------------------
        int ite = 0;
        int relax_step = 1000;
        while (ite < relax_step)
        {
            relaxation_step_water_inner.exec();
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
    InnerRelation shell_boundary_inner(shell_boundary);
    ShellInnerRelationWithContactKernel wall_curvature_inner(shell_boundary, water_block);
    ContactRelationFromShellToFluid water_block_contact(water_block, {&shell_boundary}, {false});
    ContactRelationFromFluidToShell shell_water_contact(shell_boundary, {&water_block}, {false});
    ContactRelation fluid_observer_contact_axial(fluid_axial_observer, {&water_block});
    ContactRelation fluid_observer_contact_radial(fluid_radial_observer, {&water_block});
    ContactRelation shell_observer_contact_axial(wall_axial_observer, {&shell_boundary});
    ContactRelation shell_observer_contact_displacement(wall_displacement_observer, {&shell_boundary});
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

    //----------------------------------------------------------------------
    //	Fluid dynamics
    //----------------------------------------------------------------------
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_block_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_block_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_block_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_block_contact);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step_size(water_block, U_max);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step_size(water_block);

    BodyAlignedBoxByCell left_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(emitter_translation)), emitter_halfsize));
    fluid_dynamics::BidirectionalBuffer<fluid_dynamics::NonPrescribedPressure> left_bidirection_buffer(left_buffer, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_buffer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vec3d(disposer_translation)), disposer_halfsize));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer right_bidirection_buffer(right_buffer, in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_block_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<fluid_dynamics::NonPrescribedPressure>> left_pressure_condition(left_buffer);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> right_pressure_condition(right_buffer);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_buffer);

    ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_inlet_transient_flow_rate(left_buffer, Pi*fluid_radius*fluid_radius);
    //----------------------------------------------------------------------
    //	Solid dynamics
    //----------------------------------------------------------------------
    InteractionDynamics<thin_structure_dynamics::ShellCorrectConfiguration> wall_corrected_configuration(shell_boundary_inner);
    SimpleDynamics<thin_structure_dynamics::AverageShellCurvature> shell_curvature(wall_curvature_inner);
    Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationFirstHalf> shell_stress_relaxation_first(shell_boundary_inner, 3, true);
    Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationSecondHalf> shell_stress_relaxation_second(shell_boundary_inner);
    ReduceDynamics<thin_structure_dynamics::ShellAcousticTimeStepSize> shell_time_step_size(shell_boundary);
    SimpleDynamics<thin_structure_dynamics::UpdateShellNormalDirection> shell_update_normal(shell_boundary);
    /** Exert constrain on shell. */
    BoundaryGeometry boundary_geometry(shell_boundary, "BoundaryGeometry", resolution_ref * 4);
    //SimpleDynamics<thin_structure_dynamics::ConstrainShellBodyRegion> constrain_holder(boundary_geometry);
    SimpleDynamics<FixBodyPartConstraint> constrain_holder(boundary_geometry);
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseInner<Vec3d, FixedDampingRate>>>
        shell_velocity_damping(0.2, shell_boundary_inner, "Velocity", physical_viscosity);
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseInner<Vec3d, FixedDampingRate>>>
        shell_rotation_damping(0.2, shell_boundary_inner, "AngularVelocity", physical_viscosity);
    //----------------------------------------------------------------------
    //	FSI
    //----------------------------------------------------------------------
    InteractionWithUpdate<solid_dynamics::WallShearStress> viscous_force_from_fluid(shell_water_contact);
    SimpleDynamics<solid_dynamics::HemodynamicIndiceCalculation> hemodynamic_indice_calculation(shell_boundary, 1.0);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_on_shell(shell_water_contact);
    solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(shell_boundary);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //----------------------------------------------------------------------
    ParticleSorting particle_sorting(water_block);
    BodyStatesRecordingToVtp body_states_recording(system);
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    body_states_recording.addToWrite<Vecd>(shell_boundary, "NormalDirection");
    body_states_recording.addToWrite<Matd>(shell_boundary, "MidSurfaceCauchyStress");
    body_states_recording.addDerivedVariableRecording<SimpleDynamics<Displacement>>(shell_boundary);
    body_states_recording.addToWrite<Real>(shell_boundary, "Average1stPrincipleCurvature");
    body_states_recording.addToWrite<Real>(shell_boundary, "Average2ndPrincipleCurvature");
    body_states_recording.addToWrite<Vecd>(shell_boundary, "WallShearStress");
    body_states_recording.addToWrite<Real>(shell_boundary, "TimeAveragedWallShearStress");
    body_states_recording.addToWrite<Real>(shell_boundary, "OscillatoryShearIndex");
    ObservedQuantityRecording<Vecd> write_shell_WSS_axial("WallShearStress", shell_observer_contact_axial);
    AxialVelocityRecording write_fluid_velocity_axial(fluid_observer_contact_axial);
    AxialVelocityRecording write_fluid_velocity_radial(fluid_observer_contact_radial);
    ObservedQuantityRecording<Vecd> write_wall_displacement("Position", shell_observer_contact_displacement);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    system.initializeSystemCellLinkedLists();
    system.initializeSystemConfigurations();
    wall_corrected_configuration.exec();
    shell_curvature.exec();
    water_block_complex.updateConfiguration();
    shell_water_contact.updateConfiguration();
    //correct_kernel_weights_for_interpolation.exec();
    boundary_indicator.exec();
    left_bidirection_buffer.tag_buffer_particles.exec();
    right_bidirection_buffer.tag_buffer_particles.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *system.getSystemVariableDataByName<Real>("PhysicalTime");
    size_t number_of_iterations = 0;
    int screen_output_interval = 100;
    Real end_time = 5.0;               /**< End time. */
    Real Output_Time = 0.01; /**< Time stamps for output of body states. */
    Real dt = 0.0;                     /**< Default acoustic time step sizes. */
    Real dt_s = 0.0; /**< Default acoustic time step sizes for solid. */
    Real accumulated_time = 0.01;
    int updateP_n = 0;
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
    right_pressure_condition.getTargetPressure()->setWindkesselParams(3.05E8, 9.79E-10, 1.37E9, accumulated_time);
    
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

            /** FSI for viscous force. */
            viscous_force_from_fluid.exec();
            hemodynamic_indice_calculation.exec(Dt);

            interval_computing_time_step += TickCount::now() - time_instance;
            /** Dynamics including pressure relaxation. */
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(),
                          Dt - relaxation_time);
                pressure_relaxation.exec(dt);
                /** FSI for pressure force. */
                pressure_force_on_shell.exec();

                // boundary condition implementation
                kernel_summation.exec();

                left_pressure_condition.exec(dt);
                if (physical_time >= updateP_n * accumulated_time)
                {
                    right_pressure_condition.getTargetPressure()->updateNextPressure();

                    ++updateP_n;
                }
                right_pressure_condition.exec(dt);
                inflow_velocity_condition.exec();

                density_relaxation.exec(dt);

                Real dt_s_sum = 0.0;
                average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    dt_s = shell_time_step_size.exec();
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;
                    shell_stress_relaxation_first.exec(dt_s);

                    constrain_holder.exec();
                    shell_velocity_damping.exec(dt_s);
                    shell_rotation_damping.exec(dt_s);
                    constrain_holder.exec();

                    shell_stress_relaxation_second.exec(dt_s);
                    dt_s_sum += dt_s;  
                }
                average_velocity_and_acceleration.update_averages_.exec(dt);

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
                          << "	Dt = " << Dt << "	dt = " << dt << "	dt_s = " << dt_s << "\n";
            }
            number_of_iterations++;

            time_instance = TickCount::now();

            compute_inlet_transient_flow_rate.exec();

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
            shell_update_normal.exec();
            shell_boundary.updateCellLinkedList();
            shell_boundary_inner.updateConfiguration();
            shell_curvature.exec();
            shell_water_contact.updateConfiguration();
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
        fluid_observer_contact_axial.updateConfiguration();
        fluid_observer_contact_radial.updateConfiguration();
        write_fluid_velocity_axial.writeToFile(number_of_iterations);
        write_fluid_velocity_radial.writeToFile(number_of_iterations);

        shell_observer_contact_axial.updateConfiguration();
        write_shell_WSS_axial.writeToFile(number_of_iterations);
        shell_observer_contact_displacement.updateConfiguration();
        write_wall_displacement.writeToFile(number_of_iterations);
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