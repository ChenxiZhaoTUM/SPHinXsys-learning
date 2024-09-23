/**
 * @file 	straight_vessel_with_EWall.cpp
 * @brief 	3D blood flow in straigt vessel with PIPO and elastic wall
 */
#include "bidirectional_buffer.h"
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "pressure_boundary.h"
#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_stl_file_wall_outer = "./input/wall_outer.stl";
std::string full_path_to_stl_file_fluid = "./input/water.stl";

Real DL = 0.1;                                             /**< Channel length. */
Real diameter = 0.02;                                             /**< Channel height. */
Real resolution_ref = diameter / 20.0;                             /**< Initial reference particle spacing. */
Real wall_thickness = resolution_ref * 4;                                /**< Extending width for BCs. */
Vec3d translation_fluid(DL * 0.5, 0., 0.);
BoundingBox system_domain_bounds(Vecd(0, -0.5 * diameter - wall_thickness, -0.5 * diameter - wall_thickness),
                                 Vecd(DL, 0.5 * diameter + wall_thickness, 0.5 * diameter + wall_thickness));
StdVec<Vecd> observer_location = {Vecd(0.3 * DL, diameter/2, 0.0)}; /**< Displacement observation point. */
//----------------------------------------------------------------------
//	Material parameters.
//----------------------------------------------------------------------
Real Inlet_pressure = 5000;
Real Outlet_pressure = 0.0;
Real rho0_f = 1000.0;
Real mu_f = 4.0e-3;
Real U_f = 0.8;
Real c_f = 10.0 * U_f;

Real rho0_s = 1000;           /** Normalized density. */
Real Youngs_modulus = 1.0e6; /** Normalized Youngs Modulus. */
Real poisson = 0.3;          /** Poisson ratio. */
//Real physical_viscosity = 0.25 * sqrt(rho0_s * Youngs_modulus) * DL;  //physical_viscosity = 790.569
Real physical_viscosity = 2000;
//----------------------------------------------------------------------
//	parameters of buffers
//----------------------------------------------------------------------
Vecd bidirectional_buffer_halfsize = Vecd(2.0 * resolution_ref, 0.7 * diameter, 0.7 * diameter);
Vecd left_bidirectional_translation(2.0 * resolution_ref, 0.0, 0.0);
Vecd right_bidirectional_translation(DL - 2.0 * resolution_ref, 0.0, 0.0);
//----------------------------------------------------------------------
//  Define body shapes
//----------------------------------------------------------------------
Vec3d translation(0., 0., 0.);
Real scale = 1;
class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        /** Geometry definition. */
        add<TriangleMeshShapeSTL>(full_path_to_stl_file_fluid, translation, scale);
    }
};

class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        /** Geometry definition. */
        add<TriangleMeshShapeSTL>(full_path_to_stl_file_wall_outer, translation, scale);
        subtract<TriangleMeshShapeSTL>(full_path_to_stl_file_fluid, translation, scale);
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
        Real run_time = GlobalStaticVariables::physical_time_;
        /*constant pressure*/
        Real pressure = Inlet_pressure;
        return run_time < 0.05 ? run_time*pressure/0.05 : pressure;
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
        if (base_particles_.ParticlePositions()[index_i][0] < constrain_len_ || base_particles_.ParticlePositions()[index_i][0] > DL - constrain_len_)
        {
            body_part_particles_.push_back(index_i);
        }

        /*if (base_particles_.ParticlePositions()[index_i][0] < resolution_ref || base_particles_.ParticlePositions()[index_i][0] > DL - resolution_ref)
        {
            body_part_particles_.push_back(index_i);
        }*/
    };
};

int main(int ac, char *av[])
{
    //std::cout << "U_f = " << U_f << std::endl;
    //std::cout << "physical_viscosity = " << physical_viscosity << std::endl;

    //----------------------------------------------------------------------
    //  Build up -- a SPHSystem --
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(true);        // Tag for computation with save particles distribution
#ifdef BOOST_AVAILABLE
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment(); // handle command line arguments
#endif
    //----------------------------------------------------------------------
    //	Creating bodies with corresponding materials and particles.
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
        : water_block.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    //wall_boundary.defineAdaptationRatios(1.15, 2.0);
    wall_boundary.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
    wall_boundary.defineMaterial<SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_boundary.generateParticles<BaseParticles, Reload>(wall_boundary.getName())
        : wall_boundary.generateParticles<BaseParticles, Lattice>();

    ObserverBody displacement_observer(sph_system, "DisplacementObserver");
    displacement_observer.generateParticles<ObserverParticles>(observer_location);
    
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
        BodyStatesRecordingToVtp write_body_to_vtp(sph_system);
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
        while (ite_p < 2000)
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
    ContactRelation observer_contact(displacement_observer, {&wall_boundary});
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
    // add solid dampinng
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseInner<Vecd, FixedDampingRate>>>
        wall_velocity_damping(0.5, wall_boundary_inner, "Velocity", physical_viscosity);
    /** Exert constrain on wall. */
    BoundaryGeometry boundary_geometry(wall_boundary, "BoundaryGeometry", resolution_ref * 4);
    //SimpleDynamics<FixedInAxisDirection> constrain_holder(boundary_geometry, Vecd(0.0, 1.0, 1.0));
    SimpleDynamics<FixBodyPartConstraint> constrain_holder(boundary_geometry);

    // fluid dynamics
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> kernel_correction_complex(water_block_inner, water_block_contact);
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_block_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_block_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_block_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_block_contact);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    // add fluid dampinng
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseWithWall<Vecd, FixedDampingRate>>> implicit_viscous_damping(
        0.2, ConstructorArgs(water_block_inner, "Velocity", mu_f), ConstructorArgs(water_block_contact, "Velocity", mu_f));
    // Boundary conditions
    AlignedBoxShape left_disposer_shape(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vecd(left_bidirectional_translation)), bidirectional_buffer_halfsize);
    BodyAlignedBoxByCell left_disposer(water_block, left_disposer_shape);
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_disposer_outflow_deletion(left_disposer);

    AlignedBoxShape right_disposer_shape(xAxis, Transform(Vecd(right_bidirectional_translation)), bidirectional_buffer_halfsize);
    BodyAlignedBoxByCell right_disposer(water_block, right_disposer_shape);
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> right_disposer_outflow_deletion(right_disposer);

    AlignedBoxShape left_emitter_shape(xAxis, Transform(Vecd(left_bidirectional_translation)), bidirectional_buffer_halfsize);
    BodyAlignedBoxByCell left_emitter(water_block, left_emitter_shape);
    fluid_dynamics::BidirectionalBuffer<LeftInflowPressure, SequencedPolicy> left_emitter_inflow_injection(left_emitter, in_outlet_particle_buffer);

    AlignedBoxShape right_emitter_shape(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vecd(right_bidirectional_translation)), bidirectional_buffer_halfsize);
    BodyAlignedBoxByCell right_emitter(water_block, right_emitter_shape);
    fluid_dynamics::BidirectionalBuffer<RightInflowPressure, SequencedPolicy> right_emitter_inflow_injection(right_emitter, in_outlet_particle_buffer);

    SimpleDynamics<fluid_dynamics::PressureCondition<LeftInflowPressure>> left_inflow_pressure_condition(left_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightInflowPressure>> right_inflow_pressure_condition(right_emitter);
    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_block_contact);
    

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
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    body_states_recording.addToWrite<Vecd>(wall_boundary, "NormalDirection");
    body_states_recording.addToWrite<Vecd>(wall_boundary, "PressureForceFromFluid");
    ObservedQuantityRecording<Vec3d> write_wall_displacement("Position", observer_contact);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    boundary_indicator.exec();
    left_emitter_inflow_injection.tag_buffer_particles.exec();
    right_emitter_inflow_injection.tag_buffer_particles.exec();
    wall_boundary_normal_direction.exec();
    wall_corrected_configuration.exec();
    water_block_complex.updateConfiguration();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 0.15;   /**< End time. */
    Real Output_Time = end_time/150; /**< Time stamps for output of body states. */
    Real dt = 0.0;          /**< Default acoustic time step sizes. */
    Real dt_s = 0.0;        /**< Default acoustic time step sizes for solid. */
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
    write_wall_displacement.writeToFile(number_of_iterations);
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
            Real Dt = get_fluid_advection_time_step_size.exec();
            update_fluid_density.exec();
            kernel_correction_complex.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();
            /** FSI for viscous force. */
            viscous_force_on_wall.exec();

            interval_computing_time_step += TickCount::now() - time_instance;
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);
                /*if (GlobalStaticVariables::physical_time_ < 0.05)
                {
                    implicit_viscous_damping.exec(dt);
                }*/

                pressure_relaxation.exec(dt);
                
                kernel_summation.exec();
                left_inflow_pressure_condition.exec(dt);
                right_inflow_pressure_condition.exec(dt);

                /** FSI for pressure force. */
                pressure_force_on_wall.exec();

                density_relaxation.exec(dt);

                Real dt_s_sum = 0.0;
                average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    dt_s = wall_computing_time_step_size.exec();
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;
                    // std::cout << "dt_s = " << dt_s << std::endl;
                    wall_stress_relaxation_first_half.exec(dt_s);

                    constrain_holder.exec(dt_s);
					wall_velocity_damping.exec(dt_s);
					constrain_holder.exec(dt_s);

                    wall_stress_relaxation_second_half.exec(dt_s);
                    dt_s_sum += dt_s;

                    // body_states_recording.writeToFile();
                }
                average_velocity_and_acceleration.update_averages_.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                GlobalStaticVariables::physical_time_ += dt;
            }
            interval_computing_pressure_relaxation += TickCount::now() - time_instance;
            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << GlobalStaticVariables::physical_time_
                          << "	Dt = " << Dt << "	dt = " << dt << "	dt_s = " << dt_s << "\n";

                if (number_of_iterations % observation_sample_interval == 0 && number_of_iterations != sph_system.RestartStep())
                    write_wall_displacement.writeToFile(number_of_iterations);
            }
            number_of_iterations++;

            time_instance = TickCount::now();

            left_emitter_inflow_injection.injection.exec();
            right_emitter_inflow_injection.injection.exec();
            left_disposer_outflow_deletion.exec();
            right_disposer_outflow_deletion.exec();

            water_block.updateCellLinkedListWithParticleSort(100);
            wall_update_normal.exec();
            wall_boundary.updateCellLinkedList();
            water_block_complex.updateConfiguration();
            wall_water_contact.updateConfiguration();
            observer_contact.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;
            
            boundary_indicator.exec();
            left_emitter_inflow_injection.tag_buffer_particles.exec();
            right_emitter_inflow_injection.tag_buffer_particles.exec();
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
