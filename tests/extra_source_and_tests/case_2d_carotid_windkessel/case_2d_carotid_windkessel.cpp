/**
 * @file 	carotid_VIPO_wall.cpp
 * @brief 	Carotid artery with solid wall, imposed velocity inlet and pressure outlet condition.
 */

#include "bidirectional_buffer.h"
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "particle_generation_and_detection.h"
#include "pressure_boundary.h"
#include "sphinxsys.h"
#include "windkessel_bc.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_blood_file = "./input/2d_carotid_points.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec2d translation(0.0, 0.0);
Real length_scale = pow(10, -3);
Vec2d domain_lower_bound(-7.0 * length_scale, -35.0 * length_scale);
Vec2d domain_upper_bound(20.0 * length_scale, 30.0 * length_scale);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 0.15 * length_scale;
//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1060;   /**< Reference density of fluid. */
Real U_f = 1.0;       /**< Characteristic velocity. */
Real U_max = 2 * U_f; /**< Characteristic velocity. */
/** Reference sound speed needs to consider the flow speed in the narrow channels. */
Real c_f = 10.0 * U_f;
Real mu_f = 0.00355; /**< Dynamics viscosity. */
Real Outlet_pressure = 0;

Real Rp_up = 8.61E5;
Real C_up = 3.02E-07;
Real Rd_up = 8.61E6;

Real Rp_down = 1.51E6;
Real C_down = 1.72E-07;
Real Rd_down = 1.51E7;
//----------------------------------------------------------------------
//	Define case dependent body shapes.
//----------------------------------------------------------------------
class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addAPolygonFromFile(full_path_to_blood_file, ShapeBooleanOps::add, Vecd::Zero(), length_scale);
    }
};

class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        MultiPolygon original_file;
        original_file.addAPolygonFromFile(full_path_to_blood_file, ShapeBooleanOps::add, Vecd::Zero(), length_scale);
        add<ExtrudeShape<MultiPolygonShape>>(3.0 * dp_0, original_file);
        subtract<MultiPolygonShape>(original_file);
    }
};

//----------------------------------------------------------------------
//	Define buffer locations.
//----------------------------------------------------------------------
Vecd buffer_half(2.0 * dp_0, 6.0 * length_scale);

Vecd inlet_center(1.667 * length_scale, -30.9115 * length_scale);
Vecd inlet_vector = Vecd(0.543, 5.426).normalized();
Real inlet_emitter_rotation = std::atan2(inlet_vector[1], inlet_vector[0]);
Real inlet_disposer_rotation = inlet_emitter_rotation + Pi;

Vecd outlet_up_center(-2.830 * length_scale, 21.7405 * length_scale);
Vecd outlet_up_vector = Vecd(-1.101, 3.306).normalized();
Real outlet_up_disposer_rotation = std::atan2(outlet_up_vector[1], outlet_up_vector[0]);
Real outlet_up_emitter_rotation = outlet_up_disposer_rotation + Pi;

Vecd outlet_down_center(8.962 * length_scale, 18.6785 * length_scale);
Vecd outlet_down_vector = Vecd(-0.071, 1.984).normalized();
Real outlet_down_disposer_rotation = std::atan2(outlet_down_vector[1], outlet_down_vector[0]);
Real outlet_down_emitter_rotation = outlet_down_disposer_rotation + Pi;

//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
/** carotid vel */
struct InflowVelocity
{
    Real u_ave;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ave(0.0) {}

    Vecd operator()(Vecd &position, Vecd &velocity, Real current_time)
    {
        Vecd target_velocity = velocity;

        u_ave = 0.3782;
        Real a[8] = {-0.1812, 0.1276, -0.08981, 0.04347, -0.05412, 0.02642, 0.008946, -0.009005};
        Real b[8] = {-0.07725, 0.01466, 0.004295, -0.06679, 0.05679, -0.01878, 0.01869, -0.01888};
        for (size_t i = 0; i < 8; i++)
        {
            u_ave = u_ave + a[i] * cos(8.302 * (i + 1) * current_time) + b[i] * sin(8.302 * (i + 1) * current_time);
        }

        target_velocity[0] = u_ave;
        target_velocity[1] = 0.0;

        return target_velocity;
    }
};

/** aorta vel */
// struct InflowVelocity
//{
//     Real u_ave, interval_;
//
//     template <class BoundaryConditionType>
//     InflowVelocity(BoundaryConditionType &boundary_condition)
//         : u_ave(0.0), interval_(0.66) {}
//
//     Vecd operator()(Vecd &position, Vecd &velocity, Real current_time)
//     {
//         Vecd target_velocity = velocity;
//         int n = static_cast<int>(current_time / interval_);
//         Real t_in_cycle = current_time - n * interval_;
//
//         u_ave = 5.0487;
//         Real a[8] = {4.5287, -4.3509, -5.8551, -1.5063, 1.2800, 0.9012, 0.0855, -0.0480};
//         Real b[8] = {-8.0420, -6.2637, 0.7465, 3.5239, 1.6283, -0.1306, -0.2738, -0.0449};
//
//         Real w = 2 * Pi / 1.0;
//         for (size_t i = 0; i < 8; i++)
//         {
//             u_ave = u_ave + a[i] * cos(w * (i + 1) * t_in_cycle) + b[i] * sin(w * (i + 1) * t_in_cycle);
//         }
//
//         //u_ave = fabs(u_ave);
//
//         //target_velocity[0] = SMAX(2.0 * u_ave * (1.0 - (position[1] * position[1] + position[2] * position[2]) / radius_inlet / radius_inlet),
//         //                          1.0e-2);
//
//         //target_velocity[0] = 2.0 * u_ave * (1.0 - (position[1] * position[1] + position[2] * position[2]) / radius_inlet / radius_inlet) * 5.9765 * scaling / 0.66;
//
//         target_velocity[0] = u_ave * (2.9293 * 2.9293 * 1.0E-2 * Pi / 5.9825);
//
//         target_velocity[1] = 0.0;
//         target_velocity[2] = 0.0;
//
//         return target_velocity;
//     }
// };

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
    sph_system.setReloadParticles(true);        // Tag for computation with save particles distribution
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape()->cleanLevelSet();
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
        : water_block.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineAdaptationRatios(1.15, 2.0);
    wall_boundary.defineBodyLevelSetShape()->correctLevelSetSign();
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
        InnerRelation blood_inner(water_block);

        BodyAlignedBoxByCell inlet_detection_box(wall_boundary,
                                                 makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(inlet_emitter_rotation), Vec2d(inlet_center - inlet_vector * 2 * dp_0)), buffer_half));

        BodyAlignedBoxByCell outlet_up_detection_box(wall_boundary,
                                                 makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(outlet_up_emitter_rotation), Vec2d(outlet_up_center + outlet_up_vector * 2 * dp_0)), buffer_half));

        BodyAlignedBoxByCell outlet_down_detection_box(wall_boundary,
                                                     makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(outlet_down_emitter_rotation), Vec2d(outlet_down_center + outlet_down_vector * 2 * dp_0)), buffer_half));

        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_particles(wall_boundary);
        SimpleDynamics<RandomizeParticlePosition> random_blood_particles(water_block);
        
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_up_particles_detection(outlet_up_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_down_particles_detection(outlet_down_detection_box);
        
        RelaxationStepInner relaxation_step_inner(wall_inner);
        RelaxationStepInner relaxation_step_inner_blood(blood_inner);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_state_to_vtp(sph_system);
        ParticleSorting particle_sorting(wall_boundary);
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files({&wall_boundary, &water_block});
        //----------------------------------------------------------------------
        //	Physics relaxation starts here.
        //----------------------------------------------------------------------
        random_particles.exec(0.25);
        random_blood_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_inner_blood.SurfaceBounding().exec();
        write_state_to_vtp.writeToFile(0.0);
        //----------------------------------------------------------------------
        // From here the time stepping begins.
        //----------------------------------------------------------------------
        int ite = 0;
        int relax_step = 2000;
        while (ite < relax_step)
        {
            relaxation_step_inner.exec();
            relaxation_step_inner_blood.exec();
            ite++;
            if (ite % 500 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
                write_state_to_vtp.writeToFile(ite);
            }
        }

        std::cout << "The physics relaxation process of wall particles finish !" << std::endl;
        
        inlet_particles_detection.exec();
        particle_sorting.exec();
        wall_boundary.updateCellLinkedList();

        outlet_up_particles_detection.exec();
        particle_sorting.exec();
        wall_boundary.updateCellLinkedList();

        outlet_down_particles_detection.exec();
        particle_sorting.exec();
        wall_boundary.updateCellLinkedList();
        
        write_state_to_vtp.writeToFile(ite);
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
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_wall_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_wall_contact);

    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_wall_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_wall_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_wall_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_wall_contact);

    // time step
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step_size(water_block);

    // add emitter and disposer
    BodyAlignedBoxByCell left_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(inlet_emitter_rotation), Vec2d(inlet_center + inlet_vector * 2 * dp_0)), buffer_half));
    fluid_dynamics::BidirectionalBuffer<fluid_dynamics::NonPrescribedPressure> left_buffer(left_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_up_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(outlet_up_emitter_rotation), Vec2d(outlet_up_center - outlet_up_vector * 2 * dp_0)), buffer_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer right_up_buffer(right_up_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_down_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation2d(outlet_down_emitter_rotation), Vec2d(outlet_down_center - outlet_down_vector * 2 * dp_0)), buffer_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer right_down_buffer(right_down_emitter, in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_wall_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<fluid_dynamics::NonPrescribedPressure>> left_inflow_pressure_condition(left_emitter);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> right_up_inflow_pressure_condition(right_up_emitter);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> right_down_inflow_pressure_condition(right_down_emitter);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_emitter);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    ParticleSorting particle_sorting(water_block);
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<Real>(water_block, "DensityChangeRate");
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "PositionDivergence");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    ReducedQuantityRecording<TotalKineticEnergy> write_water_kinetic_energy(water_block);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    boundary_indicator.exec();
    left_buffer.tag_buffer_particles.exec();
    right_up_buffer.tag_buffer_particles.exec();
    right_down_buffer.tag_buffer_particles.exec();
    wall_boundary_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 20.0;               /**< End time. */
    Real Output_Time = 0.01; /**< Time stamps for output of body states. */
    //Real Output_Time = 0.001; /**< Time stamps for output of body states. */
    Real dt = 0.0; /**< Default acoustic time step sizes. */
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    TimeInterval interval_computing_time_step;
    TimeInterval interval_computing_pressure_relaxation;
    TimeInterval interval_updating_configuration;
    TickCount time_instance;

    Real accumulated_time = 0.006;
    int updateP_n = 0;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    body_states_recording.writeToFile();

    right_up_inflow_pressure_condition.getTargetPressure()->setWindkesselParams(Rp_up, C_up, Rd_up, accumulated_time);
    right_down_inflow_pressure_condition.getTargetPressure()->setWindkesselParams(Rp_down, C_down, Rd_down, accumulated_time);
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
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
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);

                pressure_relaxation.exec(dt);
                kernel_summation.exec();

                inflow_velocity_condition.exec();
                // left_inflow_pressure_condition.exec(dt);
                //  windkessel model implementation
                if (physical_time >= updateP_n * accumulated_time)
                {
                    right_up_inflow_pressure_condition.getTargetPressure()->updateNextPressure();
                    right_down_inflow_pressure_condition.getTargetPressure()->updateNextPressure();

                    ++updateP_n;
                }
                right_up_inflow_pressure_condition.exec(dt);
                right_down_inflow_pressure_condition.exec(dt);

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

                write_water_kinetic_energy.writeToFile(number_of_iterations);
            }
            number_of_iterations++;

            time_instance = TickCount::now();

            left_buffer.injection.exec();
            right_up_buffer.injection.exec();
            right_down_buffer.injection.exec();

            left_buffer.deletion.exec();
            right_up_buffer.deletion.exec();
            right_down_buffer.deletion.exec();

            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                particle_sorting.exec();
            }

            water_block.updateCellLinkedList();
            water_block_complex.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;

            boundary_indicator.exec();
            left_buffer.tag_buffer_particles.exec();
            right_up_buffer.tag_buffer_particles.exec();
            right_down_buffer.tag_buffer_particles.exec();
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
