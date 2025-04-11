/**
 * @file 	carotid_VIPO_wall.cpp
 * @brief 	Carotid artery with solid wall, imposed velocity inlet and pressure outlet condition.
 */

#include "sphinxsys.h"
#include "bidirectional_buffer.h"
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "pressure_boundary.h"
#include "particle_generation_and_detection.h"
#include "windkessel_bc.h"
#include "hemodynamic_indices.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_blood_file = "./input/bif_artery.STL";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real length_scale = pow(10, -3);
Vec3d domain_lower_bound(-8.0 * length_scale, -7.0 * length_scale, -35.0 * length_scale);
Vec3d domain_upper_bound(20.0 * length_scale, 13.0 * length_scale, 30.0 * length_scale);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 0.2 * length_scale;
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

Vecd standard_direction(1, 0, 0);

// inlet R=3.130, (1.583, 5.904, -31.850), (0.0, 0.0, 1.0)
Real DW_in = 3.130 * 2 * length_scale;
Vecd inlet_buffer_half = Vecd(2.0 * dp_0, 5.0 * length_scale, 5.0 * length_scale);
Vecd inlet_normal(0.0, 0.0, 1.0);
Vecd inlet_buffer_translation = Vecd(1.583, 5.904, -31.850) * length_scale + inlet_normal * 2.0 * dp_0;
Vecd inlet_cut_translation = Vecd(1.583, 5.904, -31.850) * length_scale - inlet_normal * 2.0 * dp_0;
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, standard_direction);
Rotation3d inlet_emitter_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);
Rotation3d inlet_disposer_rotation(inlet_rotation_result.angle + Pi, inlet_rotation_result.axis);

// outlet1 R=1.501, (8.993, 0.932, 19.124), (0.0, 0.0, 1.0)
Real DW_out_up = 1.501 * 2 * length_scale;
Vecd outlet_up_buffer_half = Vecd(2.0 * dp_0, 3.0 * length_scale, 3.0 * length_scale);
Vecd outlet_up_normal(0.0, 0.0, 1.0);
Vecd outlet_up_buffer_translation = Vecd(8.993, 0.932, 19.124) * length_scale - outlet_up_normal * 2.0 * dp_0;
Vecd outlet_up_cut_translation = Vecd(8.993, 0.932, 19.124) * length_scale + outlet_up_normal * 2.0 * dp_0;
RotationResult outlet_up_rotation_result = RotationCalculator(outlet_up_normal, standard_direction);
Rotation3d outlet_up_disposer_rotation(outlet_up_rotation_result.angle, outlet_up_rotation_result.axis);
Rotation3d outlet_up_emitter_rotation(outlet_up_rotation_result.angle + Pi, outlet_up_rotation_result.axis);

// outlet2 R=2.118, (-2.991, -0.416, 22.215), (-0.316, 0.0, 0.949)
Real DW_out_down = 2.118 * 2 * length_scale;
Vecd outlet_down_buffer_half = Vecd(2.0 * dp_0, 4.0 * length_scale, 4.0 * length_scale);
Vecd outlet_down_normal(-0.316, 0.0, 0.949);
Vecd outlet_down_buffer_translation = Vecd(-2.991, -0.416, 22.215) * length_scale - outlet_down_normal * 2.0 * dp_0;
Vecd outlet_down_cut_translation = Vecd(-2.991, -0.416, 22.215) * length_scale + outlet_down_normal * 2.0 * dp_0;
RotationResult outlet_down_rotation_result = RotationCalculator(outlet_down_normal, standard_direction);
Rotation3d outlet_down_disposer_rotation(outlet_down_rotation_result.angle, outlet_down_rotation_result.axis);
Rotation3d outlet_down_emitter_rotation(outlet_down_rotation_result.angle + Pi, outlet_down_rotation_result.axis);
//----------------------------------------------------------------------
//	Global parameters on the fluid properties
//----------------------------------------------------------------------
Real rho0_f = 1060; /**< Reference density of fluid. */
Real U_f = 0.5;    /**< Characteristic velocity. */
/** Reference sound speed needs to consider the flow speed in the narrow channels. */
Real c_f = 10.0 * U_f * SMAX(Real(1), DW_in * DW_in / (DW_out_up * DW_out_up + DW_out_down * DW_out_down));
Real mu_f = 0.0035; /**< Dynamics viscosity. */
Real Outlet_pressure = 0;
//----------------------------------------------------------------------
//	Define case dependent body shapes.
//----------------------------------------------------------------------
class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeSTL>(full_path_to_blood_file, translation, length_scale);
    }
};

class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<ExtrudeShape<TriangleMeshShapeSTL>>(3.0 * dp_0, full_path_to_blood_file, translation, length_scale);
        subtract<TriangleMeshShapeSTL>(full_path_to_blood_file, translation, length_scale);
    }
};
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
/** fluent vel */ 
struct InflowVelocity
{
    Real u_ref_, t_ref_, interval_;
    AlignedBoxShape &aligned_box_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ref_(0.1), t_ref_(0.218), interval_(0.5),
        aligned_box_(boundary_condition.getAlignedBox()){}

    Vecd operator()(Vecd &position, Vecd &velocity, Real current_time)
    {
        Vecd target_velocity = velocity;
        int n = static_cast<int>(current_time / interval_);
        Real t_in_cycle = current_time - n * interval_;

        target_velocity[0] = t_in_cycle < t_ref_ ? 0.5 * sin(4 * Pi * (current_time + 0.0160236)) : u_ref_;
        return target_velocity;
    }
};
//----------------------------------------------------------------------
//	Pressure boundary definition.
//----------------------------------------------------------------------
struct RightInflowPressure
{
    template <class BoundaryConditionType>
    RightInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real p, Real curent_time)
    {
        /*constant pressure*/
        Real pressure = Outlet_pressure;
        return pressure;
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
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(true);       // Tag for computation with save particles distribution
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape()->cleanLevelSet();
    water_block.defineClosure<WeaklyCompressibleFluid, Viscosity>(ConstructArgs(rho0_f, c_f), mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
    : water_block.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineAdaptationRatios(1.15, 2.0);
    wall_boundary.defineMaterial<Solid>();
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    {
        wall_boundary.generateParticles<BaseParticles, Reload>(wall_boundary.getName());
    }
    else
    {
        wall_boundary.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
        wall_boundary.generateParticles<BaseParticles, Lattice>();
    }
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation wall_inner(wall_boundary);
        InnerRelation blood_inner(water_block);

        BodyAlignedBoxByCell inlet_detection_box(wall_boundary,
                                             makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_buffer_half));
        BodyAlignedBoxByCell outlet_up_detection_box(wall_boundary,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_up_emitter_rotation), Vec3d(outlet_up_cut_translation)), outlet_up_buffer_half));
        BodyAlignedBoxByCell outlet_down_detection_box(wall_boundary,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_down_emitter_rotation), Vec3d(outlet_down_cut_translation)), outlet_down_buffer_half));
        
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_particles(wall_boundary);
        SimpleDynamics<RandomizeParticlePosition> random_blood_particles(water_block);
        RelaxationStepInner relaxation_step_inner(wall_inner);
        RelaxationStepInner relaxation_step_inner_blood(blood_inner);

        // here, need a class to switch particles in aligned box to ghost particles (not real particles)
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_up_particles_detection(outlet_up_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_down_particles_detection(outlet_down_detection_box);

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_state_to_vtp(sph_system);
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files({ &wall_boundary, &water_block });
        ParticleSorting particle_sorting_wall(wall_boundary);
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
        int relax_step = 3000;
        while (ite < relax_step)
        {
            relaxation_step_inner.exec();
            relaxation_step_inner_blood.exec();
            ite++;
            if (ite % 500 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
                //write_state_to_vtp.writeToFile(ite);
            }
        }

        inlet_particles_detection.exec();
        particle_sorting_wall.exec();
        wall_boundary.updateCellLinkedList();
        outlet_up_particles_detection.exec();
        particle_sorting_wall.exec();
        wall_boundary.updateCellLinkedList();
        outlet_down_particles_detection.exec();
        particle_sorting_wall.exec();

        std::cout << "The physics relaxation process of wall particles finish !" << std::endl;

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
    ContactRelation wall_contact(wall_boundary, {&water_block});
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
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step_size(water_block);

    //----------------------------------------------------------------------
    //	Boundary conditions.
    //----------------------------------------------------------------------
    BodyAlignedBoxByCell left_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_buffer_translation)), inlet_buffer_half));
    fluid_dynamics::BidirectionalBuffer<fluid_dynamics::NonPrescribedPressure> left_buffer(left_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_up_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_up_emitter_rotation), Vec3d(outlet_up_buffer_translation)), outlet_up_buffer_half));
    fluid_dynamics::BidirectionalBuffer<RightInflowPressure> right_up_buffer(right_up_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_down_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_down_emitter_rotation), Vec3d(outlet_down_buffer_translation)), outlet_down_buffer_half));
    fluid_dynamics::BidirectionalBuffer<RightInflowPressure> right_down_buffer(right_down_emitter, in_outlet_particle_buffer);
    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_wall_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<fluid_dynamics::NonPrescribedPressure>> left_inflow_pressure_condition(left_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightInflowPressure>> right_up_inflow_pressure_condition(right_up_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightInflowPressure>> right_down_inflow_pressure_condition(right_down_emitter);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_emitter);

    ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_inlet_transient_flow_rate(left_emitter, Pi * pow(DW_in/2, 2));
    ReduceDynamics<fluid_dynamics::SectionTransientMassFlowRate> compute_inlet_transient_mass_flow_rate(left_emitter, Pi * pow(DW_in/2, 2));
    ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_outlet_up_transient_flow_rate(right_up_emitter, Pi * pow(DW_out_up/2, 2));
    ReduceDynamics<fluid_dynamics::SectionTransientMassFlowRate> compute_outlet_up_transient_mass_flow_rate(right_up_emitter, Pi * pow(DW_out_up/2, 2));
    ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_outlet_down_transient_flow_rate(right_down_emitter, Pi * pow(DW_out_down/2, 2));
    ReduceDynamics<fluid_dynamics::SectionTransientMassFlowRate> compute_outlet_down_transient_mass_flow_rate(right_down_emitter, Pi * pow(DW_out_down/2, 2));

    InteractionWithUpdate<solid_dynamics::WallShearStress> viscous_force_from_fluid(wall_contact);
    SimpleDynamics<solid_dynamics::FirstLayerFromFluid> solid_first_layer(wall_boundary, water_block);
    SimpleDynamics<solid_dynamics::HemodynamicIndiceCalculation> hemodynamic_indice_calculation(wall_boundary, 0.5);
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
    body_states_recording.addToWrite<int>(wall_boundary, "SolidFirstLayerIndicator");
    body_states_recording.addToWrite<Vecd>(wall_boundary, "NormalDirection");
    body_states_recording.addToWrite<Vecd>(wall_boundary, "WallShearStress");
    body_states_recording.addToWrite<Real>(wall_boundary, "TimeAveragedWallShearStress");
    body_states_recording.addToWrite<Real>(wall_boundary, "OscillatoryShearIndex");

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
    solid_first_layer.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 2.5;   /**< End time. */
    //Real Output_Time = end_time / 300; /**< Time stamps for output of body states. */
    Real Output_Time = 0.01; /**< Time stamps for output of body states. */
    Real dt = 0.0;          /**< Default acoustic time step sizes. */
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

            /** FSI for viscous force. */
            viscous_force_from_fluid.exec();
            hemodynamic_indice_calculation.exec(Dt);

            interval_computing_time_step += TickCount::now() - time_instance;
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);

                pressure_relaxation.exec(dt);

                kernel_summation.exec();
                left_inflow_pressure_condition.exec(dt);
                right_up_inflow_pressure_condition.exec(dt);
                right_down_inflow_pressure_condition.exec(dt);
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

            compute_inlet_transient_flow_rate.exec();
            compute_inlet_transient_mass_flow_rate.exec();
            compute_outlet_up_transient_flow_rate.exec();
            compute_outlet_up_transient_mass_flow_rate.exec();
            compute_outlet_down_transient_flow_rate.exec();
            compute_outlet_down_transient_mass_flow_rate.exec();

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
            wall_contact.updateConfiguration();

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
