/**
 * @file 	carotid_VIPO_shell.cpp
 * @brief 	Carotid artery with shell, imposed velocity inlet and pressure outlet condition.
 */

#include "sphinxsys.h"
#include "bidirectional_buffer.h"
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "pressure_boundary.h"
#include "particle_generation_and_detection.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_file = "./input/carotid_fluid_geo.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real scaling = pow(10, -3);
Vec3d domain_lower_bound(-6.0 * scaling, -4.0 * scaling, -32.5 * scaling);
Vec3d domain_upper_bound(12.0 * scaling, 10.0 * scaling, 23.5 * scaling);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 0.2 * scaling;
Real shell_resolution = dp_0 / 2;  /*thickness = 1.0 * shell_resolution*/ 
//----------------------------------------------------------------------
//	define the imported model.
//----------------------------------------------------------------------
class ShellShape : public ComplexShape
{
public:
    explicit ShellShape(const std::string &shape_name) : ComplexShape(shape_name),
        mesh_shape_(new TriangleMeshShapeSTL(full_path_to_file, translation, scaling))
    {
        //add<ExtrudeShape<TriangleMeshShapeSTL>>(thickness, full_path_to_file, translation, scaling);
        add<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
    }

    TriangleMeshShapeSTL* getMeshShape() const
    {
        return mesh_shape_.get();
    }

private:
    std::unique_ptr<TriangleMeshShapeSTL> mesh_shape_;
};

class WaterBlock : public ComplexShape
{
public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        //add<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
        add<ExtrudeShape<TriangleMeshShapeSTL>>(-shell_resolution/2, full_path_to_file, translation, scaling);
    }
};
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

// inlet R=2.9293, (1.5611, 5.8559, -30.8885), (0.1034, -0.0458, 0.9935)
Real DW_in = 2.9293 * 2 * scaling;
Vec3d inlet_half = Vec3d(2.0 * dp_0, 3.5 * scaling, 3.5 * scaling);
Vec3d inlet_normal(0.1034, -0.0458, 0.9935);
Vec3d inlet_cut_translation = Vec3d(1.5611, 5.8559, -30.8885) * scaling - inlet_normal * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d inlet_buffer_translation = Vec3d(1.5611, 5.8559, -30.8885) * scaling + inlet_normal * 2.0 * dp_0;
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, standard_direction);
Rotation3d inlet_emitter_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);
Rotation3d inlet_disposer_rotation(inlet_rotation_result.angle + Pi, inlet_rotation_result.axis);

// outlet1 R=1.9416, (-2.6975, -0.4330, 21.7855), (-0.3160, -0.0009, 0.9488)
Real DW_out_up = 1.9416 * 2 * scaling;
Vec3d outlet_up_half = Vec3d(2.0 * dp_0, 2.4 * scaling, 2.4 * scaling);
Vec3d outlet_up_normal(-0.3160, -0.0009, 0.9488);
Vec3d outlet_up_cut_translation = Vec3d(-2.6975, -0.4330, 21.7855) * scaling + outlet_up_normal * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_up_buffer_translation = Vec3d(-2.6975, -0.4330, 21.7855) * scaling - outlet_up_normal * 2.0 * dp_0;
RotationResult outlet_up_rotation_result = RotationCalculator(outlet_up_normal, standard_direction);
Rotation3d outlet_up_disposer_rotation(outlet_up_rotation_result.angle, outlet_up_rotation_result.axis);
Rotation3d outlet_up_emitter_rotation(outlet_up_rotation_result.angle + Pi, outlet_up_rotation_result.axis);

// outlet2 R=1.3261, (9.0220, 0.9750, 18.6389), (-0.0399, 0.0693, 0.9972)
Real DW_out_down = 1.3261 * 2 * scaling;
Vec3d outlet_down_half = Vec3d(2.0 * dp_0, 2.0 * scaling, 2.0 * scaling);
Vec3d outlet_down_normal(-0.0399, 0.0693, 0.9972);
Vec3d outlet_down_cut_translation = Vec3d(9.0220, 0.9750, 18.6389) * scaling + outlet_down_normal * (1.0 * dp_0 + 1.0 * (dp_0 - shell_resolution));
Vec3d outlet_down_buffer_translation = Vec3d(9.0220, 0.9750, 18.6389) * scaling - outlet_down_normal * 2.0 * dp_0;
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
Real mu_f = 0.00355; /**< Dynamics viscosity. */
Real Outlet_pressure = 0;

Real rho0_s = 1120;                /** Normalized density. */
Real Youngs_modulus = 1.08e6;    /** Normalized Youngs Modulus. */
Real poisson = 0.49;               /** Poisson ratio. */
//Real physical_viscosity = 0.25 * sqrt(rho0_s * Youngs_modulus) * 55.0 * scaling; /** physical damping */
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
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

class BoundaryGeometry : public BodyPartByParticle
{
  public:
    BoundaryGeometry(SPHBody &body, const std::string &body_part_name)
        : BodyPartByParticle(body, body_part_name)
    {
        TaggingParticleMethod tagging_particle_method = std::bind(&BoundaryGeometry::tagManually, this, _1);
        tagParticles(tagging_particle_method);
    };
    virtual ~BoundaryGeometry(){};

  private:
    void tagManually(size_t index_i)
    {
        if (base_particles_.ParticlePositions()[index_i][2] < -30.8885 * scaling + 0.9935 * 4 * dp_0
            || (base_particles_.ParticlePositions()[index_i][2] > 21.7855 * scaling - 0.9488 * 4 * dp_0
                && base_particles_.ParticlePositions()[index_i][0] <= 0)
            || (base_particles_.ParticlePositions()[index_i][2] > 18.6389 * scaling - 0.9972 * 4 * dp_0
                && base_particles_.ParticlePositions()[index_i][0] > 0))
        {
            body_part_particles_.push_back(index_i);
        }
    };
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
    sph_system.setRunParticleRelaxation(true); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(false);       // Tag for computation with save particles distribution
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------
    ShellShape body_from_mesh("BodyFromMesh");
    TriangleMeshShapeSTL* mesh_shape = body_from_mesh.getMeshShape();
    SolidBody shell_body(sph_system, makeShared<ShellShape>("ShellBody"));
    shell_body.defineAdaptation<SPHAdaptation>(1.15, dp_0/shell_resolution);
    shell_body.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
    shell_body.defineMaterial<Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? shell_body.generateParticles<SurfaceParticles, Reload>(shell_body.getName())
        : shell_body.generateParticles<SurfaceParticles, FromSTLFile>(mesh_shape);

    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape()->cleanLevelSet();
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    //water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
    : water_block.generateParticles<BaseParticles, Lattice>();
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation shell_inner(shell_body);
        InnerRelation blood_inner(water_block);

        BodyAlignedBoxByCell inlet_detection_box(shell_body,
                                             makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_half));
        BodyAlignedBoxByCell outlet01_detection_box(shell_body,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_up_emitter_rotation), Vec3d(outlet_up_cut_translation)), outlet_up_half));
        BodyAlignedBoxByCell outlet02_detection_box(shell_body,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_down_emitter_rotation), Vec3d(outlet_down_cut_translation)), outlet_down_half));

        RealBody test_body_in(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_half, "TestBodyIn"));
        test_body_in.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_up(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_up_emitter_rotation), Vec3d(outlet_up_cut_translation)), outlet_up_half, "TestBodyOutUp"));
        test_body_out_up.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_down(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_down_emitter_rotation), Vec3d(outlet_down_cut_translation)), outlet_down_half, "TestBodyOutDown"));
        test_body_out_down.generateParticles<BaseParticles, Lattice>();
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** A  Physics relaxation step. */
        SurfaceRelaxationStep relaxation_step_inner(shell_inner);
        ShellNormalDirectionPrediction shell_normal_prediction(shell_inner, shell_resolution * 1.0);

        RelaxationStepInner relaxation_step_inner_blood(blood_inner);

        // here, need a class to switch particles in aligned box to ghost particles (not real particles)
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet01_particles_detection(outlet01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet02_particles_detection(outlet02_detection_box);

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_shell_to_vtp({shell_body});
        write_shell_to_vtp.addToWrite<Vecd>(shell_body, "NormalDirection");
        BodyStatesRecordingToVtp write_blood_to_vtp({water_block});
        BodyStatesRecordingToVtp write_all_bodies_to_vtp({sph_system});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files({ &shell_body, &water_block });
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        relaxation_step_inner.getOnSurfaceBounding().exec();
        relaxation_step_inner_blood.SurfaceBounding().exec();
        write_shell_to_vtp.writeToFile(0.0);
        write_blood_to_vtp.writeToFile(0.0);
        shell_body.updateCellLinkedList();
        //----------------------------------------------------------------------
        //	Particle relaxation time stepping start here.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 2000)
        {
            relaxation_step_inner.exec();
            relaxation_step_inner_blood.exec();
            ite_p += 1;
            if (ite_p % 500 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";
                write_shell_to_vtp.writeToFile(ite_p);
                write_blood_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of imported model finish !" << std::endl;

        shell_normal_prediction.smoothing_normal_exec();

        inlet_particles_detection.exec();
        shell_body.updateCellLinkedList();
        outlet01_particles_detection.exec();
        shell_body.updateCellLinkedList();
        outlet02_particles_detection.exec();
        shell_body.updateCellLinkedList();

        write_all_bodies_to_vtp.writeToFile(ite_p);
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
    InnerRelation shell_inner(shell_body);
    ContactRelationFromShellToFluid water_shell_contact(water_block, {&shell_body}, {false});
    //ContactRelationFromFluidToShell shell_water_contact(shell_body, {&water_block}, {false});
    ShellInnerRelationWithContactKernel shell_curvature_inner(shell_body, water_block);
    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    // which is only used for update configuration.
    //----------------------------------------------------------------------
    ComplexRelation water_block_complex(water_block_inner, {&water_shell_contact});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    // shell dynamics
    //InteractionDynamics<thin_structure_dynamics::ShellCorrectConfiguration> shell_corrected_configuration(shell_inner);
    //Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationFirstHalf> shell_stress_relaxation_first(shell_inner, 3, true);
    //Dynamics1Level<thin_structure_dynamics::ShellStressRelaxationSecondHalf> shell_stress_relaxation_second(shell_inner);
    //ReduceDynamics<thin_structure_dynamics::ShellAcousticTimeStepSize> shell_time_step_size(shell_body);
    SimpleDynamics<thin_structure_dynamics::AverageShellCurvature> shell_average_curvature(shell_curvature_inner);
    //SimpleDynamics<thin_structure_dynamics::UpdateShellNormalDirection> shell_update_normal(shell_body);

    ///** Exert constrain on shell. */
    //BoundaryGeometry boundary_geometry(shell_body, "BoundaryGeometry");
    //SimpleDynamics<thin_structure_dynamics::ConstrainShellBodyRegion> constrain_holder(boundary_geometry);

    // fluid dynamics
    StartupAcceleration time_dependent_acceleration(Vecd(U_f, 0.0, 0.0), 2.0);
    SimpleDynamics<GravityForce<StartupAcceleration>> apply_initial_force(water_block, time_dependent_acceleration);
    
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_shell_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_shell_contact);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStep> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step_size(water_block);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_shell_contact);

    // add buffers
    BodyAlignedBoxByCell left_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_buffer_translation)), inlet_half));
    fluid_dynamics::BidirectionalBuffer<fluid_dynamics::NonPrescribedPressure> left_emitter_inflow_injection(left_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_up_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_up_emitter_rotation), Vec3d(outlet_up_buffer_translation)), outlet_up_half));
    fluid_dynamics::BidirectionalBuffer<RightOutflowPressure> right_up_emitter_inflow_injection(right_up_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_down_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_down_emitter_rotation), Vec3d(outlet_down_buffer_translation)), outlet_down_half));
    fluid_dynamics::BidirectionalBuffer<RightOutflowPressure> right_down_emitter_inflow_injection(right_down_emitter, in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_shell_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<fluid_dynamics::NonPrescribedPressure>> left_inflow_pressure_condition(left_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightOutflowPressure>> right_up_inflow_pressure_condition(right_up_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightOutflowPressure>> right_down_inflow_pressure_condition(right_down_emitter);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_emitter);

    // FSI
    /*InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_on_shell(shell_water_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_on_shell(shell_water_contact);
    solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(shell_body);*/
    //----------------------------------------------------------------------
    //	Define the configuration related particles dynamics.
    //----------------------------------------------------------------------
    ParticleSorting particle_sorting(water_block);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //	and regression tests of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    body_states_recording.addToWrite<Vecd>(shell_body, "NormalDirection");
    //body_states_recording.addToWrite<Vecd>(shell_body, "PressureForceFromFluid");
    body_states_recording.addToWrite<Real>(shell_body, "Average1stPrincipleCurvature");
    body_states_recording.addToWrite<Real>(shell_body, "Average2ndPrincipleCurvature");
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    //shell_corrected_configuration.exec();
    shell_average_curvature.exec();
    //constrain_holder.exec();
    water_block_complex.updateConfiguration();
    //shell_water_contact.updateConfiguration();
    boundary_indicator.exec();
    left_emitter_inflow_injection.tag_buffer_particles.exec();
    right_up_emitter_inflow_injection.tag_buffer_particles.exec();
    right_down_emitter_inflow_injection.tag_buffer_particles.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 2.5;   /**< End time. */
    Real Output_Time = end_time/250; /**< Time stamps for output of body states. */
    Real dt = 0.0;          /**< Default acoustic time step sizes. */
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
            apply_initial_force.exec();

            Real Dt = get_fluid_advection_time_step_size.exec();
            //std::cout << "Dt = " << Dt << std::endl;
            update_fluid_density.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();
            /** FSI for viscous force. */
            //viscous_force_on_shell.exec();

            interval_computing_time_step += TickCount::now() - time_instance;
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);
                //std::cout << "dt = " << dt << std::endl;

                pressure_relaxation.exec(dt);
                /** FSI for pressure force. */
                //pressure_force_on_shell.exec();

                kernel_summation.exec();
                left_inflow_pressure_condition.exec(dt);
                right_up_inflow_pressure_condition.exec(dt);
                right_down_inflow_pressure_condition.exec(dt);
                inflow_velocity_condition.exec();

                density_relaxation.exec(dt);

                /*Real dt_s_sum = 0.0;
                average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    dt_s = shell_time_step_size.exec();
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;
                    shell_stress_relaxation_first.exec(dt_s);

                    constrain_holder.exec(dt_s);

                    shell_stress_relaxation_second.exec(dt_s);
                    dt_s_sum += dt_s;
                }
                average_velocity_and_acceleration.update_averages_.exec(dt);*/

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

            left_emitter_inflow_injection.injection.exec();
            right_up_emitter_inflow_injection.injection.exec();
            right_down_emitter_inflow_injection.injection.exec();
            left_emitter_inflow_injection.deletion.exec();
            right_up_emitter_inflow_injection.deletion.exec();
            right_down_emitter_inflow_injection.deletion.exec();

            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                particle_sorting.exec();
            }

            water_block.updateCellLinkedList();
            //shell_update_normal.exec();
            //shell_body.updateCellLinkedList();
            //shell_curvature_inner.updateConfiguration();
            //shell_average_curvature.exec();
            //shell_water_contact.updateConfiguration();
            water_block_complex.updateConfiguration();

            interval_updating_configuration += TickCount::now() - time_instance;
            boundary_indicator.exec();

            left_emitter_inflow_injection.tag_buffer_particles.exec();
            right_up_emitter_inflow_injection.tag_buffer_particles.exec();
            right_down_emitter_inflow_injection.tag_buffer_particles.exec();
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