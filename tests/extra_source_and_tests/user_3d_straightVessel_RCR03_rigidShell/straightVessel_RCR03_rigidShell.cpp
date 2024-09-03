/**
 * @file 	straight_vessel_with_shell.cpp
 * @brief 	3D blood flow in straigt vessel with PIPO and shell
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
Real DL = 0.3;                                             /**< Channel length. */
Real diameter = 0.04;                                             /**< Channel height. */
Real fluid_radius = diameter / 2;
Real resolution_ref = diameter / 20.0;                             /**< Initial reference particle spacing. */
Real resolution_shell = resolution_ref;
Real wall_thickness = resolution_shell * 1.0;
Vec3d translation_fluid(DL * 0.5, 0., 0.);
BoundingBox system_domain_bounds(Vecd(0, -fluid_radius - wall_thickness, -fluid_radius - wall_thickness),
                                 Vecd(DL, fluid_radius + wall_thickness, fluid_radius + wall_thickness));
//----------------------------------------------------------------------
//	Material parameters.
//----------------------------------------------------------------------
Real rho0_f = 1060.0;
Real U_f = 0.2;
Real c_f = 10.0 * U_f;
Real mu_f = 4.0e-3;
Real Re =  rho0_f * U_f * diameter / mu_f;
//Real Re =  1000;
//Real mu_f = rho0_f * U_f * diameter / Re;

Real R1 = 1.21e7;
Real R2 = 1.212e8;
Real C = 1.5e-10;
//----------------------------------------------------------------------
//	parameters of buffers
//----------------------------------------------------------------------
Vecd bidirectional_buffer_halfsize = Vecd(2.0 * resolution_ref, 0.7 * diameter, 0.7 * diameter);
Vecd left_bidirectional_translation(2.0 * resolution_ref, 0.0, 0.0);
Vecd right_bidirectional_translation(DL - 2.0 * resolution_ref, 0.0, 0.0);
//----------------------------------------------------------------------
//  Define body shapes
//----------------------------------------------------------------------
int SimTK_resolution = 20;
class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), fluid_radius,
                                       DL * 0.5, SimTK_resolution,
                                       translation_fluid);
    }
};

class ShellShape : public ComplexShape
{
  public:
    explicit ShellShape(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), fluid_radius + wall_thickness,
                                       DL * 0.5, SimTK_resolution,
                                       translation_fluid);
        subtract<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), fluid_radius,
                                            DL * 0.5, SimTK_resolution,
                                            translation_fluid);
    }
};

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
        const Real radius_mid_surface = fluid_radius + resolution_shell_ * 0.5;
        const auto particle_number_mid_surface =
            int(2.0 * radius_mid_surface * Pi / resolution_shell_);
        const auto particle_number_height =
            int(DL / resolution_shell_);
        for (int i = 0; i < particle_number_mid_surface; i++)
        {
            for (int j = 0; j < particle_number_height; j++)
            {
                Real theta = (i + 0.5) * 2 * Pi / (Real)particle_number_mid_surface;
                Real x = DL * j / (Real)particle_number_height + 0.5 * resolution_shell_;
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
//	Pressure boundary definition.
//----------------------------------------------------------------------
struct LeftInflowPressure
{
    template <class BoundaryConditionType>
    LeftInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real &p_)
    {
        return p_;
    }
};
//----------------------------------------------------------------------
//	inflow velocity definition.
//----------------------------------------------------------------------
struct InflowVelocity
{
    Real t_ref_;
    AlignedBoxShape &aligned_box_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : t_ref_(1.0),
          aligned_box_(boundary_condition.getAlignedBox()) {}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = velocity;
        Real run_time = GlobalStaticVariables::physical_time_;
        int n = static_cast<int>(run_time / t_ref_);
        Real t_in_cycle = run_time - n * t_ref_;

        target_velocity[0] = 130 * sin(Pi * t_in_cycle) * 1.0e-6 / (Pi * pow(fluid_radius, 2));

        return target_velocity;
    }
};

class DisposerOutflowDeletionWithWindkessel: public fluid_dynamics::DisposerOutflowDeletion
{
  public:
    DisposerOutflowDeletionWithWindkessel(BodyAlignedBoxByCell &aligned_box_part) : 
        DisposerOutflowDeletion(aligned_box_part), flow_rate_(0.0), Vol_(*particles_->getVariableDataByName<Real>("VolumetricMeasure")){};
    virtual ~DisposerOutflowDeletionWithWindkessel(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        mutex_switch_to_buffer_.lock();
        while (aligned_box_.checkUpperBound(pos_[index_i]) && index_i < particles_->TotalRealParticles())
        {
            particles_->switchToBufferParticle(index_i);
            flow_rate_ += Vol_[index_i];
        }
        mutex_switch_to_buffer_.unlock();
    };
    Real flow_rate_;

  protected:
    StdLargeVec<Real> &Vol_;
};

class FlowPressureBuffer : public fluid_dynamics::BaseFlowBoundaryCondition
{
  public:
    FlowPressureBuffer(BodyAlignedBoxByCell &aligned_box_part) : BaseFlowBoundaryCondition(aligned_box_part),
        aligned_box_(aligned_box_part.getAlignedBoxShape()),
        alignment_axis_(aligned_box_.AlignmentAxis()),
        transform_(aligned_box_.getTransform()),
        kernel_sum_(*particles_->getVariableDataByName<Vecd>("KernelSummation")){};
    virtual ~FlowPressureBuffer(){};
    void update(size_t index_i, Real dt = 0.0)
    {
        vel_[index_i] += 2.0 * kernel_sum_[index_i] * getTargetPressure(dt) / rho_[index_i] * dt;
        Vecd frame_velocity = Vecd::Zero();
        frame_velocity[alignment_axis_] = transform_.xformBaseVecToFrame(vel_[index_i])[alignment_axis_];
        vel_[index_i] = transform_.xformFrameVecToBase(frame_velocity);
    };

  protected:
    AlignedBoxShape &aligned_box_;
    const int alignment_axis_;
    Transform &transform_;
    StdLargeVec<Vecd> &kernel_sum_;

    virtual Real getTargetPressure(Real dt) = 0;
};

class OutflowPressure : public FlowPressureBuffer
{
  public:
    OutflowPressure(BodyAlignedBoxByCell &aligned_box_part, const std::string &body_part_name, DisposerOutflowDeletionWithWindkessel &outlet_windkessel,
                    Real R1, Real R2, Real C, Real delta_t, Real average_Q, Real Q_0 = 0.0, Real Q_n = 0.0, Real p_0 = 0.0, Real p_n = 0.0,
                    Real current_flow_rate = 0.0, Real previous_flow_rate = 0.0, int count = 1)
        : FlowPressureBuffer(aligned_box_part), flow_rate_(outlet_windkessel.flow_rate_),
          R1_(R1), R2_(R2), C_(C), delta_t_(delta_t), average_Q_(average_Q), Q_0_(Q_0), Q_n_(Q_n), p_0_(p_0), p_n_(p_n),
          current_flow_rate_(current_flow_rate), previous_flow_rate_(previous_flow_rate), count_(count),
          body_part_name_(body_part_name), write_data_(false){};
    virtual ~OutflowPressure(){};

    void getFlowRate()
    {
        Real run_time = GlobalStaticVariables::physical_time_;

        if (int(run_time / delta_t_) > count_)
        {
            Q_0_ = Q_n_;
            p_0_ = p_n_;
            current_flow_rate_ = flow_rate_ - previous_flow_rate_;
            previous_flow_rate_ = flow_rate_;
            count_ += 1;
        }
    };

    Real getTargetPressure(Real dt) override
    {
        Q_n_ = current_flow_rate_ / delta_t_ - average_Q_;
        p_n_ = ((Q_n_ * (1.0 + R1_ / R2_) + C_ * R1_ * (Q_n_ - Q_0_) / delta_t_) * delta_t_ / C_ + p_0_) / (1.0 + delta_t_ / (C_ * R2_));
        // std::cout << "p_n_ = " << p_n_ << std::endl;

        write_data_ = true;

        return p_n_;
    }

    void writeOutletPressureData()
    {
        if (write_data_)
        {
            std::string output_folder = "./output";
            std::string filefullpath = output_folder + "/" + body_part_name_ + "_outlet_pressure.dat";
            std::ofstream out_file(filefullpath.c_str(), std::ios::app);
            out_file << GlobalStaticVariables::physical_time_ << "   " << p_n_ << "\n";
            out_file.close();

            write_data_ = false; // Reset the flag after writing
        }
    }

    void setupDynamics(Real dt = 0.0) override {}

  protected:
    Real &flow_rate_, R1_, R2_, C_, delta_t_, average_Q_, Q_0_, Q_n_, p_0_, p_n_, current_flow_rate_, previous_flow_rate_;
    int count_;
    std::string body_part_name_;
    bool write_data_;
};

int main(int ac, char *av[])
{
    //std::cout << "U_f = " << U_f << std::endl;

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
    water_block.defineBodyLevelSetShape();
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
    : water_block.generateParticles<BaseParticles, Lattice>();

    SolidBody shell_body(sph_system, makeShared<ShellShape>("ShellBody"));
    shell_body.defineAdaptation<SPHAdaptation>(1.15, resolution_ref / resolution_shell);
    shell_body.defineMaterial<Solid>();
    shell_body.generateParticles<SurfaceParticles, ShellBoundary>(resolution_shell, wall_thickness);

    if (sph_system.RunParticleRelaxation())
    {
        /** body topology only for particle relaxation */;
        InnerRelation water_block_inner(water_block);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** Random reset the insert body particle position. */
        SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_body_to_vtp(sph_system);
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
        while (ite_p < 2000)
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
    InnerRelation shell_inner(shell_body);
    ContactRelationFromShellToFluid water_shell_contact(water_block, {&shell_body}, {false});
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
    SimpleDynamics<thin_structure_dynamics::AverageShellCurvature> shell_average_curvature(shell_curvature_inner);

    // fluid dynamics
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_shell_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_shell_contact);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_shell_contact);

    // Boundary conditions
    AlignedBoxShape left_disposer_shape(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vecd(left_bidirectional_translation)), bidirectional_buffer_halfsize);
    BodyAlignedBoxByCell left_disposer(water_block, left_disposer_shape);
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_disposer_outflow_deletion(left_disposer);

    AlignedBoxShape right_disposer_shape(xAxis, Transform(Vecd(right_bidirectional_translation)), bidirectional_buffer_halfsize);
    BodyAlignedBoxByCell right_disposer(water_block, right_disposer_shape);
    SimpleDynamics<DisposerOutflowDeletionWithWindkessel> right_disposer_outflow_deletion(right_disposer);

    AlignedBoxShape left_emitter_shape(xAxis, Transform(Vecd(left_bidirectional_translation)), bidirectional_buffer_halfsize);
    BodyAlignedBoxByCell left_emitter(water_block, left_emitter_shape);
    fluid_dynamics::NonPrescribedPressureBidirectionalBuffer left_emitter_inflow_injection(left_emitter, in_outlet_particle_buffer);

    AlignedBoxShape right_emitter_shape(xAxis, Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vecd(right_bidirectional_translation)), bidirectional_buffer_halfsize);
    BodyAlignedBoxByCell right_emitter(water_block, right_emitter_shape);
    //ReduceDynamics<fluid_dynamics::AverageFlowRate<fluid_dynamics::TotalVelocityNormVal>> compute_flow_rate(right_emitter, Pi * pow(fluid_radius, 2));
    //fluid_dynamics::BidirectionalBufferWindkessel<fluid_dynamics::RCRPressure> right_emitter_inflow_injection(right_emitter, in_outlet_particle_buffer, R1, R2, C);
    fluid_dynamics::NonPrescribedPressureBidirectionalBuffer right_emitter_inflow_injection(right_emitter, in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_shell_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<LeftInflowPressure>> left_inflow_pressure_condition(left_emitter);
    SimpleDynamics<OutflowPressure> right_inflow_pressure_condition(right_emitter, "out01", right_disposer_outflow_deletion, 
        R1, R2, C, 8.273e-5, 0.00975);

    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_emitter);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations, observations
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<Vecd>(shell_body, "NormalDirection");
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
    
    boundary_indicator.exec();
    left_emitter_inflow_injection.tag_buffer_particles.exec();
    right_emitter_inflow_injection.tag_buffer_particles.exec();

    water_block_complex.updateConfiguration();
    //shell_water_contact.updateConfiguration();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = sph_system.RestartStep();
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 10;   /**< End time. */
    Real Output_Time = end_time/100; /**< Time stamps for output of body states. */
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
    int record_n = 0;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    body_states_recording.writeToFile();
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
                
                left_inflow_pressure_condition.exec(dt);

                // windkessel model implementation
                right_inflow_pressure_condition.getFlowRate();
                right_inflow_pressure_condition.exec(dt);

                inflow_velocity_condition.exec();

                density_relaxation.exec(dt);

                // After the loop, write pressure data once
                if (GlobalStaticVariables::physical_time_ >= record_n * 0.006)
                {
                    right_inflow_pressure_condition.writeOutletPressureData();
                    ++record_n;
                }

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
            }
            number_of_iterations++;

            time_instance = TickCount::now();

            left_emitter_inflow_injection.injection.exec();
            right_emitter_inflow_injection.injection.exec();
            left_disposer_outflow_deletion.exec();
            right_disposer_outflow_deletion.exec();

            water_block.updateCellLinkedList();
            water_block_complex.updateConfiguration();
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
