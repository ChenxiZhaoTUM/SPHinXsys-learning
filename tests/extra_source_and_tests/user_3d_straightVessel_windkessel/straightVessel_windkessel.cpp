/**
 * @file 	straightVessel_windkessel_rigidShell.cpp
 */

#include "sphinxsys.h" 
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "pressure_boundary.h"
#include "bidirectional_buffer.h"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
/**
 * @brief Namespace cite here.
 */
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
//Real mu_f = 4.0e-3;
//Real Re =  rho0_f * U_f * diameter / mu_f;
Real Re =  100;
Real mu_f = rho0_f * U_f * diameter / Re;

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
//	Fluid body definition.
//----------------------------------------------------------------------
int SimTK_resolution = 20;
class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), diameter / 2,
                                       DL * 0.5, SimTK_resolution,
                                       translation_fluid);
    }
};

//----------------------------------------------------------------------
//	Wall boundary body definition.
//----------------------------------------------------------------------
class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), diameter/2 + wall_thickness,
                                       DL * 0.5, SimTK_resolution,
                                       translation_fluid);
        subtract<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1., 0., 0.), diameter/2,
                                            DL * 0.5, SimTK_resolution,
                                            translation_fluid);
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
    Real u_ave;
    Real t_ref_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ave(0.0), t_ref_(1.0) {}

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

/*the following three classses are for windkessel model*/
class DisposerOutflowDeletionWithWindkessel: public fluid_dynamics::DisposerOutflowDeletion
{
  public:
    DisposerOutflowDeletionWithWindkessel(BodyAlignedBoxByCell &aligned_box_part, int axis) : 
        DisposerOutflowDeletion(aligned_box_part, axis), flow_rate_(0.0), Vol_(particles_->Vol_){};
    virtual ~DisposerOutflowDeletionWithWindkessel(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        mutex_switch_to_buffer_.lock();
        while (aligned_box_.checkUpperBound(axis_, pos_[index_i]) && index_i < particles_->total_real_particles_)
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
    FlowPressureBuffer(BodyPartByCell &body_part, Vecd normal_vector) : BaseFlowBoundaryCondition(body_part), 
        kernel_sum_(*particles_->getVariableByName<Vecd>("KernelSummation")), direction_(normal_vector){};
    virtual ~FlowPressureBuffer(){};
    void update(size_t index_i, Real dt = 0.0)
    {
        vel_[index_i] += 2.0 * kernel_sum_[index_i] * getTargetPressure(dt) / rho_[index_i] * dt;
        vel_[index_i] = vel_[index_i].dot(direction_) * direction_;
    };

  protected:
    StdLargeVec<Vecd> &kernel_sum_;
    Vecd direction_;

    virtual Real getTargetPressure(Real dt) = 0;
};

class OutflowPressure : public FlowPressureBuffer
{
  public:
    OutflowPressure(BodyPartByCell &constrained_region, const std::string &body_part_name, Vecd normal_vector, DisposerOutflowDeletionWithWindkessel &outlet_windkessel,
                    Real R1, Real R2, Real C, Real delta_t, Real average_Q, Real Q_0 = 0.0, Real Q_n = 0.0, Real p_0 = 0.0, Real p_n = 0.0,
                    Real current_flow_rate = 0.0, Real previous_flow_rate = 0.0, int count = 1)
        : FlowPressureBuffer(constrained_region, normal_vector), flow_rate_(outlet_windkessel.flow_rate_), 
        R1_(R1), R2_(R2), C_(C), delta_t_(delta_t), average_Q_(average_Q), Q_0_(Q_0), Q_n_(Q_n), p_0_(p_0), p_n_(p_n), 
        current_flow_rate_(current_flow_rate), previous_flow_rate_(previous_flow_rate), count_(count),
        body_part_name_(body_part_name), write_data_(false) {};
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
            out_file << GlobalStaticVariables::physical_time_ << "   " << p_n_ <<  "\n";
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

/**
 * @brief 	Main program starts here.
 */
int main(int ac, char *av[])
{
    /**
     * @brief Build up -- a SPHSystem --
     */
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.setGenerateRegressionData(false);
    /** Tag for run particle relaxation for the initial body fitted distribution. */
    sph_system.setRunParticleRelaxation(false);
    /** Tag for computation start with relaxed body fitted particles distribution. */
    sph_system.setReloadParticles(true);
    /** handle command line arguments. */
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    /**
     * @brief Material property, particles and body creation of fluid.
     */
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    //water_block.defineBodyLevelSetShape();
    water_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? water_block.generateParticles<ParticleGeneratorReload>(water_block.getName())
        : water_block.generateParticles<ParticleGeneratorLattice>();

    /**
     * @brief 	Particle and body creation of wall boundary.
     */
    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineAdaptationRatios(1.15, 2.0);
    //wall_boundary.defineBodyLevelSetShape();
    wall_boundary.defineParticlesAndMaterial<SolidParticles, Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_boundary.generateParticles<ParticleGeneratorReload>(wall_boundary.getName())
        : wall_boundary.generateParticles<ParticleGeneratorLattice>();


    /** topology */
    InnerRelation water_block_inner(water_block);
    ContactRelation water_block_contact(water_block, {&wall_boundary});
    ComplexRelation water_block_complex(water_block_inner, water_block_contact);
    if (sph_system.RunParticleRelaxation())
    {
        /** body topology only for particle relaxation */
        InnerRelation water_block_inner(water_block);
        InnerRelation wall_boundary_inner(wall_boundary);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** Random reset the insert body particle position. */
        SimpleDynamics<RandomizeParticlePosition> water_block_random_inserted_body_particles(water_block);
        SimpleDynamics<RandomizeParticlePosition> wall_boundary_random_inserted_body_particles(wall_boundary);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp water_block_write_inserted_body_to_vtp({&water_block});
        BodyStatesRecordingToVtp wall_boundary_write_inserted_body_to_vtp({&wall_boundary});
        /** Write the particle reload files. */
        ReloadParticleIO water_block_write_particle_reload_files({&water_block});
        ReloadParticleIO wall_boundary_write_particle_reload_files({&wall_boundary});
        /** A  Physics relaxation step. */
        RelaxationStepInner water_block_relaxation_step_inner(water_block_inner);
        RelaxationStepInner wall_boundary_relaxation_step_inner(wall_boundary_inner);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        water_block_random_inserted_body_particles.exec(0.25);
        wall_boundary_random_inserted_body_particles.exec(0.25);
        water_block_relaxation_step_inner.SurfaceBounding().exec();
        wall_boundary_relaxation_step_inner.SurfaceBounding().exec();
        water_block_write_inserted_body_to_vtp.writeToFile(0);
        wall_boundary_write_inserted_body_to_vtp.writeToFile(0);
        //----------------------------------------------------------------------
        //	Relax particles of the insert body.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 3000)
        {
            water_block_relaxation_step_inner.exec();
            wall_boundary_relaxation_step_inner.exec();
            ite_p += 1;
            if (ite_p % 500 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
                water_block_write_inserted_body_to_vtp.writeToFile(ite_p);
                wall_boundary_write_inserted_body_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of inserted body finish !" << std::endl;
        /** Output results. */
        water_block_write_particle_reload_files.writeToFile(0);
        wall_boundary_write_particle_reload_files.writeToFile(0);
        return 0;
    }
    /**
     * @brief 	Define all numerical methods which are used in this case.
     */
    /**
     * @brief 	Methods used for time stepping.
     */
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);

    BodyAlignedBoxByCell disposer_inlet(water_block, makeShared<AlignedBoxShape>
        (Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)),Vecd(left_bidirectional_translation)), bidirectional_buffer_halfsize));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> disposer_outflow_deletion_inlet(disposer_inlet, xAxis);

    BodyAlignedBoxByCell disposer_outlet(water_block, makeShared<AlignedBoxShape>
        (Transform(Vecd(right_bidirectional_translation)), bidirectional_buffer_halfsize));
    SimpleDynamics<DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_outlet(disposer_outlet, xAxis);


    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex>
        free_stream_surface_indicator(water_block_inner, water_block_contact);

    BodyAlignedBoxByCell inflow_emitter(water_block, makeShared<AlignedBoxShape>
        (Transform(Vecd(left_bidirectional_translation)), bidirectional_buffer_halfsize));
    fluid_dynamics::NonPrescribedPressureBidirectionalBuffer inflow_injection(inflow_emitter, xAxis, 10);

    BodyAlignedBoxByCell outflow_emitter(water_block, makeShared<AlignedBoxShape>
        (Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vecd(right_bidirectional_translation)), bidirectional_buffer_halfsize));
    fluid_dynamics::NonPrescribedPressureBidirectionalBuffer outflow_injection(outflow_emitter, xAxis, 10);

    water_block.addBodyStateForRecording<Real>("Pressure");
    water_block.addBodyStateForRecording<Real>("PositionDivergence");
    water_block.addBodyStateForRecording<int>("Indicator");
    water_block.addBodyStateForRecording<Real>("Density");
    water_block.addBodyStateForRecording<int>("BufferParticleIndicator");
    
    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_block_contact);

    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_block_contact);
    /* Time step size without considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    /** Time step size with considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    /** Pressure relaxation algorithm without Riemann solver for viscous flows. */
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_block_contact);
    /** Pressure relaxation algorithm by using position verlet time stepping. */
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_block_contact);

    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> emitter_buffer_inflow_condition(inflow_emitter);

    BodyRegionByCell outflow_pressure_region(water_block, makeShared<TransformShape<GeometricShapeBox>>
        (Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vecd(right_bidirectional_translation)), bidirectional_buffer_halfsize));
    SimpleDynamics<OutflowPressure> outflow_pressure_condition(outflow_pressure_region, "Outlet", Vecd(1.0, 0.0, 0.0), disposer_outflow_deletion_outlet,
        R1, R2, C, 0.00975, 1.0e-4 ); //8.273e-5

    /** Computing viscous acceleration. */
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_block_contact);
    /** Impose transport velocity. */
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>>
        transport_velocity_correction(water_block_inner, water_block_contact);
    /**
     * @brief Output.
     */
    
    /** Output the body states. */
    BodyStatesRecordingToVtp body_states_recording(sph_system.real_bodies_);
    /**
     * @brief Setup geometry and initial conditions.
     */
    sph_system.initializeSystemCellLinkedLists(); 
    sph_system.initializeSystemConfigurations();
    free_stream_surface_indicator.exec();
    inflow_injection.tag_buffer_particles.exec();
    outflow_injection.tag_buffer_particles.exec();

    wall_boundary_normal_direction.exec();
    
    /**
     * @brief 	Basic parameters.
     */
    size_t number_of_iterations = 0.0;
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 10.0;   /**< End time. */
    Real Output_Time = 0.1; /**< Time stamps for output of body states. */
    Real dt = 0.0;          /**< Default acoustic time step sizes. */
    /** statistics for computing CPU time. */
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    TimeInterval interval_computing_time_step;
    TimeInterval interval_computing_pressure_relaxation;
    TimeInterval interval_updating_configuration;
    TickCount time_instance;
    int record_n = 0;

    /** Output the start states of bodies. */
    body_states_recording.writeToFile(0);
    /**
     * @brief 	Main loop starts here.
    */
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
            /** Dynamics including pressure relaxation. */
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);
                pressure_relaxation.exec(dt);
                kernel_summation.exec();
                emitter_buffer_inflow_condition.exec();                
                outflow_pressure_condition.getFlowRate();
                outflow_pressure_condition.exec(dt);           
                density_relaxation.exec(dt);

                // After the loop, write pressure data once
                if (GlobalStaticVariables::physical_time_ >= record_n * 0.00975)
                {
                    outflow_pressure_condition.writeOutletPressureData();

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
                          << "	Dt = " << Dt << "	dt = " << dt << "\n";
            }
            number_of_iterations++;
            /** Update cell linked list and configuration. */
            time_instance = TickCount::now();
            /** Water block configuration and periodic condition. */
            inflow_injection.injection.exec();
            outflow_injection.injection.exec();
            disposer_outflow_deletion_inlet.exec();
            disposer_outflow_deletion_outlet.exec();

            water_block.updateCellLinkedList();
            water_block_contact.updateConfiguration();
            water_block_complex.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;
            free_stream_surface_indicator.exec();
            inflow_injection.tag_buffer_particles.exec();
            outflow_injection.tag_buffer_particles.exec();
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
