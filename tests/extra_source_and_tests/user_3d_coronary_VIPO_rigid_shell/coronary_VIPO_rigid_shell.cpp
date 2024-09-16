/**
 * @file 	carotid_VIPO_shell.cpp
 * @brief 	Carotid artery with shell, imposed velocity inlet and pressure outlet condition.
 */

#include "sphinxsys.h"
#include"coronary_VIPO_rigid_shell.h"

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
    SolidBody shell_body(sph_system, makeShared<SolidBodyFromMesh>("ShellBody"));
    shell_body.defineAdaptation<SPHAdaptation>(1.15, dp_0/shell_resolution);
    shell_body.defineMaterial<Solid>();
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    {
        shell_body.generateParticles<SurfaceParticles, Reload>(shell_body.getName());
    }
    else
    {
        shell_body.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
        shell_body.generateParticles<SurfaceParticles, FromVTPFile>(full_vtp_file_path);
    }

    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    //water_block.generateParticlesWithReserve<BaseParticles, Lattice>(in_outlet_particle_buffer);
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    {
        water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName());
    }
    else
    {
        water_block.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->cleanLevelSet();
        water_block.generateParticles<BaseParticles, Lattice>();
    }
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    if (sph_system.RunParticleRelaxation())
    {
        // for shell
        BodyAlignedCylinderByCell inlet_detection_cylinder(shell_body,
            makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_half[1], inlet_half[0]));
        BodyAlignedBoxByCell outlet_main_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_main), Vec3d(outlet_cut_translation_main)), outlet_half_main));
        BodyAlignedBoxByCell outlet_left01_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_01), Vec3d(outlet_cut_translation_left_01)), outlet_half_left_01));
        BodyAlignedBoxByCell outlet_left02_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_02), Vec3d(outlet_cut_translation_left_02)), outlet_half_left_02));
        BodyAlignedBoxByCell outlet_left03_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_03), Vec3d(outlet_cut_translation_left_03)), outlet_half_left_03));
        BodyAlignedBoxByCell outlet_rightF01_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightF_01), Vec3d(outlet_cut_translation_rightF_01)), outlet_half_rightF_01));
        BodyAlignedBoxByCell outlet_rightF02_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightF_02), Vec3d(outlet_cut_translation_rightF_02)), outlet_half_rightF_02));
        BodyAlignedBoxByCell outlet_rightB01_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_01), Vec3d(outlet_cut_translation_rightB_01)), outlet_half_rightB_01));
        BodyAlignedBoxByCell outlet_rightB02_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_02), Vec3d(outlet_cut_translation_rightB_02)), outlet_half_rightB_02));
        BodyAlignedBoxByCell outlet_rightB03_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_03), Vec3d(outlet_cut_translation_rightB_03)), outlet_half_rightB_03));
        BodyAlignedBoxByCell outlet_rightB04_detection_box(shell_body, 
            makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_04), Vec3d(outlet_cut_translation_rightB_04)), outlet_half_rightB_04));

        InnerRelation shell_inner(shell_body);
        InnerRelation blood_inner(water_block);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        //SimpleDynamics<RandomizeParticlePosition> random_imported_model_particles(imported_model);
        /** A  Physics relaxation step. */
        SurfaceRelaxationStep relaxation_step_inner_shell(shell_inner);
        RelaxationStepInner relaxation_step_inner_blood(blood_inner);
        ShellNormalDirectionPrediction shell_normal_prediction(shell_inner, thickness);

        SimpleDynamics<DeleteParticlesInCylinder> inlet_particles_detection(inlet_detection_cylinder);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_main_particles_detection(outlet_main_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_left01_particles_detection(outlet_left01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_left02_particles_detection(outlet_left02_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_left03_particles_detection(outlet_left03_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightF01_particles_detection(outlet_rightF01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightF02_particles_detection(outlet_rightF02_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightB01_particles_detection(outlet_rightB01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightB02_particles_detection(outlet_rightB02_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightB03_particles_detection(outlet_rightB03_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_rightB04_particles_detection(outlet_rightB04_detection_box);

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
        relaxation_step_inner_shell.getOnSurfaceBounding().exec();
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
            relaxation_step_inner_shell.exec();
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
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_main_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_left01_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_left02_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_left03_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightF01_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightF02_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightB01_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightB02_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightB03_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);
        outlet_rightB04_particles_detection.exec();
        shell_body.updateCellLinkedListWithParticleSort(100);

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
    /*TimeDependentAcceleration time_dependent_acceleration(Vecd::Zero());
    SimpleDynamics<GravityForce> apply_initial_force(water_block, time_dependent_acceleration);*/

    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_shell_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> boundary_indicator(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_shell_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_shell_contact);
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_shell_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> transport_velocity_correction(water_block_inner, water_shell_contact);

    // add buffers
    // disposer
    BodyAlignedCylinderByCell inlet_disposer_cylinder(water_block,
            makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_disposer_rotation), Vec3d(inlet_buffer_translation)), inlet_half[1], inlet_half[0]));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionArb<AlignedCylinderShape>> inlet_disposer_deletion(inlet_disposer_cylinder);
    BodyAlignedBoxByCell outlet_main_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_main), Vec3d(outlet_buffer_translation_main)), outlet_half_main));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> main_disposer_injection(outlet_main_disposer);
    BodyAlignedBoxByCell outlet_left_01_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_left_01), Vec3d(outlet_buffer_translation_left_01)), outlet_half_left_01));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_01_disposer_injection(outlet_left_01_disposer);
    BodyAlignedBoxByCell outlet_left_02_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_left_02), Vec3d(outlet_buffer_translation_left_02)), outlet_half_left_02));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_02_disposer_injection(outlet_left_02_disposer);
    BodyAlignedBoxByCell outlet_left_03_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_left_03), Vec3d(outlet_buffer_translation_left_03)), outlet_half_left_03));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> left_03_disposer_injection(outlet_left_03_disposer);
    BodyAlignedBoxByCell outlet_rightF_01_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightF_01), Vec3d(outlet_buffer_translation_rightF_01)), outlet_half_rightF_01));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightF_01_disposer_injection(outlet_rightF_01_disposer);
    BodyAlignedBoxByCell outlet_rightF_02_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightF_02), Vec3d(outlet_buffer_translation_rightF_02)), outlet_half_rightF_02));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightF_02_disposer_injection(outlet_rightF_02_disposer);
    BodyAlignedBoxByCell outlet_rightB_01_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightB_01), Vec3d(outlet_buffer_translation_rightB_01)), outlet_half_rightB_01));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightB_01_disposer_injection(outlet_rightB_01_disposer);
    BodyAlignedBoxByCell outlet_rightB_02_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightB_02), Vec3d(outlet_buffer_translation_rightB_02)), outlet_half_rightB_02));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightB_02_disposer_injection(outlet_rightB_02_disposer);
    BodyAlignedBoxByCell outlet_rightB_03_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightB_03), Vec3d(outlet_buffer_translation_rightB_03)), outlet_half_rightB_03));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightB_03_disposer_injection(outlet_rightB_03_disposer);
    BodyAlignedBoxByCell outlet_rightB_04_disposer(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_disposer_rotation_rightB_04), Vec3d(outlet_buffer_translation_rightB_04)), outlet_half_rightB_04));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletion> rightB_04_disposer_injection(outlet_rightB_04_disposer);

    // emitter
    BodyAlignedCylinderByCell inlet_emitter_cylinder(water_block,
            makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_buffer_translation)), inlet_half[1], inlet_half[0]));
    fluid_dynamics::NonPrescribedPressureBidirectionalBufferArb<AlignedCylinderShape> inlet_emitter_injection(inlet_emitter_cylinder, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_main_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_main), Vec3d(outlet_buffer_translation_main)), outlet_half_main));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> main_emitter_injection(outlet_main_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_left_01_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_01), Vec3d(outlet_buffer_translation_left_01)), outlet_half_left_01));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> left_01_emitter_injection(outlet_left_01_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_left_02_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_02), Vec3d(outlet_buffer_translation_left_02)), outlet_half_left_02));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> left_02_emitter_injection(outlet_left_02_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_left_03_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_left_03), Vec3d(outlet_buffer_translation_left_03)), outlet_half_left_03));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> left_03_emitter_injection(outlet_left_03_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightF_01_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightF_01), Vec3d(outlet_buffer_translation_rightF_01)), outlet_half_rightF_01));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightF_01_emitter_injection(outlet_rightF_01_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightF_02_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightF_02), Vec3d(outlet_buffer_translation_rightF_02)), outlet_half_rightF_02));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightF_02_emitter_injection(outlet_rightF_02_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightB_01_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_01), Vec3d(outlet_buffer_translation_rightB_01)), outlet_half_rightB_01));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightB_01_emitter_injection(outlet_rightB_01_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightB_02_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_02), Vec3d(outlet_buffer_translation_rightB_02)), outlet_half_rightB_02));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightB_02_emitter_injection(outlet_rightB_02_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightB_03_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_03), Vec3d(outlet_buffer_translation_rightB_03)), outlet_half_rightB_03));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightB_03_emitter_injection(outlet_rightB_03_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outlet_rightB_04_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_emitter_rotation_rightB_04), Vec3d(outlet_buffer_translation_rightB_04)), outlet_half_rightB_04));
    fluid_dynamics::BidirectionalBuffer<OutletInflowPressure> rightB_04_emitter_injection(outlet_rightB_04_emitter, in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_shell_contact);
    SimpleDynamics<fluid_dynamics::PressureConditionCylinder<InletInflowPressure>> inlet_pressure_condition(inlet_emitter_cylinder);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> main_pressure_condition(outlet_main_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> left_01_pressure_condition(outlet_left_01_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> left_02_pressure_condition(outlet_left_02_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> left_03_pressure_condition(outlet_left_03_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightF_01_pressure_condition(outlet_rightF_01_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightF_02_pressure_condition(outlet_rightF_02_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightB_01_pressure_condition(outlet_rightB_01_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightB_02_pressure_condition(outlet_rightB_02_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightB_03_pressure_condition(outlet_rightB_03_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<OutletInflowPressure>> rightB_04_pressure_condition(outlet_rightB_04_emitter);
    SimpleDynamics<fluid_dynamics::InflowVelocityConditionCylinder<InflowVelocity>> inflow_velocity_condition(inlet_emitter_cylinder);

    // FSI
    /*InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_on_shell(shell_water_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_on_shell(shell_water_contact);
    solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(shell_body);*/
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
    inlet_emitter_injection.tag_buffer_particles.exec();
    main_emitter_injection.tag_buffer_particles.exec();
    left_01_emitter_injection.tag_buffer_particles.exec();
    left_02_emitter_injection.tag_buffer_particles.exec();
    left_03_emitter_injection.tag_buffer_particles.exec();
    rightF_01_emitter_injection.tag_buffer_particles.exec();
    rightF_02_emitter_injection.tag_buffer_particles.exec();
    rightB_01_emitter_injection.tag_buffer_particles.exec();
    rightB_02_emitter_injection.tag_buffer_particles.exec();
    rightB_03_emitter_injection.tag_buffer_particles.exec();
    rightB_04_emitter_injection.tag_buffer_particles.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
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
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < Output_Time)
        {
            time_instance = TickCount::now();
            //apply_initial_force.exec();

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

                inlet_pressure_condition.exec(dt);
                main_pressure_condition.exec(dt);
                left_01_pressure_condition.exec(dt);
                left_02_pressure_condition.exec(dt);
                left_03_pressure_condition.exec(dt);
                rightF_01_pressure_condition.exec(dt);
                rightF_02_pressure_condition.exec(dt);
                rightB_01_pressure_condition.exec(dt);
                rightB_02_pressure_condition.exec(dt);
                rightB_03_pressure_condition.exec(dt);
                rightB_04_pressure_condition.exec(dt);

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

            inlet_emitter_injection.injection.exec();
            main_emitter_injection.injection.exec();
            left_01_emitter_injection.injection.exec();
            left_02_emitter_injection.injection.exec();
            left_03_emitter_injection.injection.exec();
            rightF_01_emitter_injection.injection.exec();
            rightF_02_emitter_injection.injection.exec();
            rightB_01_emitter_injection.injection.exec();
            rightB_02_emitter_injection.injection.exec();
            rightB_03_emitter_injection.injection.exec();
            rightB_04_emitter_injection.injection.exec();

            inlet_disposer_deletion.exec();
            main_disposer_injection.exec();
            left_01_disposer_injection.exec();
            left_02_disposer_injection.exec();
            left_03_disposer_injection.exec();
            rightF_01_disposer_injection.exec();
            rightF_02_disposer_injection.exec();
            rightB_01_disposer_injection.exec();
            rightB_02_disposer_injection.exec();
            rightB_03_disposer_injection.exec();
            rightB_04_disposer_injection.exec();

            water_block.updateCellLinkedListWithParticleSort(100);
            //shell_update_normal.exec();
            //shell_body.updateCellLinkedList();
            //shell_curvature_inner.updateConfiguration();
            //shell_average_curvature.exec();
            //shell_water_contact.updateConfiguration();
            water_block_complex.updateConfiguration();

            interval_updating_configuration += TickCount::now() - time_instance;
            boundary_indicator.exec();

            inlet_emitter_injection.tag_buffer_particles.exec();
            main_emitter_injection.tag_buffer_particles.exec();
            left_01_emitter_injection.tag_buffer_particles.exec();
            left_02_emitter_injection.tag_buffer_particles.exec();
            left_03_emitter_injection.tag_buffer_particles.exec();
            rightF_01_emitter_injection.tag_buffer_particles.exec();
            rightF_02_emitter_injection.tag_buffer_particles.exec();
            rightB_01_emitter_injection.tag_buffer_particles.exec();
            rightB_02_emitter_injection.tag_buffer_particles.exec();
            rightB_03_emitter_injection.tag_buffer_particles.exec();
            rightB_04_emitter_injection.tag_buffer_particles.exec();
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
