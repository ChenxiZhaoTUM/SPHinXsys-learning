/**
 * @file 	NCCV_2phase_periodic.cpp
 * @brief 	
 * @details 
 * @author 	
 */
#include "NCCV_2phase_periodic.h"

using namespace SPH; // Namespace cite here
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    FluidBody PhaseOne_diffusion_body(sph_system, makeShared<PhaseOneDiffusionBody>("PhaseOneDiffusionBody"));
    PhaseOne_diffusion_body.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicThermalDiffusion>(
        ConstructArgs(rho0_f_one, c_f), mu_f_one, ConstructArgs(diffusion_species_name, k_one, rho0_f_one, C_p_one));
    PhaseOne_diffusion_body.generateParticles<BaseParticles, Lattice>();

    FluidBody PhaseTwo_diffusion_body(sph_system, makeShared<PhaseTwoDiffusionBody>("PhaseTwoDiffusionBody"));
    PhaseTwo_diffusion_body.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicThermalDiffusion>(
        ConstructArgs(rho0_f_two, c_f), mu_f_two, ConstructArgs(diffusion_species_name, k_two, rho0_f_two, C_p_two));
    PhaseTwo_diffusion_body.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("EntireWallBoundary"));
    wall_boundary.defineMaterial<Solid>();
    wall_boundary.generateParticles<BaseParticles, Lattice>();

    SolidBody up_Dirichlet(sph_system, makeShared<UpDirichlet>("UpDirichlet"));
    up_Dirichlet.defineMaterial<Solid>();
    up_Dirichlet.generateParticles<BaseParticles, Lattice>();

    SolidBody down_Dirichlet(sph_system, makeShared<DownDirichlet>("DownDirichlet"));
    down_Dirichlet.defineMaterial<Solid>();
    down_Dirichlet.generateParticles<BaseParticles, Lattice>();

    ObserverBody diffusion_observer(sph_system, "DiffusionObserver");
    diffusion_observer.generateParticles<ObserverParticles>(createObservationPoints());
    
    //----------------------------------------------------------------------
    //	Particle and body creation of temperature observers.
    //----------------------------------------------------------------------
    //ObserverBody nu_observer(sph_system, "NuObserver");
    //nu_observer.generateParticles<ObserverParticles>(createObservationPoints());
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the range of bodies to build neighbor particle lists.
    //----------------------------------------------------------------------
    InnerRelation phase_1_inner(PhaseOne_diffusion_body);
    InnerRelation phase_2_inner(PhaseTwo_diffusion_body);

    ContactRelation phase_1_contact_up_Dirichlet(PhaseOne_diffusion_body, {&up_Dirichlet});
    ContactRelation phase_1_contact_down_Dirichlet(PhaseOne_diffusion_body, {&down_Dirichlet});
    ContactRelation phase_1_contact_wall_boundary(PhaseOne_diffusion_body, {&wall_boundary});
    ContactRelation phase_1_contact_two(PhaseOne_diffusion_body, {&PhaseTwo_diffusion_body});
    ContactRelation phase_1_contacts(PhaseOne_diffusion_body, {&PhaseTwo_diffusion_body, &wall_boundary});

    ContactRelation phase_2_contact_up_Dirichlet(PhaseTwo_diffusion_body, {&up_Dirichlet});
    ContactRelation phase_2_contact_down_Dirichlet(PhaseTwo_diffusion_body, {&down_Dirichlet});
    ContactRelation phase_2_contact_wall_boundary(PhaseTwo_diffusion_body, {&wall_boundary});
    ContactRelation phase_2_contact_one(PhaseTwo_diffusion_body, {&PhaseOne_diffusion_body});
    ContactRelation phase_2_contacts(PhaseTwo_diffusion_body, {&PhaseOne_diffusion_body, &wall_boundary});

    ContactRelation up_Dirichlet_contacts(up_Dirichlet, {&PhaseOne_diffusion_body, &PhaseTwo_diffusion_body});
    ContactRelation down_Dirichlet_contacts(down_Dirichlet, {&PhaseOne_diffusion_body, &PhaseTwo_diffusion_body});

    ComplexRelation phase_1_complex(phase_1_inner, {&phase_1_contact_up_Dirichlet, &phase_1_contact_down_Dirichlet, 
        &phase_1_contact_wall_boundary, &phase_1_contact_two, &phase_1_contacts});
    ComplexRelation phase_2_complex(phase_2_inner, {&phase_2_contact_up_Dirichlet, &phase_2_contact_down_Dirichlet, 
        &phase_2_contact_wall_boundary, &phase_2_contact_one, &phase_2_contacts});

    ContactRelation observer_diffusion_body_contact(diffusion_observer, {&PhaseOne_diffusion_body, &PhaseTwo_diffusion_body});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> phase_1_normal_direction(PhaseOne_diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> phase_2_normal_direction(PhaseTwo_diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> entire_wall_normal_direction(wall_boundary);
    SimpleDynamics<NormalDirectionFromBodyShape> up_Dirichlet_normal_direction(up_Dirichlet);
    SimpleDynamics<NormalDirectionFromBodyShape> down_Dirichlet_normal_direction(down_Dirichlet);
    
    // thermal dynamics
    MultiPhaseDiffusionBodyRelaxation phase_1_temperature_relaxation(
        phase_1_inner, phase_1_contact_two, phase_1_contact_up_Dirichlet, phase_1_contact_down_Dirichlet);
    GetDiffusionTimeStepSize phase_1_thermal_time_step(PhaseOne_diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> phase_1_initial_condition(PhaseOne_diffusion_body);

    MultiPhaseDiffusionBodyRelaxation phase_2_temperature_relaxation(
        phase_2_inner, phase_2_contact_one, phase_2_contact_up_Dirichlet, phase_2_contact_down_Dirichlet);
    GetDiffusionTimeStepSize phase_2_thermal_time_step(PhaseTwo_diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> phase_2_initial_condition(PhaseTwo_diffusion_body);

    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_up_Dirichlet_initial_condition(up_Dirichlet);
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_down_Dirichlet_initial_condition(down_Dirichlet);

    // fluid dynamics
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> phase_1_kernel_correction_complex(InteractArgs(phase_1_inner, 0.5), phase_1_contacts);
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> phase_2_kernel_correction_complex(InteractArgs(phase_2_inner, 0.5), phase_2_contacts);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfCorrectionWithWallRiemann>
        phase_1_pressure_relaxation(phase_1_inner, phase_1_contact_two, phase_1_contact_wall_boundary);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann>
        phase_1_density_relaxation(phase_1_inner, phase_1_contact_two, phase_1_contact_wall_boundary);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfCorrectionWithWallRiemann>
        phase_2_pressure_relaxation(phase_2_inner, phase_2_contact_one, phase_2_contact_wall_boundary);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann>
        phase_2_density_relaxation(phase_2_inner, phase_2_contact_one, phase_2_contact_wall_boundary);

    InteractionWithUpdate<fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>>
        phase_1_update_density_by_summation(phase_1_inner, phase_1_contact_two, phase_1_contact_wall_boundary);
    InteractionWithUpdate<fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>>
        phase_2_update_density_by_summation(phase_2_inner, phase_2_contact_one, phase_2_contact_wall_boundary);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>>
        phase_1_transport_correction(phase_1_inner, phase_1_contact_two, phase_1_contact_wall_boundary);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>>
        phase_2_transport_correction(phase_2_inner, phase_2_contact_one, phase_2_contact_wall_boundary);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall> phase_1_viscous_force(phase_1_inner, phase_1_contact_two, phase_1_contact_wall_boundary);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall> phase_2_viscous_force(phase_2_inner, phase_2_contact_one, phase_2_contact_wall_boundary);

    // extract flux
    SimpleDynamics<fluid_dynamics::BuoyancyForce> phase_1_buoyancy_force(PhaseOne_diffusion_body, thermal_expansion_one, (up_temperature+down_temperature)/2.0);
    SimpleDynamics<fluid_dynamics::BuoyancyForce> phase_2_buoyancy_force(PhaseTwo_diffusion_body, thermal_expansion_two, (up_temperature+down_temperature)/2.0);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> phase_1_advection_time_step(PhaseOne_diffusion_body, U_f);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> phase_2_advection_time_step(PhaseTwo_diffusion_body, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> phase_1_acoustic_time_step(PhaseOne_diffusion_body);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> phase_2_acoustic_time_step(PhaseTwo_diffusion_body);

    BoundingBox bounding_box(Vec2d(0.0, - H/2), Vec2d(L, H/2));
    PeriodicAlongAxis phase_1_periodic_along_x(bounding_box, xAxis);
    PeriodicConditionUsingCellLinkedList phase_1_periodic_condition(PhaseOne_diffusion_body, phase_1_periodic_along_x);
    PeriodicAlongAxis phase_2_periodic_along_x(bounding_box, xAxis);
    PeriodicConditionUsingCellLinkedList phase_2_periodic_condition(PhaseTwo_diffusion_body, phase_2_periodic_along_x);

    InteractionWithUpdate<fluid_dynamics::TargetFluidParticles> phase_1_target_fluid_particles(phase_1_contact_wall_boundary);
    InteractionWithUpdate<fluid_dynamics::TargetFluidParticles> phase_2_target_fluid_particles(phase_2_contact_wall_boundary);
    SimpleDynamics<solid_dynamics::FirstLayerFromFluids> target_up_solid_particles(up_Dirichlet, PhaseOne_diffusion_body, PhaseTwo_diffusion_body);
    SimpleDynamics<solid_dynamics::FirstLayerFromFluids> target_down_solid_particles(down_Dirichlet, PhaseOne_diffusion_body, PhaseTwo_diffusion_body);
    
    InteractionWithUpdate<fluid_dynamics::MultiPhasePhiGradientWithWalls<LinearGradientCorrection>> calculate_phase_1_phi_gradient(phase_1_inner, 
        phase_1_contact_two, phase_1_contact_up_Dirichlet, phase_1_contact_down_Dirichlet);
    InteractionWithUpdate<fluid_dynamics::MultiPhasePhiGradientWithWalls<LinearGradientCorrection>> calculate_phase_2_phi_gradient(phase_2_inner, 
        phase_2_contact_one, phase_2_contact_up_Dirichlet, phase_2_contact_down_Dirichlet);
    SimpleDynamics<fluid_dynamics::LocalNusseltNum> phase_1_local_nusselt_number(PhaseOne_diffusion_body, H/(down_temperature - up_temperature));
    SimpleDynamics<fluid_dynamics::LocalNusseltNum> phase_2_local_nusselt_number(PhaseTwo_diffusion_body, H/(down_temperature - up_temperature));

    InteractionDynamics<solid_dynamics::ProjectionForNu> up_wall_local_nusselt_number(up_Dirichlet_contacts, H/(down_temperature - up_temperature));
    InteractionDynamics<solid_dynamics::ProjectionForNu> down_wall_local_nusselt_number(down_Dirichlet_contacts, H/(down_temperature - up_temperature));
    
    ReducedQuantityRecording<QuantitySummation<Real>> write_phase_1_PhiFluxSum(PhaseOne_diffusion_body, "PhiTransferFromDownDirichletFlux");
    ReducedQuantityRecording<QuantitySummation<Real>> write_phase_1_PhiFluxSumFromPhase2(PhaseOne_diffusion_body, "PhiTransferFromPhaseTwoDiffusionBodyFlux");
    ReducedQuantityRecording<QuantitySummation<Real>> write_phase_2_PhiFluxSum(PhaseTwo_diffusion_body, "PhiTransferFromDownDirichletFlux");

    BodyRegionByParticle left_diffusion_domain(PhaseOne_diffusion_body, makeShared<MultiPolygonShape>(createLeftDiffusionDomain(), "LeftDiffusionDomain"));
    ReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>> write_left_PhiFluxSum(left_diffusion_domain, "PhiTransferFromDownDirichletFlux");
    BodyRegionByParticle middle_diffusion_domain(PhaseOne_diffusion_body, makeShared<MultiPolygonShape>(createMiddleDiffusionDomain(), "MiddleDiffusionDomain"));
    ReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>> write_middle_PhiFluxSum(middle_diffusion_domain, "PhiTransferFromDownDirichletFlux");
    BodyRegionByParticle right_diffusion_domain(PhaseOne_diffusion_body, makeShared<MultiPolygonShape>(createRightDiffusionDomain(), "RightDiffusionDomain"));
    ReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>> write_right_PhiFluxSum(right_diffusion_domain, "PhiTransferFromDownDirichletFlux");

    ReducedQuantityRecording<solid_dynamics::AveragedWallNu<SPHBody>> write_global_average_Nu(down_Dirichlet, "WallLocalNusseltNumber");
    BodyRegionByParticle left_wall_domain(down_Dirichlet, makeShared<MultiPolygonShape>(createLeftDownWallDomain(), "LeftWallDomain"));
    ReducedQuantityRecording<solid_dynamics::AveragedWallNu<BodyRegionByParticle>> write_local_left_average_Nu(left_wall_domain, "WallLocalNusseltNumber");
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    ParticleSorting phase_1_particle_sorting(PhaseOne_diffusion_body);
    ParticleSorting phase_2_particle_sorting(PhaseTwo_diffusion_body);
    BodyStatesRecordingToVtp write_states(sph_system);
    write_states.addToWrite<Real>(PhaseOne_diffusion_body, "Pressure");
    write_states.addToWrite<Vecd>(PhaseOne_diffusion_body, "BuoyancyForce");
    write_states.addToWrite<Vecd>(PhaseOne_diffusion_body, "PhiGradient");
    write_states.addToWrite<Real>(PhaseOne_diffusion_body, "LocalNusseltNumber");
    write_states.addToWrite<int>(PhaseOne_diffusion_body, "FirstLayerIndicator");
    write_states.addToWrite<int>(PhaseOne_diffusion_body, "SecondLayerIndicator");

    write_states.addToWrite<Real>(PhaseTwo_diffusion_body, "Pressure");
    write_states.addToWrite<Vecd>(PhaseTwo_diffusion_body, "BuoyancyForce");
    write_states.addToWrite<Vecd>(PhaseTwo_diffusion_body, "PhiGradient");
    write_states.addToWrite<Real>(PhaseTwo_diffusion_body, "LocalNusseltNumber");
    write_states.addToWrite<int>(PhaseTwo_diffusion_body, "FirstLayerIndicator");
    write_states.addToWrite<int>(PhaseTwo_diffusion_body, "SecondLayerIndicator");

    write_states.addToWrite<int>(up_Dirichlet, "SolidFirstLayerIndicator");
    write_states.addToWrite<Real>(up_Dirichlet, "WallLocalNusseltNumber");
    write_states.addToWrite<int>(down_Dirichlet, "SolidFirstLayerIndicator");
    write_states.addToWrite<Real>(down_Dirichlet, "WallLocalNusseltNumber");
    
    write_states.addToWrite<Vecd>(wall_boundary, "NormalDirection");
    ObservedQuantityRecording<Vecd> write_recorded_fluid_vel("Velocity", observer_diffusion_body_contact);
    ReducedQuantityRecording<TotalKineticEnergy> write_phase_1_global_kinetic_energy(PhaseOne_diffusion_body);
    ReducedQuantityRecording<TotalKineticEnergy> write_phase_2_global_kinetic_energy(PhaseTwo_diffusion_body);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    phase_1_periodic_condition.update_cell_linked_list_.exec();
    phase_2_periodic_condition.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();
    phase_1_initial_condition.exec();
    phase_2_initial_condition.exec();
    setup_up_Dirichlet_initial_condition.exec();
    setup_down_Dirichlet_initial_condition.exec();
    phase_1_normal_direction.exec();
    phase_2_normal_direction.exec();
    entire_wall_normal_direction.exec();
    up_Dirichlet_normal_direction.exec();
    down_Dirichlet_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int ite = 0;
    Real End_Time = 600.0;
    Real output_interval = 1.0; /**< time stamps for output,WriteToFile*/
    int number_of_iterations = 0;
    int screen_output_interval = 100;
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TickCount::interval_t interval;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    write_states.writeToFile();
    //write_solid_temperature.writeToFile();
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (physical_time < End_Time)
    {
        Real integration_time = 0.0;
        while (integration_time < output_interval)
        {
            Real Dt_phase_1 = phase_1_advection_time_step.exec();
            Real Dt_phase_2 = phase_2_advection_time_step.exec();
            Real Dt = SMIN(Dt_phase_1, Dt_phase_2);

            phase_1_update_density_by_summation.exec();
            phase_2_update_density_by_summation.exec();
            phase_1_kernel_correction_complex.exec();
            phase_2_kernel_correction_complex.exec();
            phase_1_viscous_force.exec();
            phase_2_viscous_force.exec();
            phase_1_transport_correction.exec();
            phase_2_transport_correction.exec();

            size_t inner_ite_dt = 0;
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                Real dt_phase_1 = phase_1_acoustic_time_step.exec();
                Real dt_phase_2 = phase_2_acoustic_time_step.exec();

                Real thermal_phase_1 = phase_1_thermal_time_step.exec();
                Real thermal_phase_2 = phase_2_thermal_time_step.exec();

                Real dt = SMIN(SMIN(dt_phase_1, dt_phase_2), SMIN(thermal_phase_1, thermal_phase_2), Dt - relaxation_time);
                phase_1_buoyancy_force.exec();
                phase_2_buoyancy_force.exec();

                phase_1_pressure_relaxation.exec(dt);
                phase_2_pressure_relaxation.exec(dt);
                phase_1_density_relaxation.exec(dt);
                phase_2_density_relaxation.exec(dt);
                phase_1_temperature_relaxation.exec(dt);
                phase_2_temperature_relaxation.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
                inner_ite_dt++;  
            }

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << physical_time
                          << "	Dt = " << Dt << "	Dt / dt = " << inner_ite_dt << "\n";

                //write_states.writeToFile();
            }
            number_of_iterations++;

            phase_1_periodic_condition.bounding_.exec();
            phase_2_periodic_condition.bounding_.exec();

            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                phase_1_particle_sorting.exec();
                phase_2_particle_sorting.exec();
            }

            PhaseOne_diffusion_body.updateCellLinkedList();
            PhaseTwo_diffusion_body.updateCellLinkedList();
            phase_1_periodic_condition.update_cell_linked_list_.exec();
            phase_2_periodic_condition.update_cell_linked_list_.exec();
            phase_1_complex.updateConfiguration();
            phase_2_complex.updateConfiguration();
            up_Dirichlet_contacts.updateConfiguration();
            down_Dirichlet_contacts.updateConfiguration();

            phase_1_target_fluid_particles.exec();
            phase_2_target_fluid_particles.exec();
            target_up_solid_particles.exec();
            target_down_solid_particles.exec();
            observer_diffusion_body_contact.updateConfiguration();
        }

        TickCount t2 = TickCount::now();

        calculate_phase_1_phi_gradient.exec();
        calculate_phase_2_phi_gradient.exec();
        phase_1_local_nusselt_number.exec();
        phase_2_local_nusselt_number.exec();
        up_wall_local_nusselt_number.exec();
        down_wall_local_nusselt_number.exec();

        write_states.writeToFile();
        write_phase_1_PhiFluxSum.writeToFile(number_of_iterations);
        write_phase_1_PhiFluxSumFromPhase2.writeToFile(number_of_iterations);
        write_phase_2_PhiFluxSum.writeToFile(number_of_iterations);
        write_recorded_fluid_vel.writeToFile(number_of_iterations);
        write_phase_1_global_kinetic_energy.writeToFile(number_of_iterations);
        write_phase_2_global_kinetic_energy.writeToFile(number_of_iterations);

        write_left_PhiFluxSum.writeToFile(number_of_iterations);
        write_middle_PhiFluxSum.writeToFile(number_of_iterations);
        write_right_PhiFluxSum.writeToFile(number_of_iterations);

        write_global_average_Nu.writeToFile();
        write_local_left_average_Nu.writeToFile();

        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();
    TickCount::interval_t tt;
    tt = t4 - t1 - interval;

    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
    std::cout << "Total physical time for computation: " << physical_time << " seconds." << std::endl;

    return 0;
}