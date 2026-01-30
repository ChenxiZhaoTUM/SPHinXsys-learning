/**
 * @file 	natural_convection_CV_periodic.cpp
 * @brief 	
 * @details 
 * @author 	
 */
#include "natural_convection_CV_periodic.h"

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
    FluidBody diffusion_body(sph_system, makeShared<DiffusionBody>("DiffusionBody"));
    diffusion_body.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicDiffusion>(
        ConstructArgs(rho0_f, c_f), mu_f, ConstructArgs(diffusion_species_name, diffusion_coeff));
    diffusion_body.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("EntireWallBoundary"));
    wall_boundary.defineMaterial<Solid>();
    wall_boundary.generateParticles<BaseParticles, Lattice>();

    SolidBody up_wall_Dirichlet(sph_system, makeShared<UpDirichletWallBoundary>("UpDirichletWallBoundary"));
    up_wall_Dirichlet.defineMaterial<Solid>();
    up_wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

    SolidBody down_wall_Dirichlet(sph_system, makeShared<DownDirichletWallBoundary>("DownDirichletWallBoundary"));
    down_wall_Dirichlet.defineMaterial<Solid>();
    down_wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

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
    InnerRelation diffusion_body_inner(diffusion_body);
    //InnerRelation wall_inner(wall_Dirichlet);
    ContactRelation up_Dirichlet_contact(up_wall_Dirichlet, {&diffusion_body});
    ContactRelation down_Dirichlet_contact(down_wall_Dirichlet, {&diffusion_body});
    ContactRelation diffusion_body_contact_all_Dirichlet(diffusion_body, {&up_wall_Dirichlet, &down_wall_Dirichlet});
    ContactRelation diffusion_body_contact_up_Dirichlet(diffusion_body, {&up_wall_Dirichlet});
    ContactRelation diffusion_body_contact_down_Dirichlet(diffusion_body, {&down_wall_Dirichlet});

    ContactRelation fluid_body_contact(diffusion_body, {&wall_boundary});
    ComplexRelation fluid_body_complex(diffusion_body_inner, fluid_body_contact);
    //ContactRelation nusselt_observer_contact(nu_observer, {&diffusion_body});
    ContactRelation observer_diffusion_body_contact(diffusion_observer, {&diffusion_body});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> diffusion_body_normal_direction(diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> entire_wall_normal_direction(wall_boundary);
    SimpleDynamics<NormalDirectionFromBodyShape> up_Dirichlet_wall_normal_direction(up_wall_Dirichlet);
    SimpleDynamics<NormalDirectionFromBodyShape> down_Dirichlet_wall_normal_direction(down_wall_Dirichlet);

    DiffusionBodyRelaxation temperature_relaxation(
        diffusion_body_inner, diffusion_body_contact_up_Dirichlet, diffusion_body_contact_down_Dirichlet);
    GetDiffusionTimeStepSize get_thermal_time_step(diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body);
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_up_Dirichlet_initial_condition(up_wall_Dirichlet);
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_down_Dirichlet_initial_condition(down_wall_Dirichlet);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> kernel_correction_complex(InteractArgs(diffusion_body_inner, 0.1), fluid_body_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfCorrectionWithWallRiemann> pressure_relaxation(diffusion_body_inner, fluid_body_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(diffusion_body_inner, fluid_body_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplex> update_density_by_summation(diffusion_body_inner, fluid_body_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_force(diffusion_body_inner, fluid_body_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionCorrectedComplex<AllParticles>> transport_velocity_correction(diffusion_body_inner, fluid_body_contact);
    SimpleDynamics<fluid_dynamics::BuoyancyForce> buoyancy_force(diffusion_body, thermal_expansion_coeff, (up_temperature+down_temperature)/2.0);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step(diffusion_body, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step(diffusion_body);
    PeriodicAlongAxis periodic_along_x(diffusion_body.getSPHBodyBounds(), xAxis);
    PeriodicConditionUsingCellLinkedList periodic_condition(diffusion_body, periodic_along_x);

    InteractionWithUpdate<fluid_dynamics::TargetFluidParticles> target_fluid_particles(diffusion_body_contact_all_Dirichlet);
    SimpleDynamics<solid_dynamics::FirstLayerFromFluid> target_up_solid_particles(up_wall_Dirichlet, diffusion_body);
    SimpleDynamics<solid_dynamics::FirstLayerFromFluid> target_down_solid_particles(down_wall_Dirichlet, diffusion_body);
    InteractionWithUpdate<fluid_dynamics::PhiGradientWithWall<LinearGradientCorrection>> calculate_phi_gradient(diffusion_body_inner, diffusion_body_contact_all_Dirichlet);
    SimpleDynamics<fluid_dynamics::LocalNusseltNum> local_nusselt_number(diffusion_body, H / (down_temperature - up_temperature));
    InteractionDynamics<solid_dynamics::ProjectionForNu> up_wall_local_nusselt_number(up_Dirichlet_contact, H / (down_temperature - up_temperature));
    InteractionDynamics<solid_dynamics::ProjectionForNu> down_wall_local_nusselt_number(down_Dirichlet_contact, H / (down_temperature - up_temperature));
    //SimpleDynamics<solid_dynamics::CalculateAveragedWallNu> calculate_averaged_wall_nu(wall_Dirichlet);
    
    ReducedQuantityRecording<QuantitySummation<Real>> write_up_PhiFluxSum(diffusion_body, "PhiTransferFromUpDirichletWallBoundaryFlux");
    ReducedQuantityRecording<QuantitySummation<Real>> write_down_PhiFluxSum(diffusion_body, "PhiTransferFromDownDirichletWallBoundaryFlux");
    BodyRegionByParticle left_diffusion_domain(diffusion_body, makeShared<MultiPolygonShape>(createLeftDiffusionDomain(), "LeftDiffusionDomain"));
    ReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>> write_left_PhiFluxSum(left_diffusion_domain, "PhiTransferFromDownDirichletWallBoundaryFlux");
    BodyRegionByParticle middle_diffusion_domain(diffusion_body, makeShared<MultiPolygonShape>(createMiddleDiffusionDomain(), "MiddleDiffusionDomain"));
    ReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>> write_middle_PhiFluxSum(middle_diffusion_domain, "PhiTransferFromDownDirichletWallBoundaryFlux");
    BodyRegionByParticle right_diffusion_domain(diffusion_body, makeShared<MultiPolygonShape>(createRightDiffusionDomain(), "RightDiffusionDomain"));
    ReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>> write_right_PhiFluxSum(right_diffusion_domain, "PhiTransferFromDownDirichletWallBoundaryFlux");

    //BodyRegionByCell left_diffusion_domain(diffusion_body, makeShared<MultiPolygonShape>(createLeftDiffusionDomain(), "LeftDiffusionDomain"));
    //SimpleDynamics<fluid_dynamics::FluidLocalVerticalHeatFlux> left_diffusion_domain_flux(left_diffusion_domain, "Left", H / (down_temperature - up_temperature) / kappa, kappa);
    //BodyRegionByCell middle_diffusion_domain(diffusion_body, makeShared<MultiPolygonShape>(createMiddleDiffusionDomain(), "MiddleDiffusionDomain"));
    //SimpleDynamics<fluid_dynamics::FluidLocalVerticalHeatFlux> middle_diffusion_domain_flux(middle_diffusion_domain, "Middle", H / (down_temperature - up_temperature) / kappa, kappa);
    //BodyRegionByCell right_diffusion_domain(diffusion_body, makeShared<MultiPolygonShape>(createRightDiffusionDomain(), "RightDiffusionDomain"));
    //SimpleDynamics<fluid_dynamics::FluidLocalVerticalHeatFlux> right_diffusion_domain_flux(right_diffusion_domain, "Right", H / (down_temperature - up_temperature) / kappa, kappa);
    //ReducedQuantityRecording<Average<QuantitySummation<Real, BodyPartByCell>>>
    //write_left_averaged_LocalVerticalHeatFlux(left_diffusion_domain, "LeftFluidLocalVerticalHeatFlux");
    //ReducedQuantityRecording<Average<QuantitySummation<Real, BodyPartByCell>>>
    //write_middle_averaged_LocalVerticalHeatFlux(middle_diffusion_domain, "MiddleFluidLocalVerticalHeatFlux");
    //ReducedQuantityRecording<Average<QuantitySummation<Real, BodyPartByCell>>>
    //write_right_averaged_LocalVerticalHeatFlux(right_diffusion_domain, "RightFluidLocalVerticalHeatFlux");
 
    ReducedQuantityRecording<solid_dynamics::AveragedWallNu<SPHBody>> write_global_average_Nu(down_wall_Dirichlet, "WallLocalNusseltNumber");
    BodyRegionByParticle left_wall_domain(down_wall_Dirichlet, makeShared<MultiPolygonShape>(createLeftDownWallDomain(), "LeftWallDomain"));
    ReducedQuantityRecording<solid_dynamics::AveragedWallNu<BodyRegionByParticle>> write_local_left_average_Nu(left_wall_domain, "WallLocalNusseltNumber");
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    ParticleSorting particle_sorting(diffusion_body);
    BodyStatesRecordingToVtp write_states(sph_system);
    write_states.addToWrite<Real>(diffusion_body, "Pressure");
    write_states.addToWrite<Vecd>(diffusion_body, "BuoyancyForce");

    write_states.addToWrite<Vecd>(diffusion_body, "PhiGradient");
    write_states.addToWrite<Real>(diffusion_body, "LocalNusseltNumber");

    write_states.addToWrite<int>(diffusion_body, "FirstLayerIndicator");
    write_states.addToWrite<int>(diffusion_body, "SecondLayerIndicator");
    write_states.addToWrite<int>(up_wall_Dirichlet, "SolidFirstLayerIndicator");
    write_states.addToWrite<Real>(up_wall_Dirichlet, "WallLocalNusseltNumber");
    write_states.addToWrite<int>(down_wall_Dirichlet, "SolidFirstLayerIndicator");
    write_states.addToWrite<Real>(down_wall_Dirichlet, "WallLocalNusseltNumber");
    
    write_states.addToWrite<Vecd>(wall_boundary, "NormalDirection");
    ObservedQuantityRecording<Vecd> write_recorded_fluid_vel("Velocity", observer_diffusion_body_contact);
    ReducedQuantityRecording<TotalKineticEnergy> write_global_kinetic_energy(diffusion_body);
    /*ObservedQuantityRecording<Real> write_local_nusselt_number("LocalNusseltNumber", nusselt_observer_contact);*/
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    periodic_condition.update_cell_linked_list_.exec();
    sph_system.initializeSystemConfigurations();
    setup_diffusion_initial_condition.exec();
    setup_up_Dirichlet_initial_condition.exec();
    setup_down_Dirichlet_initial_condition.exec();
    diffusion_body_normal_direction.exec();
    entire_wall_normal_direction.exec();
    up_Dirichlet_wall_normal_direction.exec();
    down_Dirichlet_wall_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int ite = 0;
    Real End_Time = 400.0;
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
            Real Dt = get_fluid_advection_time_step.exec();
            update_density_by_summation.exec();
            kernel_correction_complex.exec();
            viscous_force.exec();
            transport_velocity_correction.exec();

            size_t inner_ite_dt = 0;
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                Real dt = SMIN(SMIN(get_thermal_time_step.exec(), get_fluid_time_step.exec()), Dt - relaxation_time);
                buoyancy_force.exec();
                pressure_relaxation.exec(dt);
                density_relaxation.exec(dt);
                temperature_relaxation.exec(dt);

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

            periodic_condition.bounding_.exec();
            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                particle_sorting.exec();
            }
            diffusion_body.updateCellLinkedList();
            periodic_condition.update_cell_linked_list_.exec();
            diffusion_body_contact_all_Dirichlet.updateConfiguration();
            diffusion_body_contact_up_Dirichlet.updateConfiguration();
            diffusion_body_contact_down_Dirichlet.updateConfiguration();
            fluid_body_complex.updateConfiguration();
            up_Dirichlet_contact.updateConfiguration();
            down_Dirichlet_contact.updateConfiguration();
            observer_diffusion_body_contact.updateConfiguration();

            target_fluid_particles.exec();
            target_up_solid_particles.exec();
            target_down_solid_particles.exec();
        }

        TickCount t2 = TickCount::now();

        calculate_phi_gradient.exec();
        local_nusselt_number.exec();
        up_wall_local_nusselt_number.exec();
        down_wall_local_nusselt_number.exec();

        write_states.writeToFile();
        write_up_PhiFluxSum.writeToFile(number_of_iterations);
        write_down_PhiFluxSum.writeToFile(number_of_iterations);
        write_recorded_fluid_vel.writeToFile(number_of_iterations);
        write_global_kinetic_energy.writeToFile(number_of_iterations);

        write_left_PhiFluxSum.writeToFile(number_of_iterations);
        write_middle_PhiFluxSum.writeToFile(number_of_iterations);
        write_right_PhiFluxSum.writeToFile(number_of_iterations);

        write_global_average_Nu.writeToFile();
        write_local_left_average_Nu.writeToFile();

        //left_diffusion_domain_flux.exec();
        //write_left_averaged_LocalVerticalHeatFlux.writeToFile(number_of_iterations);
        //middle_diffusion_domain_flux.exec();
        //write_middle_averaged_LocalVerticalHeatFlux.writeToFile(number_of_iterations);
        //right_diffusion_domain_flux.exec();
        //write_right_averaged_LocalVerticalHeatFlux.writeToFile(number_of_iterations);

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