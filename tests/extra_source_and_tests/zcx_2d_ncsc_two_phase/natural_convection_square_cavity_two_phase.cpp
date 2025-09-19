/**
 * @file 	natural_convection_square_cavity_two_phase.cpp
 * @brief 	
 * @details 
 * @author 	
 */
#include "natural_convection_square_cavity_two_phase.h"

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
    FluidBody UpPhase_diffusion_body(sph_system, makeShared<UpPhaseDiffusionBody>("UpPhaseDiffusionBody"));
    UpPhase_diffusion_body.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicThermalDiffusion>(
        ConstructArgs(rho0_f, c_f), mu_f_up, ConstructArgs(diffusion_species_name, k, rho0_f, C_p));
    UpPhase_diffusion_body.generateParticles<BaseParticles, Lattice>();

    FluidBody DownPhase_diffusion_body(sph_system, makeShared<DownPhaseDiffusionBody>("DownPhaseDiffusionBody"));
    DownPhase_diffusion_body.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicThermalDiffusion>(
        ConstructArgs(rho0_f, c_f), mu_f_down, ConstructArgs(diffusion_species_name, k, rho0_f, C_p));
    DownPhase_diffusion_body.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("EntireWallBoundary"));
    wall_boundary.defineMaterial<Solid>();
    wall_boundary.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_Dirichlet(sph_system, makeShared<DirichletWallBoundary>("DirichletWallBoundary"));
    wall_Dirichlet.defineMaterial<Solid>();
    wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_Neumann(sph_system, makeShared<NeumannWallBoundary>("NeumannWallBoundary"));
    wall_Neumann.defineMaterial<Solid>();
    wall_Neumann.generateParticles<BaseParticles, Lattice>();

    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the range of bodies to build neighbor particle lists.
    //----------------------------------------------------------------------
    InnerRelation up_diffusion_inner(UpPhase_diffusion_body);
    InnerRelation down_diffusion_inner(DownPhase_diffusion_body);

    ContactRelation up_diffusion_contact_Dirichlet(UpPhase_diffusion_body, {&wall_Dirichlet});
    ContactRelation up_diffusion_contact_Neumann(UpPhase_diffusion_body, {&wall_Neumann});
    ContactRelation down_diffusion_contact_Dirichlet(DownPhase_diffusion_body, {&wall_Dirichlet});
    ContactRelation down_diffusion_contact_Neumann(DownPhase_diffusion_body, {&wall_Neumann});
    ContactRelation Dirichlet_contacts(wall_Dirichlet, {&UpPhase_diffusion_body, &DownPhase_diffusion_body});

    ContactRelation up_wall_contact(UpPhase_diffusion_body, {&wall_boundary});
    ContactRelation up_down_contact(UpPhase_diffusion_body, {&DownPhase_diffusion_body});
    ContactRelation up_contacts(UpPhase_diffusion_body, {&DownPhase_diffusion_body, &wall_boundary});

    ContactRelation down_wall_contact(DownPhase_diffusion_body, {&wall_boundary});
    ContactRelation down_up_contact(DownPhase_diffusion_body, {&UpPhase_diffusion_body});
    ContactRelation down_contacts(DownPhase_diffusion_body, {&UpPhase_diffusion_body, &wall_boundary});

    ComplexRelation up_complex(up_diffusion_inner, {&up_diffusion_contact_Dirichlet, &up_diffusion_contact_Neumann, &up_wall_contact, &up_down_contact, &up_contacts});
    ComplexRelation down_complex(down_diffusion_inner, {&down_diffusion_contact_Dirichlet, &down_diffusion_contact_Neumann, &down_wall_contact, &down_up_contact, &down_contacts});

    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> UpPhase_normal_direction(UpPhase_diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> DownPhase_normal_direction(DownPhase_diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> entire_wall_normal_direction(wall_boundary);
    SimpleDynamics<NormalDirectionFromBodyShape> Dirichlet_wall_normal_direction(wall_Dirichlet);
    SimpleDynamics<NormalDirectionFromBodyShape> Neumann_wall_normal_direction(wall_Neumann);

    MultiPhaseDiffusionBodyRelaxation up_temperature_relaxation(
        up_diffusion_inner, up_down_contact, up_diffusion_contact_Dirichlet, up_diffusion_contact_Neumann);
    GetDiffusionTimeStepSize up_thermal_time_step(UpPhase_diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> up_diffusion_initial_condition(UpPhase_diffusion_body);

    MultiPhaseDiffusionBodyRelaxation down_temperature_relaxation(
        down_diffusion_inner, down_up_contact, down_diffusion_contact_Dirichlet, down_diffusion_contact_Neumann);
    GetDiffusionTimeStepSize down_thermal_time_step(DownPhase_diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> down_diffusion_initial_condition(DownPhase_diffusion_body);

    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_boundary_condition_Dirichlet(wall_Dirichlet);
    SimpleDynamics<NeumannWallBoundaryInitialCondition> setup_boundary_condition_Neumann(wall_Neumann);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> up_kernel_correction_complex(InteractArgs(up_diffusion_inner, 0.1), up_contacts);
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> down_kernel_correction_complex(InteractArgs(down_diffusion_inner, 0.1), down_contacts);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfWithWallRiemann>
        up_pressure_relaxation(up_diffusion_inner, up_down_contact, up_wall_contact);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann>
        up_density_relaxation(up_diffusion_inner, up_down_contact, up_wall_contact);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfWithWallRiemann>
        down_pressure_relaxation(down_diffusion_inner, down_up_contact, down_wall_contact);
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann>
        down_density_relaxation(down_diffusion_inner, down_up_contact, down_wall_contact);

    InteractionWithUpdate<fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>>
        up_update_density_by_summation(up_diffusion_inner, up_down_contact, up_wall_contact);
    InteractionWithUpdate<fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>>
        down_update_density_by_summation(down_diffusion_inner, down_up_contact, down_wall_contact);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>>
        up_transport_correction(up_diffusion_inner, up_down_contact, up_wall_contact);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>>
        down_transport_correction(down_diffusion_inner, down_up_contact, down_wall_contact);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall> up_viscous_force(up_diffusion_inner, up_down_contact, up_wall_contact);
    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall> down_viscous_force(down_diffusion_inner, down_up_contact, down_wall_contact);

    SimpleDynamics<fluid_dynamics::BuoyancyForce> up_buoyancy_force(UpPhase_diffusion_body, thermal_expansion_coeff, initial_temperature);
    SimpleDynamics<fluid_dynamics::BuoyancyForce> down_buoyancy_force(DownPhase_diffusion_body, thermal_expansion_coeff, initial_temperature);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> up_advection_time_step(UpPhase_diffusion_body, U_f);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> down_advection_time_step(DownPhase_diffusion_body, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> up_acoustic_time_step(UpPhase_diffusion_body);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> down_acoustic_time_step(DownPhase_diffusion_body);

    
    InteractionWithUpdate<fluid_dynamics::TargetFluidParticles> up_target_fluid_particles(up_diffusion_contact_Dirichlet);
    InteractionWithUpdate<fluid_dynamics::TargetFluidParticles> down_target_fluid_particles(down_diffusion_contact_Dirichlet);
    SimpleDynamics<solid_dynamics::FirstLayerFromFluids> target_solid_particles(wall_Dirichlet, UpPhase_diffusion_body, DownPhase_diffusion_body);

    InteractionWithUpdate<fluid_dynamics::MultiPhasePhiGradientWithWall<LinearGradientCorrection>> calculate_up_phi_gradient(up_diffusion_inner, up_down_contact, up_diffusion_contact_Dirichlet);
    InteractionWithUpdate<fluid_dynamics::MultiPhasePhiGradientWithWall<LinearGradientCorrection>> calculate_down_phi_gradient(down_diffusion_inner, down_up_contact, down_diffusion_contact_Dirichlet);
    SimpleDynamics<fluid_dynamics::LocalNusseltNum> up_local_nusselt_number(UpPhase_diffusion_body, L/(left_temperature - right_temperature));
    SimpleDynamics<fluid_dynamics::LocalNusseltNum> down_local_nusselt_number(DownPhase_diffusion_body, L/(left_temperature - right_temperature));

    InteractionDynamics<solid_dynamics::ProjectionForNu> wall_local_nusselt_number(Dirichlet_contacts, L/(left_temperature - right_temperature));
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    ParticleSorting up_particle_sorting(UpPhase_diffusion_body);
    ParticleSorting down_particle_sorting(DownPhase_diffusion_body);
    BodyStatesRecordingToVtp write_states(sph_system);
    write_states.addToWrite<Real>(UpPhase_diffusion_body, "Pressure");
    write_states.addToWrite<Vecd>(UpPhase_diffusion_body, "BuoyancyForce");
    write_states.addToWrite<Vecd>(UpPhase_diffusion_body, "PhiGradient");
    write_states.addToWrite<Real>(UpPhase_diffusion_body, "LocalNusseltNumber");
    write_states.addToWrite<int>(UpPhase_diffusion_body, "FirstLayerIndicator");
    write_states.addToWrite<int>(UpPhase_diffusion_body, "SecondLayerIndicator");

    write_states.addToWrite<Real>(DownPhase_diffusion_body, "Pressure");
    write_states.addToWrite<Vecd>(DownPhase_diffusion_body, "BuoyancyForce");
    write_states.addToWrite<Vecd>(DownPhase_diffusion_body, "PhiGradient");
    write_states.addToWrite<Real>(DownPhase_diffusion_body, "LocalNusseltNumber");
    write_states.addToWrite<int>(DownPhase_diffusion_body, "FirstLayerIndicator");
    write_states.addToWrite<int>(DownPhase_diffusion_body, "SecondLayerIndicator");

    write_states.addToWrite<int>(wall_Dirichlet, "SolidFirstLayerIndicator");
    write_states.addToWrite<Real>(wall_Dirichlet, "WallLocalNusseltNumber");
    write_states.addToWrite<Vecd>(wall_boundary, "NormalDirection");
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    up_diffusion_initial_condition.exec();
    down_diffusion_initial_condition.exec();
    setup_boundary_condition_Dirichlet.exec();
    setup_boundary_condition_Neumann.exec();
    UpPhase_normal_direction.exec();
    DownPhase_normal_direction.exec();
    entire_wall_normal_direction.exec();
    Dirichlet_wall_normal_direction.exec();
    Neumann_wall_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    Real End_Time = 5; // Ra=10E4
    //Real End_Time = 5; // Ra=10E5
    //Real End_Time = 10.0; // Ra=10E6
    //Real End_Time = 10.0; // Ra=10E7
    //Real End_Time = 20.0; // Ra=10E8
    //Real End_Time = 550; // Ra=10E9
    Real output_interval = End_Time / 100.0; /**< time stamps for output,WriteToFile*/
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
            Real Dt_up = up_advection_time_step.exec();
            Real Dt_down = down_advection_time_step.exec();
            Real Dt = SMIN(Dt_up, Dt_down);

            up_update_density_by_summation.exec();
            down_update_density_by_summation.exec();
            up_kernel_correction_complex.exec();
            down_kernel_correction_complex.exec();
            up_viscous_force.exec();
            down_viscous_force.exec();
            up_transport_correction.exec();
            down_transport_correction.exec();

            size_t inner_ite_dt = 0;
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                Real dt_up = up_acoustic_time_step.exec();
                Real dt_down = down_acoustic_time_step.exec();

                Real thermal_up = up_thermal_time_step.exec();
                Real thermal_down = down_thermal_time_step.exec();

                Real dt = SMIN(SMIN(dt_up, dt_down), SMIN(thermal_up, thermal_down), Dt - relaxation_time);
                up_buoyancy_force.exec();
                down_buoyancy_force.exec();
                up_pressure_relaxation.exec(dt);
                down_pressure_relaxation.exec(dt);
                up_density_relaxation.exec(dt);
                down_density_relaxation.exec(dt);
                up_temperature_relaxation.exec(dt);
                down_temperature_relaxation.exec(dt);

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

            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                up_particle_sorting.exec();
                down_particle_sorting.exec();
            }

            UpPhase_diffusion_body.updateCellLinkedList();
            DownPhase_diffusion_body.updateCellLinkedList();
            up_complex.updateConfiguration();
            down_complex.updateConfiguration();
            Dirichlet_contacts.updateConfiguration();

            up_target_fluid_particles.exec();
            down_target_fluid_particles.exec();
            target_solid_particles.exec();
        }

        TickCount t2 = TickCount::now();

        calculate_up_phi_gradient.exec();
        calculate_down_phi_gradient.exec();
        up_local_nusselt_number.exec();
        down_local_nusselt_number.exec();
        wall_local_nusselt_number.exec();
        write_states.writeToFile();
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