/**
 * @file 	natural_convection_square_cavity.cpp
 * @brief 	
 * @details 
 * @author 	
 */
#include "natural_convection_square_cavity.h"

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
    sph_system.handleCommandlineOptions(ac, av);
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

    SolidBody wall_Dirichlet(sph_system, makeShared<DirichletWallBoundary>("DirichletWallBoundary"));
    wall_Dirichlet.defineMaterial<Solid>();
    wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_Neumann(sph_system, makeShared<NeumannWallBoundary>("NeumannWallBoundary"));
    wall_Neumann.defineMaterial<Solid>();
    wall_Neumann.generateParticles<BaseParticles, Lattice>();
    //----------------------------------------------------------------------
    //	Particle and body creation of temperature observers.
    //----------------------------------------------------------------------
    /*ObserverBody temperature_observer(sph_system, "TemperatureObserver");
    temperature_observer.generateParticles<ObserverParticles>(createObservationPoints());*/
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the range of bodies to build neighbor particle lists.
    //----------------------------------------------------------------------
    InnerRelation diffusion_body_inner(diffusion_body);
    ContactRelation diffusion_body_contact_Dirichlet(diffusion_body, {&wall_Dirichlet});
    ContactRelation diffusion_body_contact_Neumann(diffusion_body, {&wall_Neumann});

    ContactRelation fluid_body_contact(diffusion_body, {&wall_boundary});
    ComplexRelation fluid_body_complex(diffusion_body_inner, fluid_body_contact);
    //ContactRelation temperature_observer_contact(temperature_observer, {&diffusion_body});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> diffusion_body_normal_direction(diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> entire_wall_normal_direction(wall_boundary);
    SimpleDynamics<NormalDirectionFromBodyShape> Dirichlet_wall_normal_direction(wall_Dirichlet);
    SimpleDynamics<NormalDirectionFromBodyShape> Neumann_wall_normal_direction(wall_Neumann);

    DiffusionBodyRelaxation temperature_relaxation(
        diffusion_body_inner, diffusion_body_contact_Dirichlet, diffusion_body_contact_Neumann);
    GetDiffusionTimeStepSize get_thermal_time_step(diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body);
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_boundary_condition_Dirichlet(wall_Dirichlet);
    SimpleDynamics<NeumannWallBoundaryInitialCondition> setup_boundary_condition_Neumann(wall_Neumann);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> kernel_correction_complex(DynamicsArgs(diffusion_body_inner, 0.1), fluid_body_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfCorrectionWithWallRiemann> pressure_relaxation(diffusion_body_inner, fluid_body_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(diffusion_body_inner, fluid_body_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplex> update_density_by_summation(diffusion_body_inner, fluid_body_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_force(diffusion_body_inner, fluid_body_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionCorrectedComplex<AllParticles>> transport_velocity_correction(diffusion_body_inner, fluid_body_contact);
    SimpleDynamics<fluid_dynamics::BuoyancyForce> buoyancy_force(diffusion_body, thermal_expansion_coeff, initial_temperature);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step(diffusion_body, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step(diffusion_body);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    ParticleSorting particle_sorting(diffusion_body);
    BodyStatesRecordingToVtp write_states(sph_system);
    write_states.addToWrite<Real>(diffusion_body, "Pressure");
    write_states.addToWrite<Vecd>(diffusion_body, "BuoyancyForce");
    write_states.addToWrite<Vecd>(diffusion_body, "ViscousForce");
    write_states.addToWrite<Vecd>(diffusion_body, "ForcePrior");
    write_states.addToWrite<Vecd>(diffusion_body, "Force");
    write_states.addToWrite<Vecd>(wall_boundary, "NormalDirection");
    write_states.addToWrite<Vecd>(wall_Dirichlet, "NormalDirection");
    write_states.addToWrite<Vecd>(wall_Neumann, "NormalDirection");
    //ObservedQuantityRecording<Real> write_solid_temperature(diffusion_species_name, temperature_observer_contact);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    setup_diffusion_initial_condition.exec();
    setup_boundary_condition_Dirichlet.exec();
    setup_boundary_condition_Neumann.exec();
    diffusion_body_normal_direction.exec();
    entire_wall_normal_direction.exec();
    Dirichlet_wall_normal_direction.exec();
    Neumann_wall_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int ite = 0;
    Real End_Time = 5;
    Real output_interval = End_Time / 100.0; /**< time stamps for output,WriteToFile*/
    int number_of_iterations = 0;
    int screen_output_interval = 50;
    Real dt = 0.0;
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

            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                particle_sorting.exec();
            }
            diffusion_body.updateCellLinkedList();
            diffusion_body_contact_Dirichlet.updateConfiguration();
            diffusion_body_contact_Neumann.updateConfiguration();
            fluid_body_complex.updateConfiguration();
        }

        TickCount t2 = TickCount::now();
        write_states.writeToFile();
        //write_solid_temperature.writeToFile(ite);
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