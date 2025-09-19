/**
 * @file 	diffusion_NeumannBC.cpp
 * @brief 	2D test of diffusion problem with Neumann boundary condition.
 * @details This is the first case to validate multiple boundary conditions.
 * @author 	Chenxi Zhao, Bo Zhang, Chi Zhang and Xiangyu Hu
 */
#include "diffusion_NeumannBC_testMultiPhase.h"

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
    SolidBody left_diffusion_body(sph_system, makeShared<LeftDiffusionBody>("LeftDiffusionBody"));
    left_diffusion_body.defineClosure<Solid, IsotropicThermalDiffusion>(
        Solid(), diffusion_species_name);
    left_diffusion_body.generateParticles<BaseParticles, Lattice>();

    SolidBody right_diffusion_body(sph_system, makeShared<RightDiffusionBody>("RightDiffusionBody"));
    right_diffusion_body.defineClosure<Solid, IsotropicThermalDiffusion>(
        Solid(), diffusion_species_name);
    right_diffusion_body.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_Dirichlet(sph_system, makeShared<DirichletWallBoundary>("DirichletWallBoundary"));
    wall_Dirichlet.defineMaterial<Solid>();
    wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_Neumann(sph_system, makeShared<NeumannWallBoundary>("NeumannWallBoundary"));
    wall_Neumann.defineMaterial<Solid>();
    wall_Neumann.generateParticles<BaseParticles, Lattice>();
    //----------------------------------------------------------------------
    //	Particle and body creation of temperature observers.
    //----------------------------------------------------------------------
    //ObserverBody temperature_observer(sph_system, "TemperatureObserver");
    //temperature_observer.generateParticles<ObserverParticles>(createObservationPoints());
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the range of bodies to build neighbor particle lists.
    //----------------------------------------------------------------------
    InnerRelation left_diffusion_body_inner(left_diffusion_body);
    ContactRelation left_diffusion_body_contact_Dirichlet(left_diffusion_body, { &wall_Dirichlet });
    ContactRelation left_diffusion_body_contact_Neumann(left_diffusion_body, {&wall_Neumann});
    ContactRelation left_right_contact(left_diffusion_body, {&right_diffusion_body});

    InnerRelation right_diffusion_body_inner(right_diffusion_body);
    ContactRelation right_diffusion_body_contact_Dirichlet(right_diffusion_body, {&wall_Dirichlet});
    ContactRelation right_diffusion_body_contact_Neumann(right_diffusion_body, {&wall_Neumann});
    ContactRelation right_left_contact(right_diffusion_body, {&left_diffusion_body});

    //ContactRelation temperature_observer_contact(temperature_observer, {&diffusion_body});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> left_diffusion_body_normal_direction(left_diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> right_diffusion_body_normal_direction(right_diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_Neumann);

    MultiPhaseDiffusionBodyRelaxation left_temperature_relaxation(
        left_diffusion_body_inner, left_right_contact, left_diffusion_body_contact_Dirichlet, left_diffusion_body_contact_Neumann);
    MultiPhaseDiffusionBodyRelaxation right_temperature_relaxation(
        right_diffusion_body_inner, right_left_contact, right_diffusion_body_contact_Dirichlet, right_diffusion_body_contact_Neumann);
    GetDiffusionTimeStepSize left_get_time_step_size(left_diffusion_body);
    GetDiffusionTimeStepSize right_get_time_step_size(right_diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> setup_left_diffusion_initial_condition(left_diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> setup_right_diffusion_initial_condition(right_diffusion_body);
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_boundary_condition_Dirichlet(wall_Dirichlet);
    SimpleDynamics<NeumannWallBoundaryInitialCondition> setup_boundary_condition_Neumann(wall_Neumann);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(sph_system);
    // ObservedQuantityRecording<Real> write_solid_temperature("Phi", temperature_observer_contact);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    setup_left_diffusion_initial_condition.exec();
    setup_right_diffusion_initial_condition.exec();
    setup_boundary_condition_Dirichlet.exec();
    setup_boundary_condition_Neumann.exec();
    left_diffusion_body_normal_direction.exec();
    right_diffusion_body_normal_direction.exec();
    wall_boundary_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int ite = 0;
    Real T0 = 1;
    Real End_Time = T0;
    Real Observe_time = 0.01 * End_Time;
    Real Output_Time = 0.1 * End_Time;
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
        while (integration_time < Output_Time)
        {
            Real relaxation_time = 0.0;
            while (relaxation_time < Observe_time)
            {
                if (ite % 500 == 0)
                {
                    std::cout << "N=" << ite << " Time: "
                              << physical_time << "	dt: "
                              << dt << "\n";
                }

                left_temperature_relaxation.exec(dt);
                right_temperature_relaxation.exec(dt);

                ite++;
                dt = SMIN(left_get_time_step_size.exec(), right_get_time_step_size.exec());
                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
            }
        }

        TickCount t2 = TickCount::now();
        write_states.writeToFile();
        /*write_solid_temperature.writeToFile(ite);*/
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();

    TickCount::interval_t tt;
    tt = t4 - t1 - interval;

    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
    std::cout << "Total physical time for computation: " << physical_time << " seconds." << std::endl;

    //if (sph_system.GenerateRegressionData())
    //{
    //    write_solid_temperature.generateDataBase(1.0e-3, 1.0e-3);
    //}
    //else if (sph_system.RestartStep() == 0)
    //{
    //    write_solid_temperature.testResult();
    //}

    return 0;
}