/**
 * @file 	diffusion_NeumannBC.cpp
 * @brief 	2D test of diffusion problem with Neumann boundary condition.
 * @details This is the first case to validate multiple boundary conditions.
 * @author 	Chenxi Zhao, Bo Zhang, Chi Zhang and Xiangyu Hu
 */
#include "diffusion_TestForPhiGrad.h"

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
    SolidBody diffusion_body(sph_system, makeShared<DiffusionBody>("DiffusionBody"));
    diffusion_body.defineClosure<Solid, IsotropicDiffusion>(
        Solid(), ConstructArgs(diffusion_species_name, diffusion_coeff));
    diffusion_body.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_Dirichlet(sph_system, makeShared<DirichletWallBoundary>("DirichletWallBoundary"));
    wall_Dirichlet.defineMaterial<Solid>();
    wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

    //----------------------------------------------------------------------
    //	Particle and body creation of temperature observers.
    //----------------------------------------------------------------------
    ObserverBody temperature_observer(sph_system, "TemperatureObserver");
    temperature_observer.generateParticles<ObserverParticles>(createObservationPoints());
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the range of bodies to build neighbor particle lists.
    //----------------------------------------------------------------------
    InnerRelation diffusion_body_inner(diffusion_body);
    ContactRelation diffusion_body_contact_Dirichlet(diffusion_body, {&wall_Dirichlet});
    ContactRelation temperature_observer_contact(temperature_observer, {&diffusion_body});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> diffusion_body_normal_direction(diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_Dirichlet);

    ThermalRelaxationComplex temperature_relaxation(
        diffusion_body_inner, diffusion_body_contact_Dirichlet);
    GetDiffusionTimeStepSize get_time_step_size(diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body);
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_boundary_condition_Dirichlet(wall_Dirichlet);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> kernel_correction_complex(DynamicsArgs(diffusion_body_inner, 0.1), diffusion_body_contact_Dirichlet);
    InteractionDynamics<fluid_dynamics::DistanceFromWall> distance_to_wall(diffusion_body_contact_Dirichlet);
    InteractionWithUpdate<fluid_dynamics::PhiGradientWithWall<LinearGradientCorrection>> calculate_phi_gradient(diffusion_body_inner, diffusion_body_contact_Dirichlet);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(sph_system);
    //write_states.addToWrite<Vecd>(diffusion_body, "LinearGradientCorrectionMatrix");
    write_states.addToWrite<Vecd>(diffusion_body, "PhiGradient");
    ObservedQuantityRecording<Real> write_solid_temperature("Phi", temperature_observer_contact);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    setup_diffusion_initial_condition.exec();
    setup_boundary_condition_Dirichlet.exec();
    diffusion_body_normal_direction.exec();
    wall_boundary_normal_direction.exec();

    kernel_correction_complex.exec();
    distance_to_wall.exec();
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
    write_solid_temperature.writeToFile();
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

                temperature_relaxation.exec(dt);
                
                ite++;
                dt = get_time_step_size.exec();
                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
            }
        }

        TickCount t2 = TickCount::now();
        calculate_phi_gradient.exec();
        write_states.writeToFile();
        write_solid_temperature.writeToFile(ite);
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