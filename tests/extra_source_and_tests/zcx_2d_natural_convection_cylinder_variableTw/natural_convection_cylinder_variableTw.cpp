/**
 * @file 	natural_convection_cylinder_variableTw.cpp
 * @brief 	
 * @details 
 * @author 	
 */
#include "natural_convection_cylinder_variableTw.h"

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
    /** Tag for run particle relaxation for the initial body fitted distribution. */
    sph_system.setRunParticleRelaxation(false);
    /** Tag for computation start with relaxed body fitted particles distribution. */
    sph_system.setReloadParticles(true);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    FluidBody diffusion_body(sph_system, makeShared<Cylinder>("DiffusionBody"));
    diffusion_body.defineBodyLevelSetShape();
    diffusion_body.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicDiffusion>(
        ConstructArgs(rho0_f, c_f), mu_f, ConstructArgs(diffusion_species_name, diffusion_coeff));
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? diffusion_body.generateParticles<BaseParticles, Reload>(diffusion_body.getName())
        : diffusion_body.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_Dirichlet(sph_system, makeShared<Wall>("DirichletWallBoundary"));
    wall_Dirichlet.defineBodyLevelSetShape();
    wall_Dirichlet.defineMaterial<Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_Dirichlet.generateParticles<BaseParticles, Reload>(wall_Dirichlet.getName())
        : wall_Dirichlet.generateParticles<BaseParticles, Lattice>();
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the range of bodies to build neighbor particle lists.
    //----------------------------------------------------------------------
    InnerRelation diffusion_body_inner(diffusion_body);
    ContactRelation Dirichlet_contact(wall_Dirichlet, {&diffusion_body});
    ContactRelation fluid_body_contact(diffusion_body, {&wall_Dirichlet});
    ComplexRelation fluid_body_complex(diffusion_body_inner, fluid_body_contact);

    if (sph_system.RunParticleRelaxation())
    {
        /** body topology only for particle relaxation */
        InnerRelation cylinder_inner(diffusion_body);
        InnerRelation wall_inner(wall_Dirichlet);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** Random reset the insert body particle position. */
        SimpleDynamics<RandomizeParticlePosition> random_cylinder_particles(diffusion_body);
        SimpleDynamics<RandomizeParticlePosition> random_wall_particles(wall_Dirichlet);
        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_bodies_to_vtp(sph_system);
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files({&diffusion_body, &wall_Dirichlet});
        /** A  Physics relaxation step. */
        RelaxationStepInner relaxation_step_inner(cylinder_inner);
        RelaxationStepInner relaxation_step_inner_wall(wall_inner);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_cylinder_particles.exec(0.25);
        random_wall_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_inner_wall.SurfaceBounding().exec();
        write_bodies_to_vtp.writeToFile(0);
        //----------------------------------------------------------------------
        //	Relax particles of the insert body.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 1000)
        {
            relaxation_step_inner.exec();
            relaxation_step_inner_wall.exec();
            ite_p += 1;
            if (ite_p % 200 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
                write_bodies_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of inserted body finish !" << std::endl;
        /** Output results. */
        write_particle_reload_files.writeToFile(0);
        return 0;
    }
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> diffusion_body_normal_direction(diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> Dirichlet_wall_normal_direction(wall_Dirichlet);

    DiffusionBodyRelaxation temperature_relaxation(
        diffusion_body_inner, fluid_body_contact);
    GetDiffusionTimeStepSize get_thermal_time_step(diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body);
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_boundary_condition_Dirichlet(wall_Dirichlet);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> kernel_correction_complex(InteractArgs(diffusion_body_inner, 0.1), fluid_body_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfCorrectionWithWallRiemann> pressure_relaxation(diffusion_body_inner, fluid_body_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(diffusion_body_inner, fluid_body_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplex> update_density_by_summation(diffusion_body_inner, fluid_body_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_force(diffusion_body_inner, fluid_body_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionCorrectedComplex<AllParticles>> transport_velocity_correction(diffusion_body_inner, fluid_body_contact);
    SimpleDynamics<fluid_dynamics::BuoyancyForce> buoyancy_force(diffusion_body, thermal_expansion_coeff, initial_temperature);
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step(diffusion_body, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step(diffusion_body);

    InteractionWithUpdate<fluid_dynamics::TargetFluidParticles> target_fluid_particles(fluid_body_contact);
    SimpleDynamics<solid_dynamics::FirstLayerFromFluid> target_solid_particles(wall_Dirichlet, diffusion_body);
    InteractionWithUpdate<fluid_dynamics::PhiGradientWithWall<LinearGradientCorrection>> calculate_phi_gradient(diffusion_body_inner, fluid_body_contact);
    SimpleDynamics<fluid_dynamics::LocalNusseltNum> local_nusselt_number(diffusion_body, inner_circle_radius/(initial_temperature - wall_temperature));
    InteractionDynamics<solid_dynamics::ProjectionForNu> wall_local_nusselt_number(Dirichlet_contact, inner_circle_radius/(initial_temperature - wall_temperature));
    SimpleDynamics<solid_dynamics::CalculateAveragedWallNu> calculate_averaged_wall_nu(wall_Dirichlet);
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
    write_states.addToWrite<int>(wall_Dirichlet, "SolidFirstLayerIndicator");
    write_states.addToWrite<Real>(wall_Dirichlet, "WallLocalNusseltNumber");
    
    write_states.addToWrite<Vecd>(wall_Dirichlet, "NormalDirection");
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    setup_diffusion_initial_condition.exec();
    setup_boundary_condition_Dirichlet.exec();
    diffusion_body_normal_direction.exec();
    Dirichlet_wall_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int ite = 0;
    Real End_Time = 2500;
    Real output_interval = End_Time / 250.0; /**< time stamps for output,WriteToFile*/
    int number_of_iterations = 0;
    int screen_output_interval = 100;
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
                setup_boundary_condition_Dirichlet.exec();

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
            fluid_body_complex.updateConfiguration();
            Dirichlet_contact.updateConfiguration();

            target_fluid_particles.exec();
            target_solid_particles.exec();
        }

        TickCount t2 = TickCount::now();

        calculate_phi_gradient.exec();
        local_nusselt_number.exec();
        wall_local_nusselt_number.exec();
        write_states.writeToFile();
        calculate_averaged_wall_nu.exec();
        calculate_averaged_wall_nu.writeAveragedWallNu();
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