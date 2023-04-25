/**
 * @file 	2d_diffusion_test_with_NeumannBC.cpp
 * @brief 	2D diffusion test of diffusion problem with Neumann boundary condition.
 * @details This is a case to implement Neumann boundary condition.
 * @author 	Chenxi Zhao, Bo Zhang, Chi Zhang and Xiangyu Hu
 */
#include "sphinxsys.h"
#include "2d_diffusion_test_with_NeumannBC.h"

using namespace SPH; //Namespace cite here

//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char* av[])
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem.
	//----------------------------------------------------------------------
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	IOEnvironment io_environment(sph_system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	//---------------------------------------------------------------------- 
	SolidBody diffusion_body(sph_system, makeShared<DiffusionBody>("DiffusionBody"));
	diffusion_body.defineParticlesAndMaterial<DiffusionParticlesWithBoundary, DiffusionMaterial>();
	diffusion_body.generateParticles<ParticleGeneratorLattice>();

	SolidBody wall_boundary_Dirichlet(sph_system, makeShared<WallBoundaryDirichlet>("WallBoundaryDirichlet"));
	wall_boundary_Dirichlet.defineParticlesAndMaterial<WallParticles, DiffusionMaterial>();
	wall_boundary_Dirichlet.generateParticles<ParticleGeneratorLattice>();

	SolidBody wall_boundary_Neumann(sph_system, makeShared<WallBoundaryNeumann>("WallBoundaryNeumann"));
	wall_boundary_Neumann.defineParticlesAndMaterial<WallParticles, DiffusionMaterial>();
	wall_boundary_Neumann.generateParticles<ParticleGeneratorLattice>();

	//----------------------------------------------------------------------
	//	Particle and body creation of temperature observers.
	//----------------------------------------------------------------------
	ObserverBody temperature_observer(sph_system, "TemperatureObserver");
	temperature_observer.generateParticles<TemperatureObserverParticleGenerator>();
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	InnerRelation diffusion_body_inner_relation(diffusion_body);

	ContactRelation diffusion_body_contact_Dirichlet(diffusion_body, { &wall_boundary_Dirichlet });
	ContactRelation diffusion_body_contact_Neumann(diffusion_body, { &wall_boundary_Neumann });
	
	ComplexRelation diffusion_body_complex_Dirichlet(diffusion_body, { &wall_boundary_Dirichlet });
	ComplexRelation wall_boundary_complex_Dirichlet(wall_boundary_Dirichlet, { &diffusion_body });

	ComplexRelation diffusion_body_complex_Neumann(diffusion_body, { &wall_boundary_Neumann });
	ComplexRelation wall_boundary_complex_Neumann(wall_boundary_Neumann, { &diffusion_body });

	ContactRelation temperature_observer_contact(temperature_observer, { &diffusion_body });
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body);
	SimpleDynamics<WallBoundaryInitialCondition> setup_boundary_condition_Dirichlet(wall_boundary_Dirichlet);
	SimpleDynamics<WallBoundaryInitialCondition> setup_boundary_condition_Neumann(wall_boundary_Neumann);
	GetDiffusionTimeStepSize<DiffusionParticlesWithBoundary> get_time_step_size(diffusion_body);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	ObservedQuantityRecording<Real> write_solid_temperature("Phi", io_environment, temperature_observer_contact);
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	//DiffusionRelaxationWithDirichlet diffusion_with_dirichlet_boundary(diffusion_body_contact_Dirichlet);

	//DiffusionReactionContactData<DiffusionParticlesWithBoundary, DiffusionParticlesWithBoundary> contact_data(diffusion_body_contact_Dirichlet); //test

	//DiffusionRelaxationComplex complex(diffusion_body_complex_Dirichlet);
	DiffusionRelaxationSimpleContact simple(diffusion_body_contact_Dirichlet); //test

	//DiffusionRelaxationWithNeumann diffusion_with_neumann_boundary(diffusion_body_contact_Neumann);
	//DiffusionBodyRelaxation temperature_relaxation(diffusion_body_inner_relation, diffusion_body_contact_Dirichlet);

	//InteractionDynamics<UpdateUnitVectorNormalToBoundary<DiffusionParticlesWithBoundary, WallParticles>> update_diffusion_body_normal_vector_Dirichlet(diffusion_body_complex_Dirichlet);
	//InteractionDynamics<UpdateUnitVectorNormalToBoundary<DiffusionParticlesWithBoundary, WallParticles>> update_wall_boundary_normal_vector_Dirichlet(wall_boundary_complex_Dirichlet);

	InteractionDynamics<UpdateUnitVectorNormalToBoundary<DiffusionParticlesWithBoundary, WallParticles>> update_diffusion_body_normal_vector_Neumann(diffusion_body_complex_Neumann);
	InteractionDynamics<UpdateUnitVectorNormalToBoundary<DiffusionParticlesWithBoundary, WallParticles>> update_wall_boundary_normal_vector_Neumann(wall_boundary_complex_Neumann);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary. 
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();

	setup_diffusion_initial_condition.exec();

	setup_boundary_condition_Dirichlet.exec();
	setup_boundary_condition_Neumann.exec();

	//update_diffusion_body_normal_vector_Dirichlet.exec();
	//update_wall_boundary_normal_vector_Dirichlet.exec();

	update_diffusion_body_normal_vector_Neumann.exec();
	update_wall_boundary_normal_vector_Neumann.exec();

	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
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
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (GlobalStaticVariables::physical_time_ < End_Time)
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
						<< GlobalStaticVariables::physical_time_ << "	dt: "
						<< dt << "\n";
				}

				//temperature_relaxation.exec(dt);

				ite++;
				dt = get_time_step_size.exec();
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
			}
		}

		TickCount t2 = TickCount::now();
		write_states.writeToFile();
		write_solid_temperature.writeToFile(ite);
		TickCount t3 = TickCount::now();
		interval += t3 - t2;
	}
	TickCount t4 = TickCount::now();

	TickCount::interval_t tt;
	tt = t4 - t1 - interval;

	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
	std::cout << "Total physical time for computation: " << GlobalStaticVariables::physical_time_ << " seconds." << std::endl;
	return 0;
}