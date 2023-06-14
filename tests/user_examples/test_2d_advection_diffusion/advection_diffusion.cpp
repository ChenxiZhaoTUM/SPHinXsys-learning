/**
 * @file 	advection_diffusion.cpp
 * @brief 	
 * @details 
 * @author 	Chenxi Zhao and Xiangyu Hu
 */
#include "sphinxsys.h"
#include "advection_diffusion.h"

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
	SolidBody aqueous_body(sph_system, makeShared<AqueousDomain>("AqueousBody"));
	SharedPtr<LangmuirAdsorptionModel> langmuir_adsorption_model_ptr = makeShared<LangmuirAdsorptionModel>(k_A_ad, k_A_de, k_B_ad, k_B_de, adsorption_sites_A, adsorption_sites_B, Y_A_max, Y_B_max);
	aqueous_body.defineParticlesAndMaterial<AqueousParticles, AqueousSpecies>(langmuir_adsorption_model_ptr);
	aqueous_body.generateParticles<ParticleGeneratorLattice>();

	SolidBody inner_wall_boundary(sph_system, makeShared<InnerWall>("InnerWallBoundary"));
	inner_wall_boundary.defineParticlesAndMaterial<InnerWallBoundaryParticles, AdsorbedSpecies>(langmuir_adsorption_model_ptr);
	inner_wall_boundary.generateParticles<ParticleGeneratorLattice>();

	SolidBody outer_wall_boundary(sph_system, makeShared<OuterWall>("OuterWallBoundary"));
	outer_wall_boundary.defineParticlesAndMaterial<OuterWallBoundaryParticles, AqueousSpeciesNoReaction>();
	outer_wall_boundary.generateParticles<ParticleGeneratorLattice>();

	ObserverBody aqueous_concentration_observer(sph_system, "AqueousConcentrationObserver");
	aqueous_concentration_observer.generateParticles<ObserverParticleGenerator>();
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	InnerRelation aqueous_body_inner_relation(aqueous_body);

	ContactRelation adsorbed_body_contact(aqueous_body, { &inner_wall_boundary });
	ContactRelation contact_outer_wall(aqueous_body, { &outer_wall_boundary });
	ContactRelation aqueous_concentration_observer_contact(aqueous_concentration_observer, { &aqueous_body });
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	DiffusionRelaxation diffusion_relaxation(aqueous_body_inner_relation, contact_outer_wall);

	AqueousReactionRelaxationForward aqueous_reaction_relaxation_forward(aqueous_body);
	AqueousReactionRelaxationBackward aqueous_reaction_relaxation_backward(aqueous_body);
	AdsorbedReactionRelaxationForward inner_adsorbed_reaction_relaxation_forward(inner_wall_boundary);
	AdsorbedReactionRelaxationBackward inner_adsorbed_reaction_relaxation_backward(inner_wall_boundary);

	GetDiffusionTimeStepSize<AqueousParticles> get_time_step_size(aqueous_body);

	SimpleDynamics<AqueousInitialCondition> setup_diffusion_initial_condition(aqueous_body);
	SimpleDynamics<InnerWallBoundaryInitialCondition> setup_inner_wall_boundary_condition(inner_wall_boundary);
	SimpleDynamics<OuterWallBoundaryInitialCondition> setup_outer_wall_boundary_condition(outer_wall_boundary);

	SimpleDynamics<NormalDirectionFromBodyShape> aqueous_body_normal_direction(aqueous_body);
	SimpleDynamics<NormalDirectionFromBodyShape> inner_wall_normal_direction(inner_wall_boundary);
	SimpleDynamics<NormalDirectionFromBodyShape> outer_wall_normal_direction(outer_wall_boundary);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	ObservedQuantityRecording<Real> write_XA("AAqueousConcentration", io_environment, aqueous_concentration_observer_contact);
	ObservedQuantityRecording<Real> write_XB("BAqueousConcentration", io_environment, aqueous_concentration_observer_contact);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary. 
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();

	setup_diffusion_initial_condition.exec();
	setup_inner_wall_boundary_condition.exec();
	setup_outer_wall_boundary_condition.exec();

	aqueous_body_normal_direction.exec();
	inner_wall_normal_direction.exec();
	outer_wall_normal_direction.exec();
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
	//	First output before the main loop.
	//----------------------------------------------------------------------
	write_states.writeToFile();
	write_XA.writeToFile();
	write_XB.writeToFile();
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
				aqueous_reaction_relaxation_forward.exec(0.5 * dt);
				inner_adsorbed_reaction_relaxation_forward.exec(0.5 * dt);
				diffusion_relaxation.exec(dt);
				aqueous_reaction_relaxation_backward.exec(0.5 * dt);
				inner_adsorbed_reaction_relaxation_backward.exec(0.5 * dt);

				ite++;
				dt = get_time_step_size.exec();
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
			}
		}

		TickCount t2 = TickCount::now();
		write_states.writeToFile();
		write_XA.writeToFile(ite);
		write_XB.writeToFile(ite);
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