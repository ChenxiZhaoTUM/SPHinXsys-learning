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
	sph_system.generate_regression_data_ = false;
	IOEnvironment io_environment(sph_system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	FluidBody aqueous_body(sph_system, makeShared<DiffusionBody>("AqueousBody"));
	SharedPtr<LangmuirAdsorptionModel> langmuir_adsorption_model_ptr = makeShared<LangmuirAdsorptionModel>(k_A_ad, k_A_de, k_B_ad, k_B_de, adsorption_sites_A, adsorption_sites_B, Y_A_max, Y_B_max);
	aqueous_body.defineParticlesAndMaterial<AqueousParticles, AqueousSpecies>(langmuir_adsorption_model_ptr, TypeIdentity<IsotropicDiffusion>(), diff_cf_A_aqueous, diff_cf_B_aqueous);
	aqueous_body.generateParticles<ParticleGeneratorLattice>();

	SolidBody wall_boundary_Dirichlet(sph_system, makeShared<DirichletWallBoundary>("DirichletWallBoundary"));
	wall_boundary_Dirichlet.defineParticlesAndMaterial<AdsorbedParticles, AdsorbedSpecies>(langmuir_adsorption_model_ptr);
	wall_boundary_Dirichlet.generateParticles<ParticleGeneratorLattice>();
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
	InnerRelation aqueous_body_inner_relation(aqueous_body);
	ContactRelation adsorbed_body_contact_Dirichlet(aqueous_body, { &wall_boundary_Dirichlet });
	ContactRelation temperature_observer_contact(temperature_observer, { &aqueous_body });
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	DiffusionBodyRelaxation temperature_relaxation(aqueous_body_inner_relation, adsorbed_body_contact_Dirichlet);

	GetDiffusionTimeStepSize<AqueousParticles> get_time_step_size(aqueous_body);

	SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(aqueous_body);
	SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_boundary_condition_Dirichlet(wall_boundary_Dirichlet);

	SimpleDynamics<NormalDirectionFromBodyShape> diffusion_body_normal_direction(aqueous_body);
	SimpleDynamics<NormalDirectionFromBodyShape> Dirichlet_normal_direction(wall_boundary_Dirichlet);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	//ObservedQuantityRecording<Real> write_solid_temperature("Phi", io_environment, temperature_observer_contact);
	RegressionTestEnsembleAveraged<ObservedQuantityRecording<Real>>
		write_solid_temperature("Phi", io_environment, temperature_observer_contact);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary. 
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();

	setup_diffusion_initial_condition.exec();

	setup_boundary_condition_Dirichlet.exec();

	diffusion_body_normal_direction.exec();
	Dirichlet_normal_direction.exec();
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
	write_solid_temperature.writeToFile();
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

				temperature_relaxation.exec(dt);

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
	
	if (sph_system.generate_regression_data_)
	{
		write_solid_temperature.generateDataBase(1.0e-3, 1.0e-3);
	}
	else if (sph_system.RestartStep() == 0)
	{
		write_solid_temperature.testResult();
	}
	
	return 0;
}