/**
 * @file 	diffusion.cpp
 * @brief 	This is the first test to validate our anisotropic diffusion solver.
 * @author 	Chi Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //SPHinXsys Library
using namespace SPH;   // Namespace cite here
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0;
Real H = 1.0;
Real resolution_ref = H / 20.0;
Real BW = resolution_ref * 4.0;
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diffusion_coff = 1.0;
Real bias_diffusion_coff = 0.0;
Real alpha = Pi / 4.0;
Vec2d bias_direction(cos(alpha), sin(alpha));
std::array<std::string, 1> species_name_list{ "Phi" };
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real initial_temperature = 0.0;
Real left_temperature = 300.0;
Real right_temperature = 350.0;
Real heat_flux = 2000.0;
std::vector<Vecd> createThermalDomain()
{
	std::vector<Vecd> thermalDomainShape;
	thermalDomainShape.push_back(Vecd(0.0, 0.0));
	thermalDomainShape.push_back(Vecd(0.0, H));
	thermalDomainShape.push_back(Vecd(L, H));
	thermalDomainShape.push_back(Vecd(L, 0.0));
	thermalDomainShape.push_back(Vecd(0.0, 0.0));
	return thermalDomainShape;
}
std::vector<Vecd> createBoundaryDomain()
{
	std::vector<Vecd> boundaryDomain;
	boundaryDomain.push_back(Vecd(-BW, -BW));
	boundaryDomain.push_back(Vecd(-BW, H + BW));
	boundaryDomain.push_back(Vecd(L + BW, H + BW));
	boundaryDomain.push_back(Vecd(L + BW, -BW));
	boundaryDomain.push_back(Vecd(-BW, -BW));
	return boundaryDomain;
}
/** The domain used for objective function. */
std::vector<Vecd> heat_flux_boundary_domain{
	Vecd(0.45 * L, H), Vecd(0.55 * L, H), Vecd(0.55 * L, H - resolution_ref),
	Vecd(0.45 * L, H - resolution_ref), Vecd(0.45 * L, H) };
//----------------------------------------------------------------------
//	Define SPH bodies. 
//----------------------------------------------------------------------
class DiffusionBody : public MultiPolygonShape
{
public:
	explicit DiffusionBody(const std::string& shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(createThermalDomain(), ShapeBooleanOps::add);
	}
};
class WallBoundary : public MultiPolygonShape
{
public:
	explicit WallBoundary(const std::string& shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(createBoundaryDomain(), ShapeBooleanOps::add);
		multi_polygon_.addAPolygon(createThermalDomain(), ShapeBooleanOps::sub);
	}
};
MultiPolygon createHeatFluxSurfaceDomain()
{
	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(heat_flux_boundary_domain, ShapeBooleanOps::add);
	return multi_polygon;
}
//----------------------------------------------------------------------
//	Setup diffusion material properties. 
//----------------------------------------------------------------------
class DiffusionBodyMaterial : public DiffusionReaction<Solid>
{
public:
	DiffusionBodyMaterial() : DiffusionReaction<Solid>(species_name_list)
	{
		initializeAnDiffusion<LocalDirectionalDiffusion>("Phi", "Phi", diffusion_coff, bias_diffusion_coff, bias_direction);
	}
};
//----------------------------------------------------------------------
//	Application dependent initial condition. 
//----------------------------------------------------------------------
class DiffusionBodyInitialCondition
	: public DiffusionReactionInitialCondition<SolidParticles, Solid>
{
protected:
	size_t phi_;
	
public:
	DiffusionBodyInitialCondition(SPHBody& sph_body) :
		DiffusionReactionInitialCondition<SolidParticles, Solid>(sph_body)
	{
		phi_ = particles_->diffusion_reaction_material_.SpeciesIndexMap()["Phi"];
	}

	void update(size_t index_i, Real dt)
	{
		/** setup the initial random temperature, and the value is higher than averaged temperature. */
		species_n_[phi_][index_i] = 300 + 50 * (double)rand() / RAND_MAX;
	};
};
class WallBoundaryInitialCondition
	: public DiffusionReactionInitialCondition<SolidParticles, Solid>
{
protected:
	size_t phi_;
	
public:
	WallBoundaryInitialCondition(SPHBody& sph_body) :
		DiffusionReactionInitialCondition<SolidParticles, Solid>(sph_body)
	{
		phi_ = particles_->diffusion_reaction_material_.SpeciesIndexMap()["Phi"];
	}
	void update(size_t index_i, Real dt)
	{
		species_n_[phi_][index_i] = -0.0;
		if (pos_[index_i][1] < 0 && pos_[index_i][0] > 0.3 * L && pos_[index_i][0] < 0.4 * L)
		{
			species_n_[phi_][index_i] = left_temperature;
		}
		if (pos_[index_i][1] < 0 && pos_[index_i][0] > 0.6 * L && pos_[index_i][0] < 0.7 * L)
		{
			species_n_[phi_][index_i] = right_temperature;
		}
		if (pos_[index_i][1] > H && pos_[index_i][0] > 0.45 * L && pos_[index_i][0] < 0.55 * L)
		{
			heat_flux_[index_i] = heat_flux;
		}
	}
};
//----------------------------------------------------------------------
//	Specify diffusion relaxation method. 
//----------------------------------------------------------------------
class DiffusionBodyRelaxation
	:public RelaxationOfAllDiffusionSpeciesRK2<
	RelaxationOfAllDiffusionSpeciesWithBoundary<SolidParticles, Solid, SolidParticles, Solid>>
{
public:
	DiffusionBodyRelaxation(ComplexRelation& body_complex_relation)
		:RelaxationOfAllDiffusionSpeciesRK2(body_complex_relation) {};
	virtual ~DiffusionBodyRelaxation() {};
};
//----------------------------------------------------------------------
//	An observer body to measure temperature at given positions. 
//----------------------------------------------------------------------
class ThermalDiffusivityObserverParticleGenerator : public ObserverParticleGenerator
{
public:
	ThermalDiffusivityObserverParticleGenerator(SPHBody& sph_body) : ObserverParticleGenerator(sph_body)
	{
		/** A line of measuring points at the middle line. */
		size_t number_of_observation_points = 100;
		Real range_of_measure = L;
		Real start_of_measure = 0;
		for (size_t i = 0; i < number_of_observation_points; ++i)
		{
			Vec2d point_coordinate(0.5 * L, range_of_measure * Real(i) / Real(number_of_observation_points - 1) + start_of_measure);
			positions_.push_back(point_coordinate);
		}
	}
};
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main()
{
	//----------------------------------------------------------------------
	//	Build up the environment of a SPHSystem.
	//----------------------------------------------------------------------
	SPHSystem sph_system(system_domain_bounds, resolution_ref);
	
	/** output environment. */
	IOEnvironment io_environment(sph_system);
	//----------------------------------------------------------------------
	//	Creating body, materials and particles.
	//----------------------------------------------------------------------
	//---------------------------------------------------------------------- 
	SolidBody diffusion_body(sph_system, makeShared<DiffusionBody>("DiffusionBody"));
	diffusion_body.defineParticlesAndMaterial<DiffusionReactionParticles<SolidParticles,Solid>, DiffusionBodyMaterial>();
	diffusion_body.generateParticles<ParticleGeneratorLattice>();
	SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
	wall_boundary.defineParticlesAndMaterial<DiffusionReactionParticles<SolidParticles,Solid>, DiffusionBodyMaterial>();
	wall_boundary.generateParticles<ParticleGeneratorLattice>();
	MultiPolygonShape heat_flux_surface_shape(createHeatFluxSurfaceDomain(), "Heat_flux_surface");
	BodyRegionByParticle heat_flux_shape(diffusion_body, makeShared<MultiPolygonShape>("Heat_flux_surface"));
	//----------------------------  ------------------------------------------
	//	Particle and body creation of temperature observers.
	//----------------------------------------------------------------------
	ObserverBody thermal_diffusivity_observer(sph_system, "ThermalDiffusivityObserver");
	thermal_diffusivity_observer.generateParticles<ThermalDiffusivityObserverParticleGenerator>();
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	InnerRelation diffusion_body_inner_relation(diffusion_body);
	
	ComplexRelation diffusion_body_complex(diffusion_body, { &wall_boundary });
	ComplexRelation wall_boundary_complex(wall_boundary, { &diffusion_body });

	ContactRelation thermal_diffusivity_observer_contact(thermal_diffusivity_observer, { &diffusion_body });
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	//DiffusionBodyRelaxation diffusion_relaxation_body(diffusion_body_complex);
	//DiffusionBodyRelaxation diffusion_relaxation_wall(wall_boundary_complex);
	SimpleDynamics<DiffusionBodyInitialCondition> setup_diffusion_initial_condition(diffusion_body);
	SimpleDynamics<WallBoundaryInitialCondition> setup_boundary_condition(wall_boundary);
	
	InteractionDynamics<solid_dynamics::CorrectConfiguration> correct_configuration(diffusion_body_inner_relation);

	/** Calculate unit vector normal to the boundary. */
	/*UpdateUnitVectorNormalToBoundary<SolidParticles, Solid, SolidParticles, Solid>
		update_diffusion_body_normal_vector(diffusion_body_complex);
	UpdateUnitVectorNormalToBoundary<SolidParticles, Solid,SolidParticles, Solid>
		update_wall_boundary_normal_vector(wall_boundary_complex);*/

	InteractionDynamics<UpdateUnitVectorNormalToBoundary<SolidParticles, Solid, SolidParticles, Solid>> update_diffusion_body_normal_vector(diffusion_body_complex);
	InteractionDynamics<UpdateUnitVectorNormalToBoundary<SolidParticles, Solid, SolidParticles, Solid>> update_wall_boundary_normal_vector(wall_boundary_complex);

	/** Time step size calculation. */
	GetDiffusionTimeStepSize<SolidParticles, Solid> get_time_step_size(diffusion_body);
	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	//RestartIO	restart_io(io_environment, sph_system.real_bodies_);
	ObservedQuantityRecording<Real>
		write_solid_temperature("ThermalDiffusivity", io_environment, thermal_diffusivity_observer_contact);
	/*RegressionTestEnsembleAveraged<ObservedQuantityRecording<Real>>
		write_solid_temperature("Phi", io_environment, thermal_diffusivity_observer_contact);*/
	/************************************************************************/
	/*            splitting thermal diffusivity optimization                */
	/************************************************************************/
	DiffusionBodyRelaxation thermal_relaxation_with_boundary(diffusion_body_complex);
	/*TemperatureSplittingWithBoundary<SolidParticles, Solid, SolidBody, SolidParticles, Solid, Real>
		temperature_splitting_with_boundary(diffusion_body_complex, "Phi");*/
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary. 
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();

	correct_configuration.parallel_exec();

	setup_diffusion_initial_condition.parallel_exec();
	setup_boundary_condition.parallel_exec();
	update_diffusion_body_normal_vector.parallel_exec();
	update_wall_boundary_normal_vector.parallel_exec();
	//----------------------------------------------------------------------
	//	Load restart file if necessary.
	//----------------------------------------------------------------------
	/*if (sph_system.restart_step_ != 0)
	{
		GlobalStaticVariables::physical_time_ = restart_io.readRestartFiles(sph_system.restart_step_);
		diffusion_body.updateCellLinkedList();
		diffusion_body_complex.updateConfiguration();
	}*/
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	//int ite = sph_system.restart_step_;
	//int ite_splitting = 0;
	int ite = 0;
	Real T0 = 10.0;
	Real End_Time = T0;
	Real Output_Time = 0.1 * End_Time;
	Real Observe_time = 0.01 * End_Time;
	//int restart_output_interval = 1000;
	Real dt = 0.0;
	/** Output global basic parameters.*/
	write_solid_temperature.writeToFile(ite);
	write_states.writeToFile(ite);
	//----------------------------------------------------------------------
	//	Statistics for CPU time
	//----------------------------------------------------------------------
	tick_count t1 = tick_count::now();
	tick_count::interval_t interval;
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
				if (ite % 1000 == 0)
				{
					std::cout << "N=" << ite << " Time: "
						<< GlobalStaticVariables::physical_time_ << "	dt: "
						<< dt << "\n";
				}

				thermal_relaxation_with_boundary.parallel_exec(dt);

				ite++;
				dt = get_time_step_size.parallel_exec();
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
			}
			write_states.writeToFile();
		}

		tick_count t2 = tick_count::now();
		
		write_solid_temperature.writeToFile(ite);
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1;
	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
	return 0;
}