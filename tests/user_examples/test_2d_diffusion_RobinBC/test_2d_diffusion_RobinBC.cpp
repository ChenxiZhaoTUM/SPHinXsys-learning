/**
 * @file 	diffusion_heat_flux.cpp
 * @brief 	This is the second test to validate the optimization.
 * @author 	Bo Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //SPHinXsys Library
using namespace SPH; //Namespace cite here
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0;
Real H = 1.0;
Real resolution_ref = H / 100.0;
Real BW = resolution_ref * 2.0;
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real diffusion_coff = 1;
std::array<std::string, 1> species_name_list{ "Phi" };
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real initial_temperature = 0.0;
Real left_temperature = 300.0;
Real right_temperature = 350.0;
Real T_infinity = 100.0;
Real convection = 100.0;
//----------------------------------------------------------------------
//	Geometric shapes used in the system.
//----------------------------------------------------------------------
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

std::vector<Vecd> left_temperature_region
{
	Vecd(0.3 * L, 0), Vecd(0.3 * L, BW), Vecd(0.4 * L, BW),
	Vecd(0.4 * L, 0), Vecd(0.3 * L, 0)
};

std::vector<Vecd> right_temperature_region
{
	Vecd(0.6 * L, 0), Vecd(0.6 * L, BW), Vecd(0.7 * L, BW),
	Vecd(0.7 * L, 0), Vecd(0.6 * L, 0)
};

std::vector<Vecd> heat_flux_region
{
	Vecd(0.45 * L, -BW), Vecd(0.45 * L, 0), Vecd(0.55 * L, 0),
	Vecd(0.55 * L, -BW), Vecd(0.45 * L, -BW)
};

std::vector<Vecd> heat_flux_boundary
{
	Vecd(0.45 * L, 0), Vecd(0.45 * L, resolution_ref), Vecd(0.55 * L, resolution_ref),
	Vecd(0.55 * L, 0), Vecd(0.45 * L, 0)
};
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

MultiPolygon createBoundayConditionRegion()
{
	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(left_temperature_region, ShapeBooleanOps::add);
	multi_polygon.addAPolygon(right_temperature_region, ShapeBooleanOps::add);
	multi_polygon.addAPolygon(heat_flux_region, ShapeBooleanOps::add);
	return multi_polygon;
}

MultiPolygon createHeatFluxBoundary()
{
	MultiPolygon multi_polygon;
	multi_polygon.addAPolygon(heat_flux_boundary, ShapeBooleanOps::add);
	return multi_polygon;
}
//----------------------------------------------------------------------
//	Setup diffusion material properties. 
//----------------------------------------------------------------------
class DiffusionBodyMaterial :public DiffusionReaction<Solid>
{
public:
	DiffusionBodyMaterial() :DiffusionReaction<Solid>(species_name_list)
	{
		initializeAnDiffusion<IsotropicDiffusion>("Phi", "Phi", diffusion_coff);
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
	void update(size_t index_i, Real dt)
	{
		species_n_[phi_][index_i] = 400 + 50 * (double)rand() / RAND_MAX;
	};
public:
	DiffusionBodyInitialCondition(SolidBody& diffusion_body) :
		DiffusionReactionInitialCondition<SolidParticles, Solid>(diffusion_body)
	{
		phi_ = particles_->diffusion_reaction_material_.SpeciesIndexMap()["Phi"];
	}
};

class WallBoundaryInitialCondition
	: public DiffusionReactionInitialCondition<SolidParticles, Solid>
{
protected:
	size_t phi_;

	void update(size_t index_i, Real dt)
	{
		species_n_[phi_][index_i] = -0.0;
		if (pos_[index_i][1] > H && pos_[index_i][0] > 0.3 * L && pos_[index_i][0] < 0.4 * L)
		{
			species_n_[phi_][index_i] = left_temperature;
		}
		if (pos_[index_i][1] > H && pos_[index_i][0] > 0.6 * L && pos_[index_i][0] < 0.7 * L)
		{
			species_n_[phi_][index_i] = right_temperature;
		}
		if (pos_[index_i][1] < 0 && pos_[index_i][0] > 0.0 && pos_[index_i][0] < L)
		{
			T_infinity_[index_i] = T_infinity;
			convection_[index_i] = convection;
		}
	}
public:
	WallBoundaryInitialCondition(SolidBody& diffusion_body) :
		DiffusionReactionInitialCondition<SolidParticles, Solid>(diffusion_body)
	{
		phi_ = particles_->diffusion_reaction_material_.SpeciesIndexMap()["Phi"];
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
class TemperatureObserverParticleGenerator : public ObserverParticleGenerator
{
public:
	TemperatureObserverParticleGenerator(SPHBody& sph_body) : ObserverParticleGenerator(sph_body)
	{
		/** A line of measuring points at the middle line. */
		size_t number_of_observation_points = 11;
		Real range_of_measure = L;
		Real start_of_measure = 0;

		for (size_t i = 0; i < number_of_observation_points; ++i)
		{
			Vec2d point_coordinate(0.5 * L, range_of_measure * Real(i) /
				Real(number_of_observation_points - 1) + start_of_measure);
			positions_.push_back(point_coordinate);
		}
	}
};

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
	diffusion_body.defineParticlesAndMaterial<DiffusionReactionParticles<SolidParticles, Solid>, DiffusionBodyMaterial>();
	diffusion_body.generateParticles<ParticleGeneratorLattice>();

	SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
	wall_boundary.defineParticlesAndMaterial<DiffusionReactionParticles<SolidParticles, Solid>, DiffusionBodyMaterial>();
	wall_boundary.generateParticles<ParticleGeneratorLattice>();

	BodyRegionByParticle heat_flux_region(wall_boundary, makeShared<MultiPolygonShape>(createHeatFluxBoundary(), "heat_flux_region"));
	//----------------------------  ------------------------------------------
	//	Particle and body creation of temperature observers.
	//----------------------------------------------------------------------
	ObserverBody temperature_observer(sph_system, "TemperatureObserver");
	temperature_observer.generateParticles<TemperatureObserverParticleGenerator>();
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	ComplexRelation diffusion_body_complex(diffusion_body, { &wall_boundary });
	ComplexRelation wall_boundary_complex(wall_boundary, { &diffusion_body });
	ContactRelation temperature_observer_contact(temperature_observer, { &diffusion_body });
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	SimpleDynamics<DiffusionBodyInitialCondition> setup_diffusion_initial_condition(diffusion_body);
	SimpleDynamics<WallBoundaryInitialCondition> setup_boundary_condition(wall_boundary);
	GetDiffusionTimeStepSize<SolidParticles, Solid> get_time_step_size(diffusion_body);

	//----------------------------------------------------------------------
	//	Define the methods for I/O operations and observations of the simulation.
	//----------------------------------------------------------------------
	BodyStatesRecordingToVtp write_states(io_environment, sph_system.real_bodies_);
	ObservedQuantityRecording<Real> write_solid_temperature("Phi", io_environment, temperature_observer_contact);
	//----------------------------------------------------------------------
	//	Define the main numerical methods used in the simulation.
	//	Note that there may be data dependence on the constructors of these methods.
	//----------------------------------------------------------------------
	DiffusionBodyRelaxation temperature_relaxation(diffusion_body_complex);
	InteractionDynamics<UpdateUnitVectorNormalToBoundary<SolidParticles, Solid, SolidParticles, Solid>> update_diffusion_body_normal_vector(diffusion_body_complex);
	InteractionDynamics<UpdateUnitVectorNormalToBoundary<SolidParticles, Solid, SolidParticles, Solid>> update_wall_boundary_normal_vector(wall_boundary_complex);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary. 
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	setup_diffusion_initial_condition.parallel_exec();
	setup_boundary_condition.parallel_exec();
	update_diffusion_body_normal_vector.parallel_exec();
	update_wall_boundary_normal_vector.parallel_exec();
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
				if (ite % 500 == 0)
				{
					std::cout << "N=" << ite << " Time: "
						<< GlobalStaticVariables::physical_time_ << "	dt: "
						<< dt << "\n";
				}

				temperature_relaxation.parallel_exec(dt);

				ite++;
				dt = get_time_step_size.parallel_exec();
				relaxation_time += dt;
				integration_time += dt;
				GlobalStaticVariables::physical_time_ += dt;
			}
		}

		tick_count t2 = tick_count::now();
		write_states.writeToFile();
		write_solid_temperature.writeToFile(ite);
		tick_count t3 = tick_count::now();
		interval += t3 - t2;
	}
	tick_count t4 = tick_count::now();

	tick_count::interval_t tt;
	tt = t4 - t1 - interval;

	std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
	std::cout << "Total physical time for computation: " << GlobalStaticVariables::physical_time_ << " seconds." << std::endl;
	return 0;
}