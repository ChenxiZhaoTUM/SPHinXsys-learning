/**
 * @file 	2d_slabs_heat_transfer_two_phase.cpp
 * @brief 	heat transfer in slabs with different materials
 * @details 
 * @author 	
 */
#include "sphinxsys.h"
using namespace SPH;
//------------------------------------------------------------------------------
// global parameters for the case
//------------------------------------------------------------------------------
Real rho0_l = 1.226;						 /**< Reference density of left material. */
Real Cp_l = 1.012;
Real k_l = 0.0254;
Real diffusion_coff_l = k_l / (Cp_l * rho0_l);
Real initial_temperature_left = 0.0;

Real rho0_r = 1000.0;					 /**< Reference density of right material. */
Real Cp_r = 4.179;
Real k_r = 0.620;
Real diffusion_coff_r = k_r / (Cp_r * rho0_r);
Real initial_temperature_right = 1.0;

Real dp = 0.05/2.0;	/**< Initial reference particle spacing. */
std::string diffusion_species_name = "Phi";
//----------------------------------------------------------------------
//	cases-dependent geometric shape
//----------------------------------------------------------------------
std::vector<Vecd> createOverallShape()
{
	std::vector<Vecd> over_all_shape;
	over_all_shape.push_back(Vecd(0.0, 0.0));
	over_all_shape.push_back(Vecd(0.0, 0.5));
	over_all_shape.push_back(Vecd(1.0, 0.5));
	over_all_shape.push_back(Vecd(1.0, 0.0));
	over_all_shape.push_back(Vecd(0.0, 0.0));

	return over_all_shape;
}

std::vector<Vecd> createLeftBlockShape()
{
    std::vector<Vecd> left_block_shape;
	left_block_shape.push_back(Vecd(0.0, 0.0));
	left_block_shape.push_back(Vecd(0.0, 0.5));
	left_block_shape.push_back(Vecd(0.5, 0.5));
	left_block_shape.push_back(Vecd(0.5, 0.0));
	left_block_shape.push_back(Vecd(0.0, 0.0));

    return left_block_shape;
}


std::vector<Vecd> createRightBlockShape()
{
    std::vector<Vecd> right_block_shape;
	right_block_shape.push_back(Vecd(0.5, 0.0));
	right_block_shape.push_back(Vecd(0.5, 0.5));
	right_block_shape.push_back(Vecd(1.0, 0.5));
	right_block_shape.push_back(Vecd(1.0, 0.0));
	right_block_shape.push_back(Vecd(0.5, 0.0));

    return right_block_shape;
}

class LeftBlock : public MultiPolygonShape
{
public:
	explicit LeftBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(createLeftBlockShape(), ShapeBooleanOps::add);
	}
};

class RightBlock : public MultiPolygonShape
{
public:
	explicit RightBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
	{
		multi_polygon_.addAPolygon(createRightBlockShape(), ShapeBooleanOps::add);
	}
};

StdVec<Vecd> createObservationPoints()
{
    StdVec<Vecd> observation_points;
    /** A line of measuring points at the middle line. */
    size_t number_of_observation_points = 51;
    Real range_of_measure = 1.0;
    Real start_of_measure = 0;

    for (size_t i = 0; i < number_of_observation_points; ++i)
    {
        Vec2d point_coordinate(range_of_measure * Real(i) / Real(number_of_observation_points - 1) + start_of_measure, 0.5);
        observation_points.push_back(point_coordinate);
    }
    return observation_points;
};

//----------------------------------------------------------------------
//	Application dependent initial condition.
//----------------------------------------------------------------------
class LeftDiffusionInitialCondition : public LocalDynamics
{
  public:
    explicit LeftDiffusionInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name))
	{};

    void update(size_t index_i, Real dt)
    {
        phi_[index_i] = initial_temperature_left;
    };

  protected:
    Real *phi_;
};

class RightDiffusionInitialCondition : public LocalDynamics
{
  public:
    explicit RightDiffusionInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name))
	{};

    void update(size_t index_i, Real dt)
    {
        phi_[index_i] = initial_temperature_right;
    };

  protected:
    Real *phi_;
};

using MultiPhaseDiffusionBodyRelaxation = MultiPhaseDiffusionBodyRelaxationComplex<
    IsotropicThermalDiffusion, KernelGradientInner, KernelGradientContact>;

int main(int ac, char* av[])
{
	//----------------------------------------------------------------------
	//	Build up an SPHSystem.
	//----------------------------------------------------------------------
	BoundingBox system_domain_bounds(Vec2d(0.0, 0.0), Vec2d(1.0, 1.0));
	SPHSystem sph_system(system_domain_bounds, dp);
	sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
	//----------------------------------------------------------------------
	//	Creating bodies with corresponding materials and particles.
	//----------------------------------------------------------------------
	FluidBody left_diffusion_body(sph_system, makeShared<LeftBlock>("LeftDiffusionBody"));
	left_diffusion_body.defineClosure<Solid, IsotropicThermalDiffusion>(
        Solid(), ConstructArgs(diffusion_species_name, k_l, rho0_l, Cp_l));
	left_diffusion_body.generateParticles<BaseParticles, Lattice>();

	FluidBody right_diffusion_body(sph_system, makeShared<RightBlock>("RightDiffusionBody"));
	right_diffusion_body.defineClosure<Solid, IsotropicThermalDiffusion>(
        Solid(), ConstructArgs(diffusion_species_name, k_r, rho0_r, Cp_r));
	right_diffusion_body.generateParticles<BaseParticles, Lattice>();

    ObserverBody temperature_observer(sph_system, "TemperatureObserver");
    temperature_observer.generateParticles<ObserverParticles>(createObservationPoints());
	//----------------------------------------------------------------------
	//	Define body relation map.
	//	The contact map gives the topological connections between the bodies.
	//	Basically the the range of bodies to build neighbor particle lists.
	//----------------------------------------------------------------------
	InnerRelation left_diffusion_body_inner(left_diffusion_body);
	ContactRelation left_right_contact(left_diffusion_body, {&right_diffusion_body});
	InnerRelation right_diffusion_body_inner(right_diffusion_body);
	ContactRelation right_left_contact(right_diffusion_body, {&left_diffusion_body});

	ComplexRelation left_complex(left_diffusion_body_inner, {&left_right_contact});
    ComplexRelation right_complex(right_diffusion_body_inner, {&right_left_contact});

    ContactRelation temperature_observer_contact(temperature_observer, {&left_diffusion_body, &right_diffusion_body});
	//----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
	MultiPhaseDiffusionBodyRelaxation left_temperature_relaxation(
        left_diffusion_body_inner, left_right_contact);
    MultiPhaseDiffusionBodyRelaxation right_temperature_relaxation(
        right_diffusion_body_inner, right_left_contact);
    GetDiffusionTimeStepSize left_get_time_step_size(left_diffusion_body);
    GetDiffusionTimeStepSize right_get_time_step_size(right_diffusion_body);
    SimpleDynamics<LeftDiffusionInitialCondition> setup_left_diffusion_initial_condition(left_diffusion_body);
    SimpleDynamics<RightDiffusionInitialCondition> setup_right_diffusion_initial_condition(right_diffusion_body);
	//----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(sph_system);
    ObservedQuantityRecording<Real> write_temperature("Phi", temperature_observer_contact);
	//----------------------------------------------------------------------
	//	Prepare the simulation with cell linked list, configuration
	//	and case specified initial condition if necessary.
	//----------------------------------------------------------------------
	sph_system.initializeSystemCellLinkedLists();
	sph_system.initializeSystemConfigurations();
	setup_left_diffusion_initial_condition.exec();
    setup_right_diffusion_initial_condition.exec();
	//----------------------------------------------------------------------
	//	Setup for time-stepping control
	//----------------------------------------------------------------------
	Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
	int ite = 0.0;
	Real end_time = 1.0;
	Real Output_Time = 0.1 * end_time;
	Real Observe_time = 0.05 * Output_Time;
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
	//----------------------------------------------------------------------
	//	Main loop starts here.
	//----------------------------------------------------------------------
	while (physical_time < end_time)
    {
        Real integration_time = 0.0;
        while (integration_time < Output_Time)
        {
            Real relaxation_time = 0.0;
            while (relaxation_time < Observe_time)
            {
                if (ite % 10 == 0)
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
        write_temperature.writeToFile(ite);
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