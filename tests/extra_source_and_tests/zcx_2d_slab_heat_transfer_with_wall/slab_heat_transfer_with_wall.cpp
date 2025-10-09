/**
 * @file 	2d_slabs_heat_transfer_two_phase.cpp
 * @brief 	heat transfer in slabs with different materials
 * @details 
 * @author 	
 */
#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0;
Real H = 0.5;
Real resolution_ref = H / 50.0;
Real BW = resolution_ref * 3.0;
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real thermal_conductivity_ref = 1.0;
Real rho_ref = 1000.0;
Real diffusion_coeff_bg = thermal_conductivity_ref / rho_ref;
Real diffusion_coeff_max = diffusion_coeff_bg * std::exp(4*1);
std::string diffusion_species_name = "Phi";
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real left_temperature = 0;
Real right_temperature = 1;
Real heat_flux = 0.;
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

std::vector<Vecd> left_temperature_region{
    Vecd(-BW, 0.), Vecd(0., 0.), Vecd(0., H),
    Vecd(-BW, H), Vecd(-BW, 0.)};

std::vector<Vecd> right_temperature_region{
    Vecd(L, 0.), Vecd(L + BW,  0.), Vecd(L + BW, H),
    Vecd(L, H), Vecd(L, 0.)};

std::vector<Vecd> up_heat_flux_region{
    Vecd(-BW, H), Vecd(L+BW , H), Vecd(L+BW, H+BW),
    Vecd(-BW, H+BW), Vecd(-BW, H)};

std::vector<Vecd> down_heat_flux_region{
    Vecd(-BW, -BW), Vecd(L+BW , -BW), Vecd(L+BW, 0.),
    Vecd(-BW, 0.), Vecd(-BW, -BW)};

//----------------------------------------------------------------------
// Define extra classes which are used in the main program.
// These classes are defined under the namespace of SPH.
//----------------------------------------------------------------------
namespace SPH
{
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

    class LeftDirichletWallBoundary : public MultiPolygonShape
    {
    public:
        explicit LeftDirichletWallBoundary(const std::string& shape_name) : MultiPolygonShape(shape_name)
        {
            multi_polygon_.addAPolygon(left_temperature_region, ShapeBooleanOps::add);
        }
    };

    class RightDirichletWallBoundary : public MultiPolygonShape
    {
    public:
        explicit RightDirichletWallBoundary(const std::string& shape_name) : MultiPolygonShape(shape_name)
        {
            multi_polygon_.addAPolygon(right_temperature_region, ShapeBooleanOps::add);
        }
    };

    class NeumannWallBoundary : public MultiPolygonShape
    {
    public:
        explicit NeumannWallBoundary(const std::string& shape_name) : MultiPolygonShape(shape_name)
        {
            multi_polygon_.addAPolygon(up_heat_flux_region, ShapeBooleanOps::add);
            multi_polygon_.addAPolygon(down_heat_flux_region, ShapeBooleanOps::add);
        }
    };

    //----------------------------------------------------------------------
    //	Application dependent initial condition.
    //----------------------------------------------------------------------
    class DiffusionInitialCondition : public LocalDynamics
    {
    public:
        explicit DiffusionInitialCondition(SPHBody& sph_body)
            : LocalDynamics(sph_body),
            pos_(particles_->getVariableDataByName<Vecd>("Position")),
            phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) {};

        void update(size_t index_i, Real dt)
        {
            if (pos_[index_i][0] < 0.5)
            {
                phi_[index_i] = left_temperature;
            }
            else
            {
                phi_[index_i] = right_temperature;
            }
        };

    protected:
        Vecd* pos_;
        Real* phi_;
    };

    class LeftDirichletWallBoundaryInitialCondition : public LocalDynamics
    {
    public:
        explicit LeftDirichletWallBoundaryInitialCondition(SPHBody& sph_body)
            : LocalDynamics(sph_body),
            pos_(particles_->getVariableDataByName<Vecd>("Position")),
            phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) {};

        void update(size_t index_i, Real dt)
        {
            phi_[index_i] = left_temperature;
        }

    protected:
        Vecd* pos_;
        Real* phi_;
    };

    class RightDirichletWallBoundaryInitialCondition : public LocalDynamics
    {
    public:
        explicit RightDirichletWallBoundaryInitialCondition(SPHBody& sph_body)
            : LocalDynamics(sph_body),
            pos_(particles_->getVariableDataByName<Vecd>("Position")),
            phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) {};

        void update(size_t index_i, Real dt)
        {
            phi_[index_i] = right_temperature;
        }

    protected:
        Vecd* pos_;
        Real* phi_;
    };

    class NeumannWallBoundaryInitialCondition : public LocalDynamics
    {
    public:
        explicit NeumannWallBoundaryInitialCondition(SPHBody& sph_body)
            : LocalDynamics(sph_body),
            pos_(particles_->getVariableDataByName<Vecd>("Position")),
            phi_(particles_->registerStateVariable<Real>(diffusion_species_name)),
            phi_flux_(particles_->getVariableDataByName<Real>(diffusion_species_name + "Flux")) {}

        void update(size_t index_i, Real dt)
        {
            phi_flux_[index_i] = heat_flux;
        }

    protected:
        Vecd* pos_;
        Real* phi_, * phi_flux_;
    };

    class LocalDiffusivityDefinition : public LocalDynamics
    {
    public:
        explicit LocalDiffusivityDefinition(SPHBody& sph_body, Real thermal_diffusivity_ref, Real thermal_conductivity_ref)
            : LocalDynamics(sph_body),
            thermal_diffusivity_(particles_->getVariableDataByName<Real>("ThermalDiffusivity")),
            thermal_conductivity_(particles_->getVariableDataByName<Real>("ThermalConductivity")),
            phi_(particles_->getVariableDataByName<Real>(diffusion_species_name)),
            thermal_diffusivity_ref_(thermal_diffusivity_ref), thermal_conductivity_ref_(thermal_conductivity_ref) {};

        void update(size_t index_i, Real dt)
        {
            thermal_diffusivity_[index_i] = thermal_diffusivity_ref_ * std::exp(4 * phi_[index_i]);
            thermal_conductivity_[index_i] = thermal_conductivity_ref_ * std::exp(4 * phi_[index_i]);
        };

    protected:
        Real* thermal_diffusivity_, *thermal_conductivity_, * phi_;
        Real thermal_diffusivity_ref_, thermal_conductivity_ref_;
    };

    //----------------------------------------------------------------------
    //	Specify diffusion relaxation method.
    //----------------------------------------------------------------------
    using DiffusionBodyRelaxation = DiffusionBodyRelaxationComplex<
        IsotropicDiffusion, KernelGradientInner, KernelGradientContact, Dirichlet, Dirichlet, Neumann>;

    StdVec<Vecd> createObservationPoints()
    {
        StdVec<Vecd> observation_points;
        /** A line of measuring points at the middle line. */
        size_t number_of_observation_points = 51;
        Real range_of_measure = L;
        Real start_of_measure = 0;

        for (size_t i = 0; i < number_of_observation_points; ++i)
        {
            Vec2d point_coordinate(range_of_measure * Real(i) / Real(number_of_observation_points - 1) +start_of_measure, 0.5 * H);
            observation_points.push_back(point_coordinate);
        }
        return observation_points;
    };
} // namespace SPH

int main(int ac, char* av[])
{
	//----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBody diffusion_body(sph_system, makeShared<DiffusionBody>("DiffusionBody"));
    diffusion_body.defineClosure<Solid, LocalIsotropicDiffusion>(
        Solid(), ConstructArgs(diffusion_species_name, diffusion_coeff_bg, diffusion_coeff_max));
    diffusion_body.generateParticles<BaseParticles, Lattice>();

    SolidBody left_wall_Dirichlet(sph_system, makeShared<LeftDirichletWallBoundary>("LeftDirichletWallBoundary"));
    left_wall_Dirichlet.defineMaterial<Solid>();
    left_wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

    SolidBody right_wall_Dirichlet(sph_system, makeShared<RightDirichletWallBoundary>("RightDirichletWallBoundary"));
    right_wall_Dirichlet.defineMaterial<Solid>();
    right_wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_Neumann(sph_system, makeShared<NeumannWallBoundary>("NeumannWallBoundary"));
    wall_Neumann.defineMaterial<Solid>();
    wall_Neumann.generateParticles<BaseParticles, Lattice>();
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
    ContactRelation left_diffusion_body_contact_Dirichlet(diffusion_body, {&left_wall_Dirichlet});
    ContactRelation right_diffusion_body_contact_Dirichlet(diffusion_body, {&right_wall_Dirichlet});
    ContactRelation diffusion_body_contact_Neumann(diffusion_body, {&wall_Neumann});
    ContactRelation temperature_observer_contact(temperature_observer, {&diffusion_body});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> diffusion_body_normal_direction(diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_Neumann);

    DiffusionBodyRelaxation temperature_relaxation(
        diffusion_body_inner, left_diffusion_body_contact_Dirichlet, right_diffusion_body_contact_Dirichlet, diffusion_body_contact_Neumann);
    GetDiffusionTimeStepSize get_time_step_size(diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body);
    SimpleDynamics<LeftDirichletWallBoundaryInitialCondition> setup_left_boundary_condition_Dirichlet(left_wall_Dirichlet);
    SimpleDynamics<RightDirichletWallBoundaryInitialCondition> setup_right_boundary_condition_Dirichlet(right_wall_Dirichlet);
    SimpleDynamics<NeumannWallBoundaryInitialCondition> setup_boundary_condition_Neumann(wall_Neumann);
    SimpleDynamics<LocalDiffusivityDefinition> local_diffusivity(diffusion_body, diffusion_coeff_bg, thermal_conductivity_ref);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(sph_system);
    write_states.addToWrite<Real>(diffusion_body, "ThermalDiffusivity");
    ObservedQuantityRecording<Real> write_solid_temperature("Phi", temperature_observer_contact);
    ReducedQuantityRecording<QuantitySummation<Real>> write_left_PhiFluxSum(diffusion_body, "PhiTransferFromLeftDirichletWallBoundaryFlux");
    ReducedQuantityRecording<QuantitySummation<Real>> write_right_PhiFluxSum(diffusion_body, "PhiTransferFromRightDirichletWallBoundaryFlux");
    /*RegressionTestEnsembleAverage<ObservedQuantityRecording<Real>>
        write_solid_temperature(diffusion_species_name, temperature_observer_contact);*/
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    setup_diffusion_initial_condition.exec();
    setup_left_boundary_condition_Dirichlet.exec();
    setup_right_boundary_condition_Dirichlet.exec();
    setup_boundary_condition_Neumann.exec();
    diffusion_body_normal_direction.exec();
    wall_boundary_normal_direction.exec();
    local_diffusivity.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int ite = 0;
    Real End_Time = 40;
    Real Observe_time = End_Time / 40;
    Real Output_Time = End_Time / 40;
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
                local_diffusivity.exec();

                ite++;
                dt = get_time_step_size.exec();
                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
            }
        }

        TickCount t2 = TickCount::now();
        write_states.writeToFile();
        write_solid_temperature.writeToFile(ite);
        write_left_PhiFluxSum.writeToFile(ite);
        write_right_PhiFluxSum.writeToFile(ite);
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