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
Real H = 1.0;
Real resolution_ref = H / 50;
Real BW = resolution_ref * 3.0;
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(L + BW, H + BW));
//----------------------------------------------------------------------
//	Basic parameters for material properties.
//----------------------------------------------------------------------
Real thermal_conductivity_ref = 1.0;
Real rho_ref = 10.0;
Real C_v = 1.5;
Real diffusion_coeff = thermal_conductivity_ref / rho_ref/ C_v;
std::string diffusion_species_name = "Phi";
//----------------------------------------------------------------------
//	Initial and boundary conditions.
//----------------------------------------------------------------------
Real left_temperature = 0;
Real right_temperature = 0;
Real up_temperature = 0;
Real down_temperature = 0;
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

    class UpDirichletWallBoundary : public MultiPolygonShape
    {
    public:
        explicit UpDirichletWallBoundary(const std::string& shape_name) : MultiPolygonShape(shape_name)
        {
            multi_polygon_.addAPolygon(up_heat_flux_region, ShapeBooleanOps::add);
        }
    };

    class DownDirichletWallBoundary : public MultiPolygonShape
    {
    public:
        explicit DownDirichletWallBoundary(const std::string& shape_name) : MultiPolygonShape(shape_name)
        {
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

            phi_[index_i] = sin(Pi * pos_[index_i][0]) * sin(Pi * pos_[index_i][1]);
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

    class UpDirichletWallBoundaryInitialCondition : public LocalDynamics
    {
    public:
        explicit UpDirichletWallBoundaryInitialCondition(SPHBody& sph_body)
            : LocalDynamics(sph_body),
            pos_(particles_->getVariableDataByName<Vecd>("Position")),
            phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) {};

        void update(size_t index_i, Real dt)
        {
            phi_[index_i] = up_temperature;
        }

    protected:
        Vecd* pos_;
        Real* phi_;
    };

    class DownDirichletWallBoundaryInitialCondition : public LocalDynamics
    {
    public:
        explicit DownDirichletWallBoundaryInitialCondition(SPHBody& sph_body)
            : LocalDynamics(sph_body),
            pos_(particles_->getVariableDataByName<Vecd>("Position")),
            phi_(particles_->registerStateVariable<Real>(diffusion_species_name)) {};

        void update(size_t index_i, Real dt)
        {
            phi_[index_i] = down_temperature;
        }

    protected:
        Vecd* pos_;
        Real* phi_;
    };

    //----------------------------------------------------------------------
    //	Specify diffusion relaxation method.
    //----------------------------------------------------------------------
    using DiffusionBodyRelaxation = DiffusionBodyRelaxationComplex<
        IsotropicDiffusion, KernelGradientInner, KernelGradientContact, Dirichlet, Dirichlet, Dirichlet, Dirichlet>;

    StdVec<Vecd> createXObservationPoints()
    {
        StdVec<Vecd> observation_points;
        /** A line of measuring points at the middle line. */
        size_t number_of_observation_points = 51;
        Real range_of_measure = L;
        Real start_of_measure = 0;

        for (size_t i = 0; i < number_of_observation_points; ++i)
        {
            Vec2d point_coordinate(range_of_measure * Real(i) / Real(number_of_observation_points - 1) + start_of_measure, 0.5 * H);
            observation_points.push_back(point_coordinate);
        }
        return observation_points;
    };

    StdVec<Vecd> createYObservationPoints()
    {
        StdVec<Vecd> observation_points;
        /** A line of measuring points at the middle line. */
        size_t number_of_observation_points = 11;
        Real range_of_measure = H;
        Real start_of_measure = 0;

        for (size_t i = 0; i < number_of_observation_points; ++i)
        {
            Vec2d point_coordinate( 0.5 * L, range_of_measure * Real(i) / Real(number_of_observation_points - 1) + start_of_measure);
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
    diffusion_body.defineClosure<Solid, IsotropicDiffusion>(
        Solid(), ConstructArgs(diffusion_species_name, diffusion_coeff));
    diffusion_body.generateParticles<BaseParticles, Lattice>();

    SolidBody left_wall_Dirichlet(sph_system, makeShared<LeftDirichletWallBoundary>("LeftDirichletWallBoundary"));
    left_wall_Dirichlet.defineMaterial<Solid>();
    left_wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

    SolidBody right_wall_Dirichlet(sph_system, makeShared<RightDirichletWallBoundary>("RightDirichletWallBoundary"));
    right_wall_Dirichlet.defineMaterial<Solid>();
    right_wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

    SolidBody up_wall_Dirichlet(sph_system, makeShared<UpDirichletWallBoundary>("UpDirichletWallBoundary"));
    up_wall_Dirichlet.defineMaterial<Solid>();
    up_wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

    SolidBody down_wall_Dirichlet(sph_system, makeShared<DownDirichletWallBoundary>("DownDirichletWallBoundary"));
    down_wall_Dirichlet.defineMaterial<Solid>();
    down_wall_Dirichlet.generateParticles<BaseParticles, Lattice>();
    //----------------------------------------------------------------------
    //	Particle and body creation of temperature observers.
    //----------------------------------------------------------------------
    ObserverBody x_temperature_observer(sph_system, "XTemperatureObserver");
    x_temperature_observer.generateParticles<ObserverParticles>(createXObservationPoints());
    ObserverBody y_temperature_observer(sph_system, "YTemperatureObserver");
    y_temperature_observer.generateParticles<ObserverParticles>(createYObservationPoints());
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the range of bodies to build neighbor particle lists.
    //----------------------------------------------------------------------
    InnerRelation diffusion_body_inner(diffusion_body);
    ContactRelation left_diffusion_body_contact_Dirichlet(diffusion_body, {&left_wall_Dirichlet});
    ContactRelation right_diffusion_body_contact_Dirichlet(diffusion_body, {&right_wall_Dirichlet});
    ContactRelation up_diffusion_body_contact_Dirichlet(diffusion_body, {&up_wall_Dirichlet});
    ContactRelation down_diffusion_body_contact_Dirichlet(diffusion_body, {&down_wall_Dirichlet});
    ContactRelation x_temperature_observer_contact(x_temperature_observer, {&diffusion_body});
    ContactRelation y_temperature_observer_contact(y_temperature_observer, {&diffusion_body});
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> diffusion_body_normal_direction(diffusion_body);
    SimpleDynamics<NormalDirectionFromBodyShape> left_wall_normal_direction(left_wall_Dirichlet);
    SimpleDynamics<NormalDirectionFromBodyShape> right_wall_normal_direction(right_wall_Dirichlet);
    SimpleDynamics<NormalDirectionFromBodyShape> up_wall_normal_direction(up_wall_Dirichlet);
    SimpleDynamics<NormalDirectionFromBodyShape> down_wall_normal_direction(down_wall_Dirichlet);

    DiffusionBodyRelaxation temperature_relaxation(
        diffusion_body_inner, left_diffusion_body_contact_Dirichlet, right_diffusion_body_contact_Dirichlet, up_diffusion_body_contact_Dirichlet, down_diffusion_body_contact_Dirichlet);
    GetDiffusionTimeStepSize get_time_step_size(diffusion_body);
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition(diffusion_body);
    SimpleDynamics<LeftDirichletWallBoundaryInitialCondition> setup_left_boundary_condition_Dirichlet(left_wall_Dirichlet);
    SimpleDynamics<RightDirichletWallBoundaryInitialCondition> setup_right_boundary_condition_Dirichlet(right_wall_Dirichlet);
    SimpleDynamics<UpDirichletWallBoundaryInitialCondition> setup_up_boundary_condition_Dirichlet(up_wall_Dirichlet);
    SimpleDynamics<DownDirichletWallBoundaryInitialCondition> setup_down_boundary_condition_Dirichlet(down_wall_Dirichlet);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states(sph_system);
    ObservedQuantityRecording<Real> write_solid_x_temperature("Phi", x_temperature_observer_contact);
    ObservedQuantityRecording<Real> write_solid_y_temperature("Phi", y_temperature_observer_contact);
    ReducedQuantityRecording<QuantitySummation<Real>> write_left_PhiFluxSum(diffusion_body, "PhiTransferFromLeftDirichletWallBoundaryFlux");
    ReducedQuantityRecording<QuantitySummation<Real>> write_right_PhiFluxSum(diffusion_body, "PhiTransferFromRightDirichletWallBoundaryFlux");
    ReducedQuantityRecording<QuantitySummation<Real>> write_up_PhiFluxSum(diffusion_body, "PhiTransferFromUpDirichletWallBoundaryFlux");
    ReducedQuantityRecording<QuantitySummation<Real>> write_down_PhiFluxSum(diffusion_body, "PhiTransferFromDownDirichletWallBoundaryFlux");
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
    setup_up_boundary_condition_Dirichlet.exec();
    setup_down_boundary_condition_Dirichlet.exec();
    diffusion_body_normal_direction.exec();
    left_wall_normal_direction.exec();
    right_wall_normal_direction.exec();
    up_wall_normal_direction.exec();
    down_wall_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int ite = 0;
    Real End_Time = 4;
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
    write_solid_x_temperature.writeToFile();
    write_solid_y_temperature.writeToFile();
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
        write_states.writeToFile();
        write_solid_x_temperature.writeToFile(ite);
        write_solid_y_temperature.writeToFile(ite);
        write_left_PhiFluxSum.writeToFile(ite);
        write_right_PhiFluxSum.writeToFile(ite);
        write_up_PhiFluxSum.writeToFile(ite);
        write_down_PhiFluxSum.writeToFile(ite);
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