/**
 * @file    natural_convection_RL_python.cpp
 * @brief    
 * @author   
 */
#include "natural_convection_RL_python.h"
#include "custom_io_environment.h"
#include "custom_io_observation.h"
#include "sphinxsys.h"
#include <pybind11/pybind11.h>

using namespace SPH;
namespace py = pybind11;

class SphBasicSystemSetting : public SphBasicGeometrySetting
{
  protected:
    BoundingBox system_domain_bounds;
    SPHSystem sph_system;
    CustomIOEnvironment custom_io_environment;

  public:
    SphBasicSystemSetting(int parallel_env, int episode_env)
        : system_domain_bounds(Vecd(-BW, - H/2 -BW), Vecd(L + BW, H/2 + BW)),
          sph_system(system_domain_bounds, particle_spacing_ref),
          custom_io_environment(sph_system, true, parallel_env, episode_env) {}
};

class SphBodyReloadEnvironment : public SphBasicSystemSetting
{
  protected:
    FluidBody diffusion_body;
    SolidBody wall_boundary, up_wall_Dirichlet, down_wall_Dirichlet, wall_Neumann;
    ObserverBody diffusion_observer;

  public:
    SphBodyReloadEnvironment(int parallel_env, int episode_env)
        : SphBasicSystemSetting(parallel_env, episode_env),
          diffusion_body(sph_system, makeShared<DiffusionBody>("DiffusionBody")),
          wall_boundary(sph_system, makeShared<WallBoundary>("EntireWallBoundary")),
          up_wall_Dirichlet(sph_system, makeShared<UpDirichletWallBoundary>("UpDirichletWallBoundary")),
          down_wall_Dirichlet(sph_system, makeShared<DownDirichletWallBoundary>("DownDirichletWallBoundary")),
          wall_Neumann(sph_system, makeShared<NeumannWallBoundary>("NeumannWallBoundary")),
          diffusion_observer(sph_system, "DiffusionObserver")
    {
        diffusion_body.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicDiffusion>(
        ConstructArgs(rho0_f, c_f), mu_f, ConstructArgs(diffusion_species_name, diffusion_coeff));
        diffusion_body.generateParticles<BaseParticles, Lattice>();

        wall_boundary.defineMaterial<Solid>();
        wall_boundary.generateParticles<BaseParticles, Lattice>();

        up_wall_Dirichlet.defineMaterial<Solid>();
        up_wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

        down_wall_Dirichlet.defineMaterial<Solid>();
        down_wall_Dirichlet.generateParticles<BaseParticles, Lattice>();

        wall_Neumann.defineMaterial<Solid>();
        wall_Neumann.generateParticles<BaseParticles, Lattice>();

        diffusion_observer.generateParticles<ObserverParticles>(createObservationPoints());
    }
};

class SphNaturalConvection : public SphBodyReloadEnvironment
{
    using DiffusionBodyRelaxation = DiffusionBodyRelaxationComplex<
        IsotropicDiffusion, KernelGradientInner, KernelGradientContact, Dirichlet, Dirichlet, Neumann>;

  protected:
    SPHSystem &sph_system_;
    InnerRelation diffusion_body_inner;
    ContactRelation up_Dirichlet_contact, down_Dirichlet_contact, diffusion_body_contact_all_Dirichlet, diffusion_body_contact_up_Dirichlet,
        diffusion_body_contact_down_Dirichlet, diffusion_body_contact_Neumann, fluid_body_contact, observer_diffusion_body_contact;
    ComplexRelation fluid_body_complex;
    //----------------------------------------------------------------------
    //	    Define all numerical methods which are used in this case.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> diffusion_body_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> entire_wall_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> up_Dirichlet_wall_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> down_Dirichlet_wall_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> Neumann_wall_normal_direction;

    DiffusionBodyRelaxation temperature_relaxation;
    GetDiffusionTimeStepSize get_thermal_time_step;
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition;
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_up_Dirichlet_initial_condition;
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_down_Dirichlet_initial_condition;
    SimpleDynamics<NeumannWallBoundaryInitialCondition> setup_boundary_condition_Neumann;

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> kernel_correction_complex;
    Dynamics1Level<fluid_dynamics::Integration1stHalfCorrectionWithWallRiemann> pressure_relaxation;
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation;
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplex> update_density_by_summation;
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_force;
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionCorrectedComplex<AllParticles>> transport_velocity_correction;
    SimpleDynamics<fluid_dynamics::BuoyancyForce> buoyancy_force;
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step;
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step;

    InteractionWithUpdate<fluid_dynamics::TargetFluidParticles> target_fluid_particles;
    SimpleDynamics<solid_dynamics::FirstLayerFromFluid> target_up_solid_particles;
    SimpleDynamics<solid_dynamics::FirstLayerFromFluid> target_down_solid_particles;
    InteractionWithUpdate<fluid_dynamics::PhiGradientWithWall<LinearGradientCorrection>> calculate_phi_gradient;;
    SimpleDynamics<fluid_dynamics::LocalNusseltNum> local_nusselt_number;
    InteractionDynamics<solid_dynamics::ProjectionForNu> up_wall_local_nusselt_number;
    InteractionDynamics<solid_dynamics::ProjectionForNu> down_wall_local_nusselt_number;
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    ParticleSorting<ParallelPolicy> particle_sorting;
    BodyStatesRecordingToVtp write_states;
    RestartIO restart_io;
    
    ExtendedReducedQuantityRecording<QuantitySummation<Real>> write_up_PhiFluxSum;
    ExtendedReducedQuantityRecording<QuantitySummation<Real>> write_down_PhiFluxSum;
    BodyRegionByParticle left_diffusion_domain;
    ExtendedReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>> write_left_PhiFluxSum;
    BodyRegionByParticle middle_diffusion_domain;
    ExtendedReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>> write_middle_PhiFluxSum;
    BodyRegionByParticle right_diffusion_domain;
    ExtendedReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>> write_right_PhiFluxSum;

    ObservedQuantityRecording<Vecd> write_recorded_fluid_vel;
    ExtendedReducedQuantityRecording<TotalKineticEnergy> write_global_kinetic_energy;
    //----------------------------------------------------------------------
    //	    Basic control parameters for time stepping.
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int ite = 0;
    Real End_Time = 120;
    Real output_interval = End_Time / 120.0;
    int number_of_iterations = 0;
    int screen_output_interval = 100;
    /** statistics for computing time. */
    TickCount t1 = TickCount::now();
    TimeInterval interval;

  public:
    explicit SphNaturalConvection(int parallel_env, int episode_env)
        : SphBodyReloadEnvironment(parallel_env, episode_env),
          sph_system_(sph_system),
          diffusion_body_inner(diffusion_body),
          up_Dirichlet_contact(up_wall_Dirichlet, {&diffusion_body}),
          down_Dirichlet_contact(down_wall_Dirichlet, {&diffusion_body}),
          diffusion_body_contact_all_Dirichlet(diffusion_body, {&up_wall_Dirichlet, &down_wall_Dirichlet}),
          diffusion_body_contact_up_Dirichlet(diffusion_body, {&up_wall_Dirichlet}),
          diffusion_body_contact_down_Dirichlet(diffusion_body, {&down_wall_Dirichlet}),
          diffusion_body_contact_Neumann(diffusion_body, {&wall_Neumann}),
          fluid_body_contact(diffusion_body, {&wall_boundary}),
          fluid_body_complex(diffusion_body_inner, fluid_body_contact),
          observer_diffusion_body_contact(diffusion_observer, {&diffusion_body}),

          diffusion_body_normal_direction(diffusion_body),
          entire_wall_normal_direction(wall_boundary),
          up_Dirichlet_wall_normal_direction(up_wall_Dirichlet),
          down_Dirichlet_wall_normal_direction(down_wall_Dirichlet),
          Neumann_wall_normal_direction(wall_Neumann),
          temperature_relaxation(
          diffusion_body_inner, diffusion_body_contact_up_Dirichlet, diffusion_body_contact_down_Dirichlet, diffusion_body_contact_Neumann),
          get_thermal_time_step(diffusion_body),
          setup_diffusion_initial_condition(diffusion_body),
          setup_up_Dirichlet_initial_condition(up_wall_Dirichlet),
          setup_down_Dirichlet_initial_condition(down_wall_Dirichlet),
          setup_boundary_condition_Neumann(wall_Neumann),

          kernel_correction_complex(InteractArgs(diffusion_body_inner, 0.1), fluid_body_contact),
          pressure_relaxation(diffusion_body_inner, fluid_body_contact),
          density_relaxation(diffusion_body_inner, fluid_body_contact),
          update_density_by_summation(diffusion_body_inner, fluid_body_contact),
          viscous_force(diffusion_body_inner, fluid_body_contact),
          transport_velocity_correction(diffusion_body_inner, fluid_body_contact),
          buoyancy_force(diffusion_body, thermal_expansion_coeff, (up_temperature+down_temperature)/2.0),
          get_fluid_advection_time_step(diffusion_body, U_f),
          get_fluid_time_step(diffusion_body),

          target_fluid_particles(diffusion_body_contact_all_Dirichlet),
          target_up_solid_particles(up_wall_Dirichlet, diffusion_body),
          target_down_solid_particles(down_wall_Dirichlet, diffusion_body),
          calculate_phi_gradient(diffusion_body_inner, diffusion_body_contact_all_Dirichlet),
          local_nusselt_number(diffusion_body, H / (down_temperature - up_temperature)),
          up_wall_local_nusselt_number(up_Dirichlet_contact, H / (down_temperature - up_temperature)),
          down_wall_local_nusselt_number(down_Dirichlet_contact, H / (down_temperature - up_temperature)),

          particle_sorting(diffusion_body),
          write_states(sph_system),
          restart_io(sph_system),
          write_up_PhiFluxSum(diffusion_body, "PhiTransferFromUpDirichletWallBoundaryFlux"),
          write_down_PhiFluxSum(diffusion_body, "PhiTransferFromDownDirichletWallBoundaryFlux"),
          left_diffusion_domain(diffusion_body, makeShared<MultiPolygonShape>(createLeftDiffusionDomain(), "LeftDiffusionDomain")),
          write_left_PhiFluxSum(left_diffusion_domain, "PhiTransferFromDownDirichletWallBoundaryFlux"),
          middle_diffusion_domain(diffusion_body, makeShared<MultiPolygonShape>(createMiddleDiffusionDomain(), "MiddleDiffusionDomain")),
          write_middle_PhiFluxSum(middle_diffusion_domain, "PhiTransferFromDownDirichletWallBoundaryFlux"),
          right_diffusion_domain(diffusion_body, makeShared<MultiPolygonShape>(createRightDiffusionDomain(), "RightDiffusionDomain")),
          write_right_PhiFluxSum(right_diffusion_domain, "PhiTransferFromDownDirichletWallBoundaryFlux"),

          write_recorded_fluid_vel("Velocity", observer_diffusion_body_contact),
          write_global_kinetic_energy(diffusion_body)
    {
        physical_time = 0.0;
        //----------------------------------------------------------------------
        //	Prepare the simulation with cell linked list, configuration
        //	and case specified initial condition if necessary.
        //----------------------------------------------------------------------
        sph_system.initializeSystemCellLinkedLists();
        sph_system.initializeSystemConfigurations();
        setup_diffusion_initial_condition.exec();
        setup_up_Dirichlet_initial_condition.exec();
        setup_down_Dirichlet_initial_condition.exec();
        setup_boundary_condition_Neumann.exec();
        diffusion_body_normal_direction.exec();
        entire_wall_normal_direction.exec();
        up_Dirichlet_wall_normal_direction.exec();
        down_Dirichlet_wall_normal_direction.exec();
        Neumann_wall_normal_direction.exec();
        //----------------------------------------------------------------------
        //	First output before the main loop.
        //----------------------------------------------------------------------
        write_states.writeToFile(0);
        write_up_PhiFluxSum.writeToFile(0);
        write_down_PhiFluxSum.writeToFile(0);
        write_recorded_fluid_vel.writeToFile(0);
        write_global_kinetic_energy.writeToFile(0);
        write_left_PhiFluxSum.writeToFile(0);
        write_middle_PhiFluxSum.writeToFile(0);
        write_right_PhiFluxSum.writeToFile(0);
    }

    virtual ~SphNaturalConvection(){};
    //----------------------------------------------------------------------
    //	    For ctest.
    //----------------------------------------------------------------------
    int cmakeTest()
    {
        return 1;
    }
    //----------------------------------------------------------------------
    //  Get heat flux.
    //----------------------------------------------------------------------
    Real getPhiFluxSum()
    {
        return write_down_PhiFluxSum.getReducedQuantity();
    };

    Real getLeftPhiFlux()
    {
        return write_left_PhiFluxSum.getReducedQuantity();
    };

    Real getMiddlePhiFlux()
    {
        return write_middle_PhiFluxSum.getReducedQuantity();
    };

    Real getRightPhiFlux()
    {
        return write_right_PhiFluxSum.getReducedQuantity();
    };
    //----------------------------------------------------------------------
    //  Get kinetic energy.
    //----------------------------------------------------------------------
    Real getLocalVelocity(int number, int direction)
    {
        return write_recorded_fluid_vel.getObservedQuantity()[number][direction];
    };
    
    Real getGlobalKineticEnergy()
    {
        return write_global_kinetic_energy.getReducedQuantity();
    };
    //----------------------------------------------------------------------
    //	    Main loop of time stepping starts here. && For changing damping coefficient.
    //----------------------------------------------------------------------
    void runCase(Real pause_time_from_python)
    {
        while (physical_time < pause_time_from_python)
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
                diffusion_body_contact_all_Dirichlet.updateConfiguration();
                diffusion_body_contact_up_Dirichlet.updateConfiguration();
                diffusion_body_contact_down_Dirichlet.updateConfiguration();
                diffusion_body_contact_Neumann.updateConfiguration();
                fluid_body_complex.updateConfiguration();
                up_Dirichlet_contact.updateConfiguration();
                down_Dirichlet_contact.updateConfiguration();
                observer_diffusion_body_contact.updateConfiguration();

                target_fluid_particles.exec();
                target_up_solid_particles.exec();
                target_down_solid_particles.exec();
            }

            TickCount t2 = TickCount::now();
            calculate_phi_gradient.exec();
            local_nusselt_number.exec();
            up_wall_local_nusselt_number.exec();
            down_wall_local_nusselt_number.exec();

            write_states.writeToFile();
            write_up_PhiFluxSum.writeToFile(number_of_iterations);
            write_down_PhiFluxSum.writeToFile(number_of_iterations);
            write_recorded_fluid_vel.writeToFile(number_of_iterations);
            write_global_kinetic_energy.writeToFile(number_of_iterations);

            write_left_PhiFluxSum.writeToFile(number_of_iterations);
            write_middle_PhiFluxSum.writeToFile(number_of_iterations);
            write_right_PhiFluxSum.writeToFile(number_of_iterations);

            TickCount t3 = TickCount::now();
            interval += t3 - t2;
        }
        TickCount t4 = TickCount::now();
        TimeInterval tt;
        tt = t4 - t1 - interval;
    };
};

//----------------------------------------------------------------------
//	Use pybind11 to expose.
//----------------------------------------------------------------------
PYBIND11_MODULE(zcx_2d_natural_convection_RL_python, m)
{
    py::class_<SphNaturalConvection>(m, "natural_convection_from_sph_cpp")
        .def(py::init<const int &, const int &>())
        .def("cmake_test", &SphNaturalConvection::cmakeTest)
        .def("get_global_heat_flux", &SphNaturalConvection::getPhiFluxSum)
        .def("get_local_heat_flux_left", &SphNaturalConvection::getLeftPhiFlux)
        .def("get_local_heat_flux_middle", &SphNaturalConvection::getMiddlePhiFlux)
        .def("get_local_heat_flux_right", &SphNaturalConvection::getRightPhiFlux)
        .def("get_local_velocity", &SphNaturalConvection::getLocalVelocity)
        .def("get_global_kinetic_energy", &SphNaturalConvection::getGlobalKineticEnergy)
        .def("run_case", &SphNaturalConvection::runCase);
}
