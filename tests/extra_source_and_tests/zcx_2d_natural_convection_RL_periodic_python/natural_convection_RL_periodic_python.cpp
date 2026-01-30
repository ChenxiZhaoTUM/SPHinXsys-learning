/**
 * @file    natural_convection_RL_periodic_python.cpp
 * @brief    
 * @author   
 */
#include "natural_convection_RL_periodic_python.h"
#include "custom_io_environment.h"
#include "custom_io_observation.h"
#include "sphinxsys.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace SPH;
namespace py = pybind11;

class SphBasicSystemSetting : public SphBasicGeometrySetting
{
  protected:
    BoundingBox system_domain_bounds;
    SPHSystem sph_system;
    std::unique_ptr<CustomIOEnvironment> custom_io_environment;

  public:
    SphBasicSystemSetting(int parallel_env, int episode_env, int set_restart_step = 0)
        : system_domain_bounds(Vecd(-BW, - H/2 -BW), Vecd(L + BW, H/2 + BW)),
          sph_system(system_domain_bounds, particle_spacing_ref)
    {
        sph_system.setRestartStep(set_restart_step);
        custom_io_environment = std::make_unique<CustomIOEnvironment>(sph_system, (set_restart_step == 0), parallel_env, episode_env);
    }
};

class SphBodyReloadEnvironment : public SphBasicSystemSetting
{
  protected:
    FluidBody diffusion_body;
    SolidBody wall_boundary, up_wall_Dirichlet, down_wall_Dirichlet;
    ObserverBody diffusion_observer;

  public:
    SphBodyReloadEnvironment(int parallel_env, int episode_env, int set_restart_step = 0)
        : SphBasicSystemSetting(parallel_env, episode_env, set_restart_step),
          diffusion_body(sph_system, makeShared<DiffusionBody>("DiffusionBody")),
          wall_boundary(sph_system, makeShared<WallBoundary>("EntireWallBoundary")),
          up_wall_Dirichlet(sph_system, makeShared<UpDirichletWallBoundary>("UpDirichletWallBoundary")),
          down_wall_Dirichlet(sph_system, makeShared<DownDirichletWallBoundary>("DownDirichletWallBoundary")),
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

        diffusion_observer.generateParticles<ObserverParticles>(createObservationPoints());
    }

    FluidBody &getFluidBody()
    { 
        return diffusion_body; 
    }

    SolidBody &getDownWallBody()
    { 
        return down_wall_Dirichlet; 
    }
};

class SphNaturalConvection : public SphBodyReloadEnvironment
{
    using DiffusionBodyRelaxation = DiffusionBodyRelaxationComplex<
        IsotropicDiffusion, KernelGradientInner, KernelGradientContact, Dirichlet, Dirichlet>;

  protected:
    SPHSystem &sph_system_;
    InnerRelation diffusion_body_inner;
    ContactRelation up_Dirichlet_contact, down_Dirichlet_contact, diffusion_body_contact_all_Dirichlet, diffusion_body_contact_up_Dirichlet,
        diffusion_body_contact_down_Dirichlet, fluid_body_contact, observer_diffusion_body_contact;
    ComplexRelation fluid_body_complex;
    //----------------------------------------------------------------------
    //	    Define all numerical methods which are used in this case.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> diffusion_body_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> entire_wall_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> up_Dirichlet_wall_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> down_Dirichlet_wall_normal_direction;

    DiffusionBodyRelaxation temperature_relaxation;
    GetDiffusionTimeStepSize get_thermal_time_step;
    SimpleDynamics<DiffusionInitialCondition> setup_diffusion_initial_condition;
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_up_Dirichlet_initial_condition;
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_down_Dirichlet_initial_condition;

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> kernel_correction_complex;
    Dynamics1Level<fluid_dynamics::Integration1stHalfCorrectionWithWallRiemann> pressure_relaxation;
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation;
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplex> update_density_by_summation;
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_force;
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionCorrectedComplex<AllParticles>> transport_velocity_correction;
    SimpleDynamics<fluid_dynamics::BuoyancyForce> buoyancy_force;
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step;
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step;
    PeriodicAlongAxis periodic_along_x;
    PeriodicConditionUsingCellLinkedList periodic_condition;

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
    ObservedQuantityRecording<Vecd> write_recorded_fluid_vel;
    ObservedQuantityRecording<Real> write_recorded_fluid_temperature;
    ExtendedReducedQuantityRecording<TotalKineticEnergy> write_global_kinetic_energy;

    ExtendedReducedQuantityRecording<solid_dynamics::AveragedWallNu<>> write_global_average_Nu;

    //----------------------------------------------------------------------
    //	    Basic control parameters for time stepping.
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int ite = 0;
    Real output_interval = 1.0;
    int number_of_iterations;
    int screen_output_interval = 100;
    int restart_output_interval = screen_output_interval * 10;

    /** statistics for computing time. */
    TickCount t1 = TickCount::now();
    TimeInterval interval;

  public:
    explicit SphNaturalConvection(int parallel_env, int episode_env, int set_restart_step = 0)
        : SphBodyReloadEnvironment(parallel_env, episode_env, set_restart_step),
          sph_system_(sph_system),
          diffusion_body_inner(diffusion_body),
          up_Dirichlet_contact(up_wall_Dirichlet, {&diffusion_body}),
          down_Dirichlet_contact(down_wall_Dirichlet, {&diffusion_body}),
          diffusion_body_contact_all_Dirichlet(diffusion_body, {&up_wall_Dirichlet, &down_wall_Dirichlet}),
          diffusion_body_contact_up_Dirichlet(diffusion_body, {&up_wall_Dirichlet}),
          diffusion_body_contact_down_Dirichlet(diffusion_body, {&down_wall_Dirichlet}),
          fluid_body_contact(diffusion_body, {&wall_boundary}),
          fluid_body_complex(diffusion_body_inner, fluid_body_contact),
          observer_diffusion_body_contact(diffusion_observer, {&diffusion_body}),

          diffusion_body_normal_direction(diffusion_body),
          entire_wall_normal_direction(wall_boundary),
          up_Dirichlet_wall_normal_direction(up_wall_Dirichlet),
          down_Dirichlet_wall_normal_direction(down_wall_Dirichlet),
          temperature_relaxation(
          diffusion_body_inner, diffusion_body_contact_up_Dirichlet, diffusion_body_contact_down_Dirichlet),
          get_thermal_time_step(diffusion_body),
          setup_diffusion_initial_condition(diffusion_body),
          setup_up_Dirichlet_initial_condition(up_wall_Dirichlet),
          setup_down_Dirichlet_initial_condition(down_wall_Dirichlet),

          kernel_correction_complex(InteractArgs(diffusion_body_inner, 0.1), fluid_body_contact),
          pressure_relaxation(diffusion_body_inner, fluid_body_contact),
          density_relaxation(diffusion_body_inner, fluid_body_contact),
          update_density_by_summation(diffusion_body_inner, fluid_body_contact),
          viscous_force(diffusion_body_inner, fluid_body_contact),
          transport_velocity_correction(diffusion_body_inner, fluid_body_contact),
          buoyancy_force(diffusion_body, thermal_expansion_coeff, (up_temperature+down_temperature)/2.0),
          get_fluid_advection_time_step(diffusion_body, U_f),
          get_fluid_time_step(diffusion_body),
          periodic_along_x(diffusion_body.getSPHBodyBounds(), xAxis),
          periodic_condition(diffusion_body, periodic_along_x),

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
          write_recorded_fluid_vel("Velocity", observer_diffusion_body_contact),
          write_recorded_fluid_temperature("Phi", observer_diffusion_body_contact),
          write_global_kinetic_energy(diffusion_body),
          write_global_average_Nu(down_wall_Dirichlet, "WallLocalNusseltNumber")
    {
        physical_time = 0.0;
        
        //----------------------------------------------------------------------
        //	Prepare the simulation with cell linked list, configuration
        //	and case specified initial condition if necessary.
        //----------------------------------------------------------------------
        sph_system.initializeSystemCellLinkedLists();
        periodic_condition.update_cell_linked_list_.exec();
        sph_system.initializeSystemConfigurations();
        diffusion_body_normal_direction.exec();
        entire_wall_normal_direction.exec();
        up_Dirichlet_wall_normal_direction.exec();
        down_Dirichlet_wall_normal_direction.exec();
        
        number_of_iterations = sph_system.RestartStep();

        if (sph_system.RestartStep() == 0)
        {
            StdVec<Real> baseline_temps(n_seg, 2.0);
            SphBasicGeometrySetting::setDownWallSegmentTemperatures(baseline_temps);

            setup_diffusion_initial_condition.exec();
            setup_up_Dirichlet_initial_condition.exec();
            setup_down_Dirichlet_initial_condition.exec();
        }

        if (sph_system.RestartStep() != 0)
        {
            physical_time = restart_io.readRestartFiles(sph_system.RestartStep());
            diffusion_body.updateCellLinkedList();
            down_wall_Dirichlet.updateCellLinkedList();
            fluid_body_complex.updateConfiguration();
            up_Dirichlet_contact.updateConfiguration();
            down_Dirichlet_contact.updateConfiguration();
            diffusion_body_contact_all_Dirichlet.updateConfiguration();
            diffusion_body_contact_up_Dirichlet.updateConfiguration();
            diffusion_body_contact_down_Dirichlet.updateConfiguration();
            observer_diffusion_body_contact.updateConfiguration();
        }
        //----------------------------------------------------------------------
        //	First output before the main loop.
        //----------------------------------------------------------------------
        write_states.writeToFile();
        write_up_PhiFluxSum.writeToFile();
        write_down_PhiFluxSum.writeToFile();
        write_recorded_fluid_vel.writeToFile();
        write_recorded_fluid_temperature.writeToFile();
        write_global_kinetic_energy.writeToFile();
        write_global_average_Nu.writeToFile();
    }

    virtual ~SphNaturalConvection(){};
    //----------------------------------------------------------------------
    //	    For ctest.
    //----------------------------------------------------------------------
    int cmakeTest()
    {
        return 1;
    }

    int debugSmokeTest()
    {
        std::cout << "---- debugSmokeTest() begin ----\n";

        std::cout << "[0] physical_time = " << physical_time << "\n";

        // step 1: set some arbitrary temperatures
        StdVec<Real> temps_test;
        temps_test.push_back(2.2);
        temps_test.push_back(2.0);
        temps_test.push_back(1.8);
        temps_test.push_back(2.5);

        std::cout << "[1] calling set_segment_temperatures ...\n";
        setDownWallSegmentTemperatures(temps_test);

        // step 2: advance the simulation to t = physical_time + 10 (a short run)
        Real target_time = physical_time + 10;
        std::cout << "[2] runCase(" << target_time << ") ...\n";
        runCase(target_time);

        // step 3: read back some quantities
        Real q_flux_sum = getPhiFluxSum();
        Real q_flux_0   = getLocalPhiFlux(0);
        Real q_flux_1   = getLocalPhiFlux(1);
        Real q_flux_2   = getLocalPhiFlux(2);
        Real q_flux_3   = getLocalPhiFlux(3);

        Real ke_global  = getGlobalKineticEnergy();

        // also sample one probe velocity just to check indexing works
        Real vx0 = getLocalVelocity(0, 0);
        Real vy0 = getLocalVelocity(0, 1);

        std::cout << "[3] after runCase:\n";
        std::cout << "    physical_time = " << physical_time << "\n";
        std::cout << "    global_heat_flux = " << q_flux_sum << "\n";
        std::cout << "    local_flux[0,1,2,3] = "
                  << q_flux_0 << ", "
                  << q_flux_1 << ", "
                  << q_flux_2 << ", "
                  << q_flux_3 << "\n";
        std::cout << "    global_kinetic_energy = " << ke_global << "\n";
        std::cout << "    probe0 vel = (" << vx0 << ", " << vy0 << ")\n";

        std::cout << "---- debugSmokeTest() end ----\n";

        return 1;
    }
    //----------------------------------------------------------------------
    //  Get heat flux.
    //----------------------------------------------------------------------
    Real getPhiFluxSum()
    {
        return write_down_PhiFluxSum.getReducedQuantity();
    };

    Real getLocalPhiFlux(int i_seg)
    {
        // how many control segments we currently have
        size_t n = down_wall_segment_T.size();
        if (n == 0) return Real(0.0);
        if (i_seg < 0 || static_cast<size_t>(i_seg) >= n) return Real(0.0);

        // horizontal extent of this segment
        Real seg_len = L / Real(n);
        Real x0 = seg_len * Real(i_seg);
        Real x1 = x0 + seg_len;

        // build a vertical strip spanning full cavity height
        std::vector<Vecd> seg_poly;
        seg_poly.push_back(Vecd(x0, -H/2)); // bottom inner fluid boundary
        seg_poly.push_back(Vecd(x1, -H/2));
        seg_poly.push_back(Vecd(x1,  H/2));
        seg_poly.push_back(Vecd(x0,  H/2));
        seg_poly.push_back(Vecd(x0, -H/2));

        MultiPolygon seg_shape;
        seg_shape.addAPolygon(seg_poly, ShapeBooleanOps::add);

        // define a particle region over the diffusion_body
        BodyRegionByParticle seg_region(
            getFluidBody(),
            makeShared<MultiPolygonShape>(seg_shape, "seg_region_tmp")
        );

        // sum the same reduced quantity
        ExtendedReducedQuantityRecording<
            QuantitySummation<Real, BodyRegionByParticle>
        > seg_flux(
            seg_region,
            "PhiTransferFromDownDirichletWallBoundaryFlux"
        );

        return seg_flux.getReducedQuantity();
    }
    //----------------------------------------------------------------------
    //  Get kinetic energy.
    //----------------------------------------------------------------------
    Real getLocalVelocity(int number, int direction)
    {
        return write_recorded_fluid_vel.getObservedQuantity()[number][direction];
    };

    Real getLocalTemperature(int number)
    {
        return write_recorded_fluid_temperature.getObservedQuantity()[number];
    };
    
    Real getGlobalKineticEnergy()
    {
        return write_global_kinetic_energy.getReducedQuantity();
    };

    Real getGlobalNusseltNumber()
    {
        return write_global_average_Nu.getReducedQuantity();
    };

    Real getLocalNusselt(int i_seg)
    {
        // how many control segments we currently have
        size_t n = down_wall_segment_T.size();
        if (n == 0) return Real(0.0);
        if (i_seg < 0 || static_cast<size_t>(i_seg) >= n) return Real(0.0);

        // horizontal extent of this segment
        Real seg_len = L / Real(n);
        Real x0 = seg_len * Real(i_seg);
        Real x1 = x0 + seg_len;

        // build a vertical strip spanning full cavity height
        std::vector<Vecd> seg_poly;
        seg_poly.push_back(Vecd(x0, -H/2 - BW)); // bottom down wall boundary
        seg_poly.push_back(Vecd(x1, -H/2 - BW));
        seg_poly.push_back(Vecd(x1, -H/2));
        seg_poly.push_back(Vecd(x0, -H/2));
        seg_poly.push_back(Vecd(x0, -H/2 - BW));

        MultiPolygon seg_shape;
        seg_shape.addAPolygon(seg_poly, ShapeBooleanOps::add);

        // define a particle region over the diffusion_body
        BodyRegionByParticle seg_region(
            getDownWallBody(),
            makeShared<MultiPolygonShape>(seg_shape, "down_wall_seg_region_tmp")
        );

        // sum the same reduced quantity
        ExtendedReducedQuantityRecording<solid_dynamics::AveragedWallNu<BodyRegionByParticle>> seg_Nu(seg_region, "WallLocalNusseltNumber");

        return seg_Nu.getReducedQuantity();
    }
    //----------------------------------------------------------------------
    //  Set bottom-wall temperature layout.
    //----------------------------------------------------------------------
    void setDownWallSegmentTemperatures(const StdVec<Real> &Ts)
    {
        if (Ts.empty()) return;

        SphBasicGeometrySetting::setDownWallSegmentTemperatures(Ts);
        setup_down_Dirichlet_initial_condition.exec();
    }
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

                    if (sph_system.RestartStep() == 0 && number_of_iterations % restart_output_interval == 0)
                    { 
                        restart_io.writeToFile(number_of_iterations);
                    }

                    write_states.writeToFile();  // save memory of disk
                        
                }
                number_of_iterations++;

                periodic_condition.bounding_.exec();
                if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
                {
                    particle_sorting.exec();
                }
                diffusion_body.updateCellLinkedList();
                periodic_condition.update_cell_linked_list_.exec();
                diffusion_body_contact_all_Dirichlet.updateConfiguration();
                diffusion_body_contact_up_Dirichlet.updateConfiguration();
                diffusion_body_contact_down_Dirichlet.updateConfiguration();
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

            //write_states.writeToFile();
            write_up_PhiFluxSum.writeToFile(number_of_iterations);
            write_down_PhiFluxSum.writeToFile(number_of_iterations);
            write_recorded_fluid_vel.writeToFile(number_of_iterations);
            write_global_kinetic_energy.writeToFile(number_of_iterations);
            write_global_average_Nu.writeToFile(number_of_iterations);
            
            TickCount t3 = TickCount::now();
            interval += t3 - t2;
        }
        TickCount t4 = TickCount::now();
        if (sph_system.RestartStep() == 0)
        { 
            restart_io.writeToFile(number_of_iterations);
        }
        TimeInterval tt;
        tt = t4 - t1 - interval;
    };
};

//----------------------------------------------------------------------
//	Use pybind11 to expose.
//----------------------------------------------------------------------
PYBIND11_MODULE(zcx_2d_natural_convection_RL_periodic_python, m)
{
    py::class_<SphNaturalConvection>(m, "natural_convection_from_sph_cpp")
        .def(py::init<const int&, const int&, const int&>(), py::arg("parallel_env"), py::arg("episode_env"), py::arg("set_restart_step") = 0)
        .def("cmake_test", &SphNaturalConvection::cmakeTest)
        .def("set_segment_temperatures", &SphNaturalConvection::setDownWallSegmentTemperatures, py::arg("temps"))
        .def("get_global_heat_flux", &SphNaturalConvection::getPhiFluxSum)
        .def("get_local_phi_flux", &SphNaturalConvection::getLocalPhiFlux, py::arg("i_seg"))
        .def("get_local_velocity", &SphNaturalConvection::getLocalVelocity, py::arg("probe_index"), py::arg("component"))
        .def("get_local_temperature", &SphNaturalConvection::getLocalTemperature, py::arg("probe_index"))
        .def("get_global_kinetic_energy", &SphNaturalConvection::getGlobalKineticEnergy)
        .def("get_global_Nusselt_number", &SphNaturalConvection::getGlobalNusseltNumber)
        .def("get_local_Nusselt", &SphNaturalConvection::getLocalNusselt, py::arg("i_seg"))
        .def("run_case", &SphNaturalConvection::runCase, py::arg("target_time"))
        .def("debug_smoke_test", &SphNaturalConvection::debugSmokeTest);
}
