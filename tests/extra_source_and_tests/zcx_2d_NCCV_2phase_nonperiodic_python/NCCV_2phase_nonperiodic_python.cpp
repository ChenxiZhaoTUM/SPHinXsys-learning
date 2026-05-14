/**
 * @file    NCCV_2phase_nonperiodic_python.cpp
 * @brief
 * @author
 */
#include "NCCV_2phase_nonperiodic_python.h"
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
        : system_domain_bounds(Vecd(-BW, -H / 2 - BW), Vecd(L + BW, H / 2 + BW)),
          sph_system(system_domain_bounds, particle_spacing_ref)
    {
        sph_system.setRestartStep(set_restart_step);
        custom_io_environment = std::make_unique<CustomIOEnvironment>(sph_system, (set_restart_step == 0), parallel_env, episode_env);
    }
};

class SphBodyReloadEnvironment : public SphBasicSystemSetting
{
  protected:
    FluidBody PhaseOne_diffusion_body, PhaseTwo_diffusion_body;
    SolidBody wall_boundary, up_Dirichlet, down_Dirichlet, wall_Neumann;
    ObserverBody diffusion_observer;

  public:
    SphBodyReloadEnvironment(int parallel_env, int episode_env, int set_restart_step = 0)
        : SphBasicSystemSetting(parallel_env, episode_env, set_restart_step),
          PhaseOne_diffusion_body(sph_system, makeShared<PhaseOneDiffusionBody>("PhaseOneDiffusionBody")),
          PhaseTwo_diffusion_body(sph_system, makeShared<PhaseTwoDiffusionBody>("PhaseTwoDiffusionBody")),
          wall_boundary(sph_system, makeShared<WallBoundary>("EntireWallBoundary")),
          up_Dirichlet(sph_system, makeShared<UpDirichlet>("UpDirichlet")),
          down_Dirichlet(sph_system, makeShared<DownDirichlet>("DownDirichlet")),
          wall_Neumann(sph_system, makeShared<NeumannBoundary>("NeumannBoundary")),  
          diffusion_observer(sph_system, "DiffusionObserver")
    {
        PhaseOne_diffusion_body.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicThermalDiffusion>(
            ConstructArgs(rho0_f_one, c_f), mu_f_one, ConstructArgs(diffusion_species_name, k_one, rho0_f_one, C_p_one));
        PhaseOne_diffusion_body.generateParticles<BaseParticles, Lattice>();

        PhaseTwo_diffusion_body.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicThermalDiffusion>(
            ConstructArgs(rho0_f_two, c_f), mu_f_two, ConstructArgs(diffusion_species_name, k_two, rho0_f_two, C_p_two));
        PhaseTwo_diffusion_body.generateParticles<BaseParticles, Lattice>();

        wall_boundary.defineMaterial<Solid>();
        wall_boundary.generateParticles<BaseParticles, Lattice>();

        up_Dirichlet.defineMaterial<Solid>();
        up_Dirichlet.generateParticles<BaseParticles, Lattice>();

        down_Dirichlet.defineMaterial<Solid>();
        down_Dirichlet.generateParticles<BaseParticles, Lattice>();

        wall_Neumann.defineMaterial<Solid>();
        wall_Neumann.generateParticles<BaseParticles, Lattice>();

        diffusion_observer.generateParticles<ObserverParticles>(createObservationPoints());
    }

    FluidBody &getPhaseOneBody()
    {
        return PhaseOne_diffusion_body;
    }

    FluidBody &getPhaseTwoBody()
    {
        return PhaseTwo_diffusion_body;
    }

    SolidBody &getDownDirichletBody()
    {
        return down_Dirichlet;
    }
};

class SphNaturalConvection : public SphBodyReloadEnvironment
{
    using MultiPhaseDiffusionBodyRelaxation = MultiPhaseDiffusionBodyRelaxationComplex<
        IsotropicThermalDiffusion, KernelGradientInner, KernelGradientContact, Dirichlet, Dirichlet, Neumann>;

  protected:
    SPHSystem &sph_system_;
    InnerRelation phase_1_inner, phase_2_inner;
    ContactRelation phase_1_contact_up_Dirichlet, phase_1_contact_down_Dirichlet, phase_1_contact_Neumann, phase_1_contact_wall_boundary, phase_1_contact_two, phase_1_contacts,
        phase_2_contact_up_Dirichlet, phase_2_contact_down_Dirichlet, phase_2_contact_Neumann, phase_2_contact_wall_boundary, phase_2_contact_one, phase_2_contacts,
        up_Dirichlet_contacts, down_Dirichlet_contacts, observer_diffusion_body_contact;
    ComplexRelation phase_1_complex, phase_2_complex;
    //----------------------------------------------------------------------
    //	    Define all numerical methods which are used in this case.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> phase_1_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> phase_2_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> entire_wall_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> up_Dirichlet_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> down_Dirichlet_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> Neumann_wall_normal_direction;

    // thermal dynamics
    MultiPhaseDiffusionBodyRelaxation phase_1_temperature_relaxation;
    GetDiffusionTimeStepSize phase_1_thermal_time_step;
    SimpleDynamics<DiffusionInitialCondition> phase_1_initial_condition;

    MultiPhaseDiffusionBodyRelaxation phase_2_temperature_relaxation;
    GetDiffusionTimeStepSize phase_2_thermal_time_step;
    SimpleDynamics<DiffusionInitialCondition> phase_2_initial_condition;

    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_up_Dirichlet_initial_condition;
    SimpleDynamics<DirichletWallBoundaryInitialCondition> setup_down_Dirichlet_initial_condition;
    SimpleDynamics<NeumannWallBoundaryInitialCondition> setup_Neumann_initial_condition;

    // fluid dynamics
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> phase_1_kernel_correction_complex;
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> phase_2_kernel_correction_complex;
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfCorrectionWithWallRiemann> phase_1_pressure_relaxation;
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann> phase_1_density_relaxation;
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfCorrectionWithWallRiemann> phase_2_pressure_relaxation;
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann> phase_2_density_relaxation;

    InteractionWithUpdate<fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>> phase_1_update_density_by_summation;
    InteractionWithUpdate<fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>> phase_2_update_density_by_summation;
    InteractionWithUpdate<fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>> phase_1_transport_correction;
    InteractionWithUpdate<fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>> phase_2_transport_correction;
    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall> phase_1_viscous_force;
    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall> phase_2_viscous_force;

    // extract flux
    SimpleDynamics<fluid_dynamics::BuoyancyForce> phase_1_buoyancy_force;
    SimpleDynamics<fluid_dynamics::BuoyancyForce> phase_2_buoyancy_force;
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> phase_1_advection_time_step;
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> phase_2_advection_time_step;
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> phase_1_acoustic_time_step;
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> phase_2_acoustic_time_step;

    InteractionWithUpdate<fluid_dynamics::TargetFluidParticles> phase_1_target_fluid_particles;
    InteractionWithUpdate<fluid_dynamics::TargetFluidParticles> phase_2_target_fluid_particles;
    SimpleDynamics<solid_dynamics::FirstLayerFromFluids> target_up_solid_particles;
    SimpleDynamics<solid_dynamics::FirstLayerFromFluids> target_down_solid_particles;

    InteractionWithUpdate<fluid_dynamics::MultiPhasePhiGradientWithWalls<LinearGradientCorrection>> calculate_phase_1_phi_gradient;
    InteractionWithUpdate<fluid_dynamics::MultiPhasePhiGradientWithWalls<LinearGradientCorrection>> calculate_phase_2_phi_gradient;
    SimpleDynamics<fluid_dynamics::LocalNusseltNum> phase_1_local_nusselt_number;
    SimpleDynamics<fluid_dynamics::LocalNusseltNum> phase_2_local_nusselt_number;

    InteractionDynamics<solid_dynamics::ProjectionForNu> up_wall_local_nusselt_number;
    InteractionDynamics<solid_dynamics::ProjectionForNu> down_wall_local_nusselt_number;
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    ParticleSorting<ParallelPolicy> phase_1_particle_sorting;
    ParticleSorting<ParallelPolicy> phase_2_particle_sorting;
    BodyStatesRecordingToVtp write_states;
    RestartIO restart_io;

    ExtendedReducedQuantityRecording<QuantitySummation<Real>> write_phase_1_PhiFluxSum;
    ExtendedReducedQuantityRecording<QuantitySummation<Real>> write_phase_2_PhiFluxSum;

    ObservedQuantityRecording<Vecd> write_observed_fluid_vel;
    ObservedQuantityRecording<Real> write_observed_fluid_temperature;

    ExtendedReducedQuantityRecording<TotalKineticEnergy> write_phase_1_GlobalKineticEnergy;
    ExtendedReducedQuantityRecording<TotalKineticEnergy> write_phase_2_GlobalKineticEnergy;
    ExtendedReducedQuantityRecording<solid_dynamics::AveragedWallNu<SPHBody>> write_GlobalAveragedNu;
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
          phase_1_inner(PhaseOne_diffusion_body),
          phase_2_inner(PhaseTwo_diffusion_body),
          phase_1_contact_up_Dirichlet(PhaseOne_diffusion_body, {&up_Dirichlet}),
          phase_1_contact_down_Dirichlet(PhaseOne_diffusion_body, {&down_Dirichlet}),
          phase_1_contact_Neumann(PhaseOne_diffusion_body, {&wall_Neumann}),
          phase_1_contact_wall_boundary(PhaseOne_diffusion_body, {&wall_boundary}),
          phase_1_contact_two(PhaseOne_diffusion_body, {&PhaseTwo_diffusion_body}),
          phase_1_contacts(PhaseOne_diffusion_body, {&PhaseTwo_diffusion_body, &wall_boundary}),
          phase_2_contact_up_Dirichlet(PhaseTwo_diffusion_body, {&up_Dirichlet}),
          phase_2_contact_down_Dirichlet(PhaseTwo_diffusion_body, {&down_Dirichlet}),
          phase_2_contact_Neumann(PhaseTwo_diffusion_body, {&wall_Neumann}),
          phase_2_contact_wall_boundary(PhaseTwo_diffusion_body, {&wall_boundary}),
          phase_2_contact_one(PhaseTwo_diffusion_body, {&PhaseOne_diffusion_body}),
          phase_2_contacts(PhaseTwo_diffusion_body, {&PhaseOne_diffusion_body, &wall_boundary}),
          up_Dirichlet_contacts(up_Dirichlet, {&PhaseOne_diffusion_body, &PhaseTwo_diffusion_body}),
          down_Dirichlet_contacts(down_Dirichlet, {&PhaseOne_diffusion_body, &PhaseTwo_diffusion_body}),
          phase_1_complex(phase_1_inner, {&phase_1_contact_up_Dirichlet, &phase_1_contact_down_Dirichlet, &phase_1_contact_Neumann, &phase_1_contact_wall_boundary, &phase_1_contact_two, &phase_1_contacts}),
          phase_2_complex(phase_2_inner, {&phase_2_contact_up_Dirichlet, &phase_2_contact_down_Dirichlet, &phase_2_contact_Neumann, &phase_2_contact_wall_boundary, &phase_2_contact_one, &phase_2_contacts}),
          observer_diffusion_body_contact(diffusion_observer, {&PhaseOne_diffusion_body, &PhaseTwo_diffusion_body}),

          // normal direction
          phase_1_normal_direction(PhaseOne_diffusion_body),
          phase_2_normal_direction(PhaseTwo_diffusion_body),
          entire_wall_normal_direction(wall_boundary),
          up_Dirichlet_normal_direction(up_Dirichlet),
          down_Dirichlet_normal_direction(down_Dirichlet),
          Neumann_wall_normal_direction(wall_Neumann),

          // thermal dynamics
          phase_1_temperature_relaxation(phase_1_inner, phase_1_contact_two, phase_1_contact_up_Dirichlet, phase_1_contact_down_Dirichlet, phase_1_contact_Neumann),
          phase_1_thermal_time_step(PhaseOne_diffusion_body),
          phase_1_initial_condition(PhaseOne_diffusion_body),

          phase_2_temperature_relaxation(phase_2_inner, phase_2_contact_one, phase_2_contact_up_Dirichlet, phase_2_contact_down_Dirichlet, phase_2_contact_Neumann),
          phase_2_thermal_time_step(PhaseTwo_diffusion_body),
          phase_2_initial_condition(PhaseTwo_diffusion_body),

          setup_up_Dirichlet_initial_condition(up_Dirichlet),
          setup_down_Dirichlet_initial_condition(down_Dirichlet),
          setup_Neumann_initial_condition(wall_Neumann),

          // fluid dynamics
          phase_1_kernel_correction_complex(InteractArgs(phase_1_inner, 0.5), phase_1_contacts),
          phase_2_kernel_correction_complex(InteractArgs(phase_2_inner, 0.5), phase_2_contacts),
          phase_1_pressure_relaxation(phase_1_inner, phase_1_contact_two, phase_1_contact_wall_boundary),
          phase_1_density_relaxation(phase_1_inner, phase_1_contact_two, phase_1_contact_wall_boundary),
          phase_2_pressure_relaxation(phase_2_inner, phase_2_contact_one, phase_2_contact_wall_boundary),
          phase_2_density_relaxation(phase_2_inner, phase_2_contact_one, phase_2_contact_wall_boundary),
          phase_1_update_density_by_summation(phase_1_inner, phase_1_contact_two, phase_1_contact_wall_boundary),
          phase_2_update_density_by_summation(phase_2_inner, phase_2_contact_one, phase_2_contact_wall_boundary),
          phase_1_transport_correction(phase_1_inner, phase_1_contact_two, phase_1_contact_wall_boundary),
          phase_2_transport_correction(phase_2_inner, phase_2_contact_one, phase_2_contact_wall_boundary),
          phase_1_viscous_force(phase_1_inner, phase_1_contact_two, phase_1_contact_wall_boundary),
          phase_2_viscous_force(phase_2_inner, phase_2_contact_one, phase_2_contact_wall_boundary),

          // extract flux
          phase_1_buoyancy_force(PhaseOne_diffusion_body, thermal_expansion_one, (up_temperature + down_temperature) / 2.0),
          phase_2_buoyancy_force(PhaseTwo_diffusion_body, thermal_expansion_two, (up_temperature + down_temperature) / 2.0),
          phase_1_advection_time_step(PhaseOne_diffusion_body, U_f),
          phase_2_advection_time_step(PhaseTwo_diffusion_body, U_f),
          phase_1_acoustic_time_step(PhaseOne_diffusion_body),
          phase_2_acoustic_time_step(PhaseTwo_diffusion_body),

          phase_1_target_fluid_particles(phase_1_contact_wall_boundary),
          phase_2_target_fluid_particles(phase_2_contact_wall_boundary),
          target_up_solid_particles(up_Dirichlet, PhaseOne_diffusion_body, PhaseTwo_diffusion_body),
          target_down_solid_particles(down_Dirichlet, PhaseOne_diffusion_body, PhaseTwo_diffusion_body),

          calculate_phase_1_phi_gradient(phase_1_inner, phase_1_contact_two, phase_1_contact_up_Dirichlet, phase_1_contact_down_Dirichlet),
          calculate_phase_2_phi_gradient(phase_2_inner, phase_2_contact_one, phase_2_contact_up_Dirichlet, phase_2_contact_down_Dirichlet),
          phase_1_local_nusselt_number(PhaseOne_diffusion_body, H / (down_temperature - up_temperature)),
          phase_2_local_nusselt_number(PhaseTwo_diffusion_body, H / (down_temperature - up_temperature)),

          up_wall_local_nusselt_number(up_Dirichlet_contacts, H / (down_temperature - up_temperature)),
          down_wall_local_nusselt_number(down_Dirichlet_contacts, H / (down_temperature - up_temperature)),

          phase_1_particle_sorting(PhaseOne_diffusion_body),
          phase_2_particle_sorting(PhaseTwo_diffusion_body),
          write_states(sph_system),
          restart_io(sph_system),
          write_phase_1_PhiFluxSum(PhaseOne_diffusion_body, "PhiTransferFromDownDirichletFlux"),
          write_phase_2_PhiFluxSum(PhaseTwo_diffusion_body, "PhiTransferFromDownDirichletFlux"),
          write_observed_fluid_vel("Velocity", observer_diffusion_body_contact),
          write_observed_fluid_temperature("Phi", observer_diffusion_body_contact),
          write_phase_1_GlobalKineticEnergy(PhaseOne_diffusion_body),
          write_phase_2_GlobalKineticEnergy(PhaseTwo_diffusion_body),
          write_GlobalAveragedNu(down_Dirichlet, "WallLocalNusseltNumber")
    {
        physical_time = 0.0;

        //----------------------------------------------------------------------
        //	Prepare the simulation with cell linked list, configuration
        //	and case specified initial condition if necessary.
        //----------------------------------------------------------------------
        sph_system.initializeSystemCellLinkedLists();
        sph_system.initializeSystemConfigurations();
        phase_1_normal_direction.exec();
        phase_2_normal_direction.exec();
        entire_wall_normal_direction.exec();
        up_Dirichlet_normal_direction.exec();
        down_Dirichlet_normal_direction.exec();
        Neumann_wall_normal_direction.exec();

        number_of_iterations = sph_system.RestartStep();

        if (sph_system.RestartStep() == 0)
        {
            StdVec<Real> baseline_temps(n_seg, 2.0);
            SphBasicGeometrySetting::setDownWallSegmentTemperatures(baseline_temps);

            phase_1_initial_condition.exec();
            phase_2_initial_condition.exec();
            setup_up_Dirichlet_initial_condition.exec();
            setup_down_Dirichlet_initial_condition.exec();
            setup_Neumann_initial_condition.exec();
        }

        if (sph_system.RestartStep() != 0)
        {
            physical_time = restart_io.readRestartFiles(sph_system.RestartStep());
            PhaseOne_diffusion_body.updateCellLinkedList();
            PhaseTwo_diffusion_body.updateCellLinkedList();
            down_Dirichlet.updateCellLinkedList();
            phase_1_complex.updateConfiguration();
            phase_2_complex.updateConfiguration();
            up_Dirichlet_contacts.updateConfiguration();
            down_Dirichlet_contacts.updateConfiguration();
            observer_diffusion_body_contact.updateConfiguration();
        }
        //----------------------------------------------------------------------
        //	First output before the main loop.
        //----------------------------------------------------------------------
        write_states.writeToFile();
        write_phase_1_PhiFluxSum.writeToFile();
        write_phase_2_PhiFluxSum.writeToFile();
        write_observed_fluid_vel.writeToFile();
        write_observed_fluid_temperature.writeToFile();
        write_phase_1_GlobalKineticEnergy.writeToFile();
        write_phase_2_GlobalKineticEnergy.writeToFile();
        write_GlobalAveragedNu.writeToFile();
    }

    virtual ~SphNaturalConvection() {};
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
        Real q_flux_0 = getLocalPhiFlux(0);
        Real q_flux_1 = getLocalPhiFlux(1);
        Real q_flux_2 = getLocalPhiFlux(2);
        Real q_flux_3 = getLocalPhiFlux(3);

        Real ke_global = getGlobalKineticEnergy();

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
        return write_phase_1_PhiFluxSum.getReducedQuantity() + write_phase_2_PhiFluxSum.getReducedQuantity();
    };

    Real getLocalPhiFlux(int i_seg)
    {
        // how many control segments we currently have
        size_t n = down_wall_segment_T.size();
        if (n == 0)
            return Real(0.0);
        if (i_seg < 0 || static_cast<size_t>(i_seg) >= n)
            return Real(0.0);

        // horizontal extent of this segment
        Real seg_len = L / Real(n);
        Real x0 = seg_len * Real(i_seg);
        Real x1 = x0 + seg_len;

        // build a vertical strip spanning full cavity height
        std::vector<Vecd> seg_poly;
        seg_poly.push_back(Vecd(x0, -H / 2)); // bottom inner fluid boundary
        seg_poly.push_back(Vecd(x1, -H / 2));
        seg_poly.push_back(Vecd(x1, H / 2));
        seg_poly.push_back(Vecd(x0, H / 2));
        seg_poly.push_back(Vecd(x0, -H / 2));

        MultiPolygon seg_shape;
        seg_shape.addAPolygon(seg_poly, ShapeBooleanOps::add);

        // define a particle region over the diffusion_body
        BodyRegionByParticle phase_1_seg_region(getPhaseOneBody(), makeShared<MultiPolygonShape>(seg_shape, "phase_1_seg_region_tmp"));
        BodyRegionByParticle phase_2_seg_region(getPhaseTwoBody(), makeShared<MultiPolygonShape>(seg_shape, "phase_2_seg_region_tmp"));

        // sum the same reduced quantity
        ExtendedReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>>
            phase_1_seg_flux(phase_1_seg_region, "PhiTransferFromDownDirichletFlux");
        ExtendedReducedQuantityRecording<QuantitySummation<Real, BodyRegionByParticle>>
            phase_2_seg_flux(phase_2_seg_region, "PhiTransferFromDownDirichletFlux");

        return phase_1_seg_flux.getReducedQuantity() + phase_2_seg_flux.getReducedQuantity();
    }
    //----------------------------------------------------------------------
    //  Get kinetic energy.
    //----------------------------------------------------------------------
    Real getLocalVelocity(int number, int direction)
    {
        return write_observed_fluid_vel.getObservedQuantity()[number][direction];
    };

    Real getLocalTemperature(int number)
    {
        return write_observed_fluid_temperature.getObservedQuantity()[number];
    };

    Real getGlobalKineticEnergy()
    {
        return write_phase_1_GlobalKineticEnergy.getReducedQuantity() + write_phase_2_GlobalKineticEnergy.getReducedQuantity();
    };

    Real getGlobalNusseltNumber()
    {
        return write_GlobalAveragedNu.getReducedQuantity();
    };

    Real getLocalNusselt(int i_seg)
    {
        // how many control segments we currently have
        size_t n = down_wall_segment_T.size();
        if (n == 0)
            return Real(0.0);
        if (i_seg < 0 || static_cast<size_t>(i_seg) >= n)
            return Real(0.0);

        // horizontal extent of this segment
        Real seg_len = L / Real(n);
        Real x0 = seg_len * Real(i_seg);
        Real x1 = x0 + seg_len;

        // build a vertical strip spanning full cavity height
        std::vector<Vecd> seg_poly;
        seg_poly.push_back(Vecd(x0, -H / 2 - BW)); // bottom down wall boundary
        seg_poly.push_back(Vecd(x1, -H / 2 - BW));
        seg_poly.push_back(Vecd(x1, -H / 2));
        seg_poly.push_back(Vecd(x0, -H / 2));
        seg_poly.push_back(Vecd(x0, -H / 2 - BW));

        MultiPolygon seg_shape;
        seg_shape.addAPolygon(seg_poly, ShapeBooleanOps::add);

        // define a particle region over the diffusion_body
        BodyRegionByParticle seg_region(getDownDirichletBody(), makeShared<MultiPolygonShape>(seg_shape, "down_wall_seg_region_tmp"));

        // sum the same reduced quantity
        ExtendedReducedQuantityRecording<solid_dynamics::AveragedWallNu<BodyRegionByParticle>> seg_Nu(seg_region, "WallLocalNusseltNumber");

        return seg_Nu.getReducedQuantity();
    }
    //----------------------------------------------------------------------
    //  Set bottom-wall temperature layout.
    //----------------------------------------------------------------------
    void setDownWallSegmentTemperatures(const StdVec<Real> &Ts)
    {
        if (Ts.empty())
            return;

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
                Real Dt_phase_1 = phase_1_advection_time_step.exec();
                Real Dt_phase_2 = phase_2_advection_time_step.exec();
                Real Dt = SMIN(Dt_phase_1, Dt_phase_2);

                phase_1_update_density_by_summation.exec();
                phase_2_update_density_by_summation.exec();
                phase_1_kernel_correction_complex.exec();
                phase_2_kernel_correction_complex.exec();
                phase_1_viscous_force.exec();
                phase_2_viscous_force.exec();
                phase_1_transport_correction.exec();
                phase_2_transport_correction.exec();

                size_t inner_ite_dt = 0;
                Real relaxation_time = 0.0;
                while (relaxation_time < Dt)
                {
                    Real dt_phase_1 = phase_1_acoustic_time_step.exec();
                    Real dt_phase_2 = phase_2_acoustic_time_step.exec();

                    Real thermal_phase_1 = phase_1_thermal_time_step.exec();
                    Real thermal_phase_2 = phase_2_thermal_time_step.exec();

                    Real dt = SMIN(SMIN(dt_phase_1, dt_phase_2), SMIN(thermal_phase_1, thermal_phase_2), Dt - relaxation_time);
                    phase_1_buoyancy_force.exec();
                    phase_2_buoyancy_force.exec();
                    phase_1_pressure_relaxation.exec(dt);
                    phase_2_pressure_relaxation.exec(dt);
                    phase_1_density_relaxation.exec(dt);
                    phase_2_density_relaxation.exec(dt);
                    phase_1_temperature_relaxation.exec(dt);
                    phase_2_temperature_relaxation.exec(dt);

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

                    //write_states.writeToFile(); // save memory of disk
                }
                number_of_iterations++;

                if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
                {
                    phase_1_particle_sorting.exec();
                    phase_2_particle_sorting.exec();
                }
                PhaseOne_diffusion_body.updateCellLinkedList();
                PhaseTwo_diffusion_body.updateCellLinkedList();
                phase_1_complex.updateConfiguration();
                phase_2_complex.updateConfiguration();
                up_Dirichlet_contacts.updateConfiguration();
                down_Dirichlet_contacts.updateConfiguration();

                phase_1_target_fluid_particles.exec();
                phase_2_target_fluid_particles.exec();
                target_up_solid_particles.exec();
                target_down_solid_particles.exec();
                observer_diffusion_body_contact.updateConfiguration();
            }

            TickCount t2 = TickCount::now();
            calculate_phase_1_phi_gradient.exec();
            calculate_phase_2_phi_gradient.exec();
            phase_1_local_nusselt_number.exec();
            phase_2_local_nusselt_number.exec();
            up_wall_local_nusselt_number.exec();
            down_wall_local_nusselt_number.exec();

            write_states.writeToFile();
            write_phase_1_PhiFluxSum.writeToFile(number_of_iterations);
            write_phase_2_PhiFluxSum.writeToFile(number_of_iterations);
            write_observed_fluid_vel.writeToFile(number_of_iterations);
            write_observed_fluid_temperature.writeToFile(number_of_iterations);
            write_phase_1_GlobalKineticEnergy.writeToFile(number_of_iterations);
            write_phase_2_GlobalKineticEnergy.writeToFile(number_of_iterations);
            write_GlobalAveragedNu.writeToFile(number_of_iterations);

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
PYBIND11_MODULE(zcx_2d_NCCV_2phase_nonperiodic_python, m)
{
    py::class_<SphNaturalConvection>(m, "natural_convection_from_sph_cpp")
        .def(py::init<const int &, const int &, const int &>(), py::arg("parallel_env"), py::arg("episode_env"), py::arg("set_restart_step") = 0)
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
