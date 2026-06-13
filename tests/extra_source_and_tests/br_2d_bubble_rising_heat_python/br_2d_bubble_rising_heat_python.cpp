/**
 * @file    bubble_rising_heat_python.cpp
 * @brief   Python-callable wrapper for bubble_rising_heat.
 */

#include "br_2d_bubble_rising_heat_python.h"
#include "custom_io_environment.h"
#include "custom_io_observation.h"
#include "sphinxsys.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

using namespace SPH;
namespace py = pybind11;

//==============================================================================
// Basic SPH system setting.
//==============================================================================
class SphBasicSystemSetting : public SphBasicGeometrySetting
{
  protected:
    SPHSystem sph_system;
    std::unique_ptr<CustomIOEnvironment> custom_io_environment;

    bool reload_particles_;
    bool write_output_;

  public:
    SphBasicSystemSetting(
        int parallel_env,
        int episode_env,
        bool reload_particles = false,
        bool write_output = false)
        : sph_system(system_domain_bounds, resolution_ref),
          reload_particles_(reload_particles),
          write_output_(write_output)
    {
        sph_system.setRunParticleRelaxation(false);
        sph_system.setReloadParticles(reload_particles_);

        custom_io_environment = std::make_unique<CustomIOEnvironment>(
            sph_system,
            true,
            parallel_env,
            episode_env);
    }
};

//==============================================================================
// Body creation environment.
//==============================================================================
class SphBodyReloadEnvironment : public SphBasicSystemSetting
{
  protected:
    FluidBody liquid_body;
    FluidBody bubble_body;

    SolidBody wall_boundary;
    SolidBody no_slip_wall;
    SolidBody left_Dirichlet;
    SolidBody right_Dirichlet;

    ObserverBody flow_observer;

  public:
    SphBodyReloadEnvironment(
        int parallel_env,
        int episode_env,
        bool reload_particles = false,
        bool write_output = false)
        : SphBasicSystemSetting(
              parallel_env,
              episode_env,
              reload_particles,
              write_output),
          liquid_body(sph_system, makeShared<LiquidBody>("LiquidBody")),
          bubble_body(sph_system, makeShared<BubbleBody>("BubbleBody")),
          wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary")),
          no_slip_wall(sph_system, makeShared<NoSlipWall>("NoSlipWall")),
          left_Dirichlet(sph_system, makeShared<LeftDirichletWall>("LeftDirichlet")),
          right_Dirichlet(sph_system, makeShared<RightDirichletWall>("RightDirichlet")),
          flow_observer(sph_system, "FlowObserver")
    {
        //----------------------------------------------------------------------
        // Fluid bodies.
        //----------------------------------------------------------------------
        liquid_body.defineComponentLevelSetShape("OuterBoundary");
        liquid_body.defineClosure<
            WeaklyCompressibleFluid,
            Viscosity,
            IsotropicThermalDiffusion>(
            ConstructArgs(rho0_l, c_f),
            mu_l,
            ConstructArgs(diffusion_species_name, k_l, rho0_l, C_p_l));

        if (reload_particles_)
        {
            liquid_body.generateParticles<BaseParticles, Reload>(
                liquid_body.getName());
        }
        else
        {
            liquid_body.generateParticles<BaseParticles, Lattice>();
        }

        bubble_body.defineBodyLevelSetShape();
        bubble_body.defineClosure<
            WeaklyCompressibleFluid,
            Viscosity,
            IsotropicThermalDiffusion>(
            ConstructArgs(rho0_g, c_f),
            mu_g,
            ConstructArgs(diffusion_species_name, k_g, rho0_g, C_p_g));

        if (reload_particles_)
        {
            bubble_body.generateParticles<BaseParticles, Reload>(
                bubble_body.getName());
        }
        else
        {
            bubble_body.generateParticles<BaseParticles, Lattice>();
        }

        //----------------------------------------------------------------------
        // Solid bodies.
        //----------------------------------------------------------------------
        wall_boundary.defineMaterial<Solid>();
        wall_boundary.generateParticles<BaseParticles, Lattice>();

        no_slip_wall.defineMaterial<Solid>();
        no_slip_wall.generateParticles<BaseParticles, Lattice>();

        left_Dirichlet.defineMaterial<Solid>();
        left_Dirichlet.generateParticles<BaseParticles, Lattice>();

        right_Dirichlet.defineMaterial<Solid>();
        right_Dirichlet.generateParticles<BaseParticles, Lattice>();

        //----------------------------------------------------------------------
        // Observer body.
        // createObservationPoints() should be a grid over the whole flow field.
        //----------------------------------------------------------------------
        flow_observer.generateParticles<ObserverParticles>(
            createObservationPoints());
    }

    FluidBody &getLiquidBody()
    {
        return liquid_body;
    }

    FluidBody &getBubbleBody()
    {
        return bubble_body;
    }

    SolidBody &getLeftDirichletWall()
    {
        return left_Dirichlet;
    }

    SolidBody &getRightDirichletWall()
    {
        return right_Dirichlet;
    }
};

//==============================================================================
// Python-callable simulation class.
//==============================================================================
class SphBubbleRisingHeat : public SphBodyReloadEnvironment
{
  protected:
    //----------------------------------------------------------------------
    // Body relations.
    //----------------------------------------------------------------------
    InnerRelation liquid_inner;
    InnerRelation bubble_inner;

    ContactRelation liquid_contact_bubble;
    ContactRelation liquid_contact_wall;
    ContactRelation liquid_contact_no_slip_wall;
    ContactRelation liquid_contact_left_Dirichlet;
    ContactRelation liquid_contact_right_Dirichlet;
    ContactRelation liquid_contact;

    ContactRelation bubble_contact_liquid;
    ContactRelation bubble_contact_wall;
    ContactRelation bubble_contact_no_slip_wall;
    ContactRelation bubble_contact_left_Dirichlet;
    ContactRelation bubble_contact_right_Dirichlet;
    ContactRelation bubble_contact;

    ComplexRelation liquid_complex;
    ComplexRelation bubble_complex;

    ContactRelation observer_contact;

    //----------------------------------------------------------------------
    // Normal directions.
    //----------------------------------------------------------------------
    SimpleDynamics<NormalDirectionFromBodyShape> liquid_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> bubble_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> wall_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> no_slip_wall_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> left_Dirichlet_normal_direction;
    SimpleDynamics<NormalDirectionFromBodyShape> right_Dirichlet_normal_direction;

    //----------------------------------------------------------------------
    // Thermal dynamics.
    //----------------------------------------------------------------------
    MultiPhaseDiffusionBodyRelaxation liquid_temperature_relaxation;
    MultiPhaseDiffusionBodyRelaxation bubble_temperature_relaxation;

    GetDiffusionTimeStepSize liquid_thermal_time_step;
    GetDiffusionTimeStepSize bubble_thermal_time_step;

    SimpleDynamics<DiffusionInitialCondition> liquid_initial_condition;
    SimpleDynamics<DiffusionInitialCondition> bubble_initial_condition;

    SimpleDynamics<DirichletWallBoundaryInitialCondition>
        setup_left_Dirichlet_initial_condition;
    SimpleDynamics<DirichletWallBoundaryInitialCondition>
        setup_right_Dirichlet_initial_condition;
    SimpleDynamics<NeumannWallBoundaryInitialCondition>
        setup_Neumann_initial_condition;

    SimpleDynamics<fluid_dynamics::BuoyancyForce>
        liquid_buoyancy_force;
    SimpleDynamics<fluid_dynamics::BuoyancyForce>
        bubble_buoyancy_force;

    //----------------------------------------------------------------------
    // Fluid dynamics.
    //----------------------------------------------------------------------
    Gravity gravity;

    SimpleDynamics<GravityForce<Gravity>> liquid_gravity;
    SimpleDynamics<GravityForce<Gravity>> bubble_gravity;

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex>
        liquid_kernel_correction_complex;
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex>
        bubble_kernel_correction_complex;

    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfWithWallRiemann>
        liquid_pressure_relaxation;
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann>
        liquid_density_relaxation;

    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration1stHalfWithWallRiemann>
        bubble_pressure_relaxation;
    Dynamics1Level<fluid_dynamics::MultiPhaseIntegration2ndHalfWithWallRiemann>
        bubble_density_relaxation;

    InteractionWithUpdate<
        fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>>
        liquid_update_density_by_summation;
    InteractionWithUpdate<
        fluid_dynamics::BaseDensitySummationComplex<Inner<>, Contact<>, Contact<>>>
        bubble_update_density_by_summation;

    InteractionWithUpdate<
        fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>>
        liquid_transport_correction;
    InteractionWithUpdate<
        fluid_dynamics::MultiPhaseTransportVelocityCorrectionComplex<AllParticles>>
        bubble_transport_correction;

    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall>
        liquid_viscous_force;
    InteractionWithUpdate<fluid_dynamics::MultiPhaseViscousForceWithWall>
        bubble_viscous_force;

    InteractionDynamics<fluid_dynamics::SurfaceTensionStress>
        liquid_surface_tension_stress;
    InteractionDynamics<fluid_dynamics::SurfaceTensionStress>
        bubble_surface_tension_stress;

    InteractionWithUpdate<fluid_dynamics::SurfaceStressForceComplex>
        liquid_surface_tension_force;
    InteractionWithUpdate<fluid_dynamics::SurfaceStressForceComplex>
        bubble_surface_tension_force;

    InteractionWithUpdate<fluid_dynamics::InterfaceSharpnessForceComplex>
        liquid_interface_sharpness_force;
    InteractionWithUpdate<fluid_dynamics::InterfaceSharpnessForceComplex>
        bubble_interface_sharpness_force;

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep>
        liquid_advection_time_step;
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep>
        bubble_advection_time_step;

    ReduceDynamics<fluid_dynamics::AcousticTimeStep>
        liquid_acoustic_time_step;
    ReduceDynamics<fluid_dynamics::AcousticTimeStep>
        bubble_acoustic_time_step;

    ParticleSorting<ParallelPolicy> liquid_particle_sorting;
    ParticleSorting<ParallelPolicy> bubble_particle_sorting;

    //----------------------------------------------------------------------
    // Output, observation and metrics.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_states;

    ObservedQuantityRecording<Vecd> write_observed_flow_velocity;
    ObservedQuantityRecording<Real> write_observed_flow_temperature;

    //ExtendedReducedQuantityRecording<TotalKineticEnergy>
    //    write_liquid_kinetic_energy;
    //ExtendedReducedQuantityRecording<TotalKineticEnergy>
    //    write_bubble_kinetic_energy;

    BubbleControlMetricsCalculator bubble_metrics_calculator;

    //----------------------------------------------------------------------
    // Time control.
    //----------------------------------------------------------------------
    Real &physical_time =
        *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");

    int number_of_iterations = 0;
    int screen_output_interval = 100;
    Real output_interval = 0.02;

  public:
    explicit SphBubbleRisingHeat(
        int parallel_env = 0,
        int episode_env = 0,
        bool reload_particles = false,
        bool write_output = false)
        : SphBodyReloadEnvironment(
              parallel_env,
              episode_env,
              reload_particles,
              write_output),

          //------------------------------------------------------------------
          // Relations.
          //------------------------------------------------------------------
          liquid_inner(liquid_body),
          bubble_inner(bubble_body),

          liquid_contact_bubble(liquid_body, {&bubble_body}),
          liquid_contact_wall(liquid_body, {&wall_boundary}),
          liquid_contact_no_slip_wall(liquid_body, {&no_slip_wall}),
          liquid_contact_left_Dirichlet(liquid_body, {&left_Dirichlet}),
          liquid_contact_right_Dirichlet(liquid_body, {&right_Dirichlet}),
          liquid_contact(liquid_body, {&bubble_body, &wall_boundary}),

          bubble_contact_liquid(bubble_body, {&liquid_body}),
          bubble_contact_wall(bubble_body, {&wall_boundary}),
          bubble_contact_no_slip_wall(bubble_body, {&no_slip_wall}),
          bubble_contact_left_Dirichlet(bubble_body, {&left_Dirichlet}),
          bubble_contact_right_Dirichlet(bubble_body, {&right_Dirichlet}),
          bubble_contact(bubble_body, {&liquid_body, &wall_boundary}),

          liquid_complex(
              liquid_inner,
              {&liquid_contact_bubble,
               &liquid_contact_wall,
               &liquid_contact_no_slip_wall,
               &liquid_contact_left_Dirichlet,
               &liquid_contact_right_Dirichlet}),
          bubble_complex(
              bubble_inner,
              {&bubble_contact_liquid,
               &bubble_contact_wall,
               &bubble_contact_no_slip_wall,
               &bubble_contact_left_Dirichlet,
               &bubble_contact_right_Dirichlet}),

          // Observer must contact both phases to observe full-field u, v, T.
          observer_contact(flow_observer, {&liquid_body, &bubble_body}),

          //------------------------------------------------------------------
          // Normal directions.
          //------------------------------------------------------------------
          liquid_normal_direction(liquid_body),
          bubble_normal_direction(bubble_body),
          wall_normal_direction(wall_boundary),
          no_slip_wall_normal_direction(no_slip_wall),
          left_Dirichlet_normal_direction(left_Dirichlet),
          right_Dirichlet_normal_direction(right_Dirichlet),

          //------------------------------------------------------------------
          // Thermal dynamics.
          //------------------------------------------------------------------
          liquid_temperature_relaxation(
              liquid_inner,
              liquid_contact_bubble,
              liquid_contact_left_Dirichlet,
              liquid_contact_right_Dirichlet,
              liquid_contact_no_slip_wall),
          bubble_temperature_relaxation(
              bubble_inner,
              bubble_contact_liquid,
              bubble_contact_left_Dirichlet,
              bubble_contact_right_Dirichlet,
              bubble_contact_no_slip_wall),

          liquid_thermal_time_step(liquid_body),
          bubble_thermal_time_step(bubble_body),

          liquid_initial_condition(liquid_body),
          bubble_initial_condition(bubble_body),

          setup_left_Dirichlet_initial_condition(left_Dirichlet),
          setup_right_Dirichlet_initial_condition(right_Dirichlet),
          setup_Neumann_initial_condition(no_slip_wall),
          
          liquid_buoyancy_force(liquid_body, thermal_expansion_l, reference_temperature),
          bubble_buoyancy_force(bubble_body, thermal_expansion_g, reference_temperature),

          //------------------------------------------------------------------
          // Fluid dynamics.
          //------------------------------------------------------------------
          gravity(Vecd(0.0, -gravity_g)),

          liquid_gravity(liquid_body, gravity),
          bubble_gravity(bubble_body, gravity),

          liquid_kernel_correction_complex(
              InteractArgs(liquid_inner, 0.5),
              liquid_contact),
          bubble_kernel_correction_complex(
              InteractArgs(bubble_inner, 0.5),
              bubble_contact),

          liquid_pressure_relaxation(
              liquid_inner,
              liquid_contact_bubble,
              liquid_contact_wall),
          liquid_density_relaxation(
              liquid_inner,
              liquid_contact_bubble,
              liquid_contact_wall),

          bubble_pressure_relaxation(
              bubble_inner,
              bubble_contact_liquid,
              bubble_contact_wall),
          bubble_density_relaxation(
              bubble_inner,
              bubble_contact_liquid,
              bubble_contact_wall),

          liquid_update_density_by_summation(
              liquid_inner,
              liquid_contact_bubble,
              liquid_contact_wall),
          bubble_update_density_by_summation(
              bubble_inner,
              bubble_contact_liquid,
              bubble_contact_wall),

          liquid_transport_correction(
              liquid_inner,
              liquid_contact_bubble,
              liquid_contact_wall),
          bubble_transport_correction(
              bubble_inner,
              bubble_contact_liquid,
              bubble_contact_wall),

          liquid_viscous_force(
              liquid_inner,
              liquid_contact_bubble,
              liquid_contact_no_slip_wall),
          bubble_viscous_force(
              bubble_inner,
              bubble_contact_liquid,
              bubble_contact_no_slip_wall),

          liquid_surface_tension_stress(
              liquid_contact_bubble,
              StdVec<Real>{surface_tension}),
          bubble_surface_tension_stress(
              bubble_contact_liquid,
              StdVec<Real>{surface_tension / 10.0}),

          liquid_surface_tension_force(
              liquid_inner,
              liquid_contact_bubble),
          bubble_surface_tension_force(
              bubble_inner,
              bubble_contact_liquid),

          liquid_interface_sharpness_force(
              liquid_inner,
              liquid_contact_bubble),
          bubble_interface_sharpness_force(
              bubble_inner,
              bubble_contact_liquid),

          liquid_advection_time_step(liquid_body, U_f),
          bubble_advection_time_step(bubble_body, U_f),

          liquid_acoustic_time_step(liquid_body),
          bubble_acoustic_time_step(bubble_body),

          liquid_particle_sorting(liquid_body),
          bubble_particle_sorting(bubble_body),

          //------------------------------------------------------------------
          // Output, observation and metrics.
          //------------------------------------------------------------------
          write_states(sph_system),
          write_observed_flow_velocity("Velocity", observer_contact),
          write_observed_flow_temperature(diffusion_species_name, observer_contact),
          //write_liquid_kinetic_energy(liquid_body),
          //write_bubble_kinetic_energy(bubble_body),
          bubble_metrics_calculator(bubble_body)
    {
        physical_time = 0.0;

        if (write_output_)
        {
            write_states.addToWrite<Real>(liquid_body, "Pressure");
            write_states.addToWrite<Real>(liquid_body, diffusion_species_name);

            write_states.addToWrite<Real>(bubble_body, "Pressure");
            write_states.addToWrite<Real>(bubble_body, diffusion_species_name);

            write_states.addToWrite<Vecd>(wall_boundary, "NormalDirection");
            write_states.addToWrite<Real>(left_Dirichlet, diffusion_species_name);
            write_states.addToWrite<Real>(right_Dirichlet, diffusion_species_name);
        }

        //----------------------------------------------------------------------
        // Prepare system.
        //----------------------------------------------------------------------
        sph_system.initializeSystemCellLinkedLists();
        sph_system.initializeSystemConfigurations();

        //----------------------------------------------------------------------
        // Initial conditions.
        //----------------------------------------------------------------------
        liquid_initial_condition.exec();
        bubble_initial_condition.exec();

        setup_left_Dirichlet_initial_condition.exec();
        setup_right_Dirichlet_initial_condition.exec();
        setup_Neumann_initial_condition.exec();

        //----------------------------------------------------------------------
        // Normal directions.
        //----------------------------------------------------------------------
        liquid_normal_direction.exec();
        bubble_normal_direction.exec();
        wall_normal_direction.exec();
        no_slip_wall_normal_direction.exec();
        left_Dirichlet_normal_direction.exec();
        right_Dirichlet_normal_direction.exec();

        //----------------------------------------------------------------------
        // Initial body force.
        //----------------------------------------------------------------------
        liquid_gravity.exec();
        bubble_gravity.exec();

        //----------------------------------------------------------------------
        // Initialize observer values once.
        //----------------------------------------------------------------------
        write_observed_flow_velocity.writeToFile(number_of_iterations);
        write_observed_flow_temperature.writeToFile(number_of_iterations);

        if (write_output_)
        {
            write_states.writeToFile();
            //write_liquid_kinetic_energy.writeToFile(number_of_iterations);
            //write_bubble_kinetic_energy.writeToFile(number_of_iterations);
        }
    }

    virtual ~SphBubbleRisingHeat() {}

    //--------------------------------------------------------------------------
    // Basic control.
    //--------------------------------------------------------------------------
    int cmakeTest()
    {
        return 1;
    }

    void set_output_interval(Real interval)
    {
        if (interval > Real(0))
        {
            output_interval = interval;
        }
    }

    Real get_physical_time() const
    {
        return physical_time;
    }

    int get_number_of_iterations() const
    {
        return number_of_iterations;
    }

    //--------------------------------------------------------------------------
    // Left-wall segmented temperature control.
    //--------------------------------------------------------------------------
    void setLeftWallSegmentTemperatures(
        const StdVec<Real> &temperatures,
        bool enforce_mean = true,
        Real mean_temperature = 1.0)
    {
        if (temperatures.empty())
        {
            return;
        }

        SphBasicGeometrySetting::setLeftWallSegmentTemperatures(
            temperatures,
            enforce_mean,
            mean_temperature);

        setup_left_Dirichlet_initial_condition.exec();
    }

    void setLeftWallSegmentActions(
        const StdVec<Real> &actions,
        Real amplitude = 0.3,
        Real mean_temperature = 1.0)
    {
        if (actions.empty())
        {
            return;
        }

        SphBasicGeometrySetting::setLeftWallSegmentActions(
            actions,
            amplitude,
            mean_temperature);

        setup_left_Dirichlet_initial_condition.exec();
    }

    StdVec<Real> getLeftWallSegmentTemperatures() const
    {
        return SphBasicGeometrySetting::left_wall_segment_T;
    }

    //--------------------------------------------------------------------------
    // Main time stepping.
    //--------------------------------------------------------------------------
    void runCase(Real pause_time_from_python)
    {
        while (physical_time < pause_time_from_python)
        {
            Real integration_time = 0.0;

            while (integration_time < output_interval &&
                   physical_time < pause_time_from_python)
            {
                Real Dt_liquid = liquid_advection_time_step.exec();
                Real Dt_bubble = bubble_advection_time_step.exec();

                Real Dt = SMIN(Dt_liquid, Dt_bubble);
                Dt = SMIN(Dt, pause_time_from_python - physical_time);

                //------------------------------------------------------------------
                // Non-pressure force and transport stage.
                //------------------------------------------------------------------
                liquid_update_density_by_summation.exec();
                bubble_update_density_by_summation.exec();

                liquid_kernel_correction_complex.exec();
                bubble_kernel_correction_complex.exec();

                liquid_viscous_force.exec();
                bubble_viscous_force.exec();

                liquid_transport_correction.exec();
                bubble_transport_correction.exec();

                liquid_surface_tension_stress.exec();
                bubble_surface_tension_stress.exec();

                liquid_surface_tension_force.exec();
                bubble_surface_tension_force.exec();

                liquid_interface_sharpness_force.exec();
                bubble_interface_sharpness_force.exec();

                //------------------------------------------------------------------
                // Acoustic and thermal sub-stepping.
                //------------------------------------------------------------------
                size_t inner_ite_dt = 0;
                Real relaxation_time = 0.0;

                while (relaxation_time < Dt)
                {
                    Real dt_liquid = liquid_acoustic_time_step.exec();
                    Real dt_bubble = bubble_acoustic_time_step.exec();

                    Real thermal_dt_liquid = liquid_thermal_time_step.exec();
                    Real thermal_dt_bubble = bubble_thermal_time_step.exec();

                    Real acoustic_dt = SMIN(dt_liquid, dt_bubble);
                    Real thermal_dt = SMIN(thermal_dt_liquid, thermal_dt_bubble);

                    Real dt = SMIN(
                        SMIN(acoustic_dt, thermal_dt),
                        Dt - relaxation_time);

                    liquid_buoyancy_force.exec();
                    bubble_buoyancy_force.exec();

                    liquid_pressure_relaxation.exec(dt);
                    bubble_pressure_relaxation.exec(dt);

                    liquid_density_relaxation.exec(dt);
                    bubble_density_relaxation.exec(dt);

                    liquid_temperature_relaxation.exec(dt);
                    bubble_temperature_relaxation.exec(dt);

                    relaxation_time += dt;
                    integration_time += dt;
                    physical_time += dt;
                    inner_ite_dt++;
                }

                if (number_of_iterations % screen_output_interval == 0)
                {
                    std::cout << std::fixed << std::setprecision(9)
                              << "N=" << number_of_iterations
                              << "\tTime=" << physical_time
                              << "\tDt=" << Dt
                              << "\tDt/dt=" << inner_ite_dt
                              << "\n";
                }

                number_of_iterations++;

                if (number_of_iterations % 100 == 0 &&
                    number_of_iterations != 1)
                {
                    liquid_particle_sorting.exec();
                    bubble_particle_sorting.exec();
                }

                //------------------------------------------------------------------
                // Update configurations.
                //------------------------------------------------------------------
                liquid_body.updateCellLinkedList();
                bubble_body.updateCellLinkedList();

                liquid_complex.updateConfiguration();
                bubble_complex.updateConfiguration();

                observer_contact.updateConfiguration();
            }

            //----------------------------------------------------------------------
            // Keep observer quantities fresh for Python calls.
            //----------------------------------------------------------------------
            write_observed_flow_velocity.writeToFile(number_of_iterations);
            write_observed_flow_temperature.writeToFile(number_of_iterations);

            if (write_output_)
            {
                write_states.writeToFile();
                //write_liquid_kinetic_energy.writeToFile(number_of_iterations);
                //write_bubble_kinetic_energy.writeToFile(number_of_iterations);
            }
        }
    }

    void step(Real delta_time)
    {
        if (delta_time <= Real(0))
        {
            return;
        }

        runCase(physical_time + delta_time);
    }

    //--------------------------------------------------------------------------
    // Bubble metrics.
    //--------------------------------------------------------------------------
    BubbleControlMetrics getBubbleMetrics()
    {
        return bubble_metrics_calculator.compute();
    }

    py::dict getBubbleMetricsDict()
    {
        BubbleControlMetrics m = bubble_metrics_calculator.compute();

        py::dict d;

        d["center_x"] = m.center_x;
        d["center_y"] = m.center_y;
        d["center_u"] = m.center_u;
        d["center_v"] = m.center_v;

        d["x_min"] = m.x_min;
        d["x_max"] = m.x_max;
        d["y_min"] = m.y_min;
        d["y_max"] = m.y_max;

        d["bubble_width"] = m.bubble_width;
        d["bubble_height"] = m.bubble_height;
        d["deformation_index"] = m.deformation_index;
        d["aspect_ratio"] = m.aspect_ratio;

        d["bubble_area"] = m.bubble_area;
        d["area_ratio"] = m.area_ratio;

        d["centroid_in_target"] = m.centroid_in_target;
        d["reached_target_height"] = m.reached_target_height;

        d["left_particle_in_target"] = m.left_particle_in_target;
        d["right_particle_in_target"] = m.right_particle_in_target;
        d["bottom_particle_in_target"] = m.bottom_particle_in_target;
        d["top_particle_in_target"] = m.top_particle_in_target;

        d["all_extreme_particles_in_target"] =
            m.all_extreme_particles_in_target;

        return d;
    }

    //--------------------------------------------------------------------------
    // Observations: reward is deliberately separated from observations.
    //--------------------------------------------------------------------------
    StdVec<Real> getFlowObservation()
    {
        StdVec<Real> obs;

        Vecd *velocities =
            write_observed_flow_velocity.getObservedQuantity();

        Real *temperatures =
            write_observed_flow_temperature.getObservedQuantity();

        const size_t n = flow_observer.getBaseParticles().TotalRealParticles();

        obs.reserve(3 * n);

        for (size_t i = 0; i != n; ++i)
        {
            obs.push_back(velocities[i][0] / (U_f + Eps));
            obs.push_back(velocities[i][1] / (U_f + Eps));
            obs.push_back(temperatures[i]);
        }

        return obs;
    }

    StdVec<Real> getBubbleObservation()
    {
        BubbleControlMetrics m = bubble_metrics_calculator.compute();

        StdVec<Real> obs;
        obs.reserve(16);

        obs.push_back(m.center_x / DL);
        obs.push_back(m.center_y / DH);
        obs.push_back(m.center_u / (U_f + Eps));
        obs.push_back(m.center_v / (U_f + Eps));

        obs.push_back(m.x_min / DL);
        obs.push_back(m.x_max / DL);
        obs.push_back(m.y_min / DH);
        obs.push_back(m.y_max / DH);

        obs.push_back(m.bubble_width / DL);
        obs.push_back(m.bubble_height / DH);

        obs.push_back(m.deformation_index);
        obs.push_back(m.aspect_ratio);
        obs.push_back(m.area_ratio);

        obs.push_back(Real(m.centroid_in_target));
        obs.push_back(Real(m.reached_target_height));
        obs.push_back(Real(m.all_extreme_particles_in_target));

        return obs;
    }

    //--------------------------------------------------------------------------
    // Termination / state checks.
    //--------------------------------------------------------------------------
    bool hasReachedTargetHeight()
    {
        BubbleControlMetrics m = bubble_metrics_calculator.compute();
        return m.reached_target_height == 1;
    }

    bool isBubbleInTargetRegion()
    {
        BubbleControlMetrics m = bubble_metrics_calculator.compute();
        return m.centroid_in_target == 1;
    }

    bool isWholeBubbleInTargetRegion()
    {
        BubbleControlMetrics m = bubble_metrics_calculator.compute();
        return m.all_extreme_particles_in_target == 1;
    }
 
    //--------------------------------------------------------------------------
    // Debug helper.
    //--------------------------------------------------------------------------
    int debugSmokeTest()
    {
        std::cout << "---- debugSmokeTest() begin ----\n";

        std::cout << "[0] physical_time = " << physical_time << "\n";

        StdVec<Real> actions;
        actions.push_back(1.0);
        actions.push_back(0.2);
        actions.push_back(-0.4);
        actions.push_back(-0.8);

        std::cout << "[1] setLeftWallSegmentActions ...\n";
        setLeftWallSegmentActions(actions, 0.3, 1.0);

        Real target_time = physical_time + 0.02;
        std::cout << "[2] runCase(" << target_time << ") ...\n";
        runCase(target_time);

        BubbleControlMetrics m = getBubbleMetrics();

        StdVec<Real> flow_obs = getFlowObservation();
        StdVec<Real> bubble_obs = getBubbleObservation();

        std::cout << "[3] after runCase:\n";
        std::cout << "    physical_time = " << physical_time << "\n";
        std::cout << "    center = (" << m.center_x << ", " << m.center_y << ")\n";
        std::cout << "    velocity = (" << m.center_u << ", " << m.center_v << ")\n";
        std::cout << "    deformation_index = " << m.deformation_index << "\n";
        std::cout << "    area_ratio = " << m.area_ratio << "\n";
        std::cout << "    bubble_obs_size = " << bubble_obs.size() << "\n";
        std::cout << "    flow_obs_size = " << flow_obs.size() << "\n";

        std::cout << "---- debugSmokeTest() end ----\n";

        return 1;
    }
};

//==============================================================================
// pybind11 module.
//==============================================================================
PYBIND11_MODULE(br_2d_bubble_rising_heat_python, m)
{
    py::class_<BubbleControlMetrics>(m, "BubbleControlMetrics")
        .def_readonly("center_x", &BubbleControlMetrics::center_x)
        .def_readonly("center_y", &BubbleControlMetrics::center_y)
        .def_readonly("center_u", &BubbleControlMetrics::center_u)
        .def_readonly("center_v", &BubbleControlMetrics::center_v)

        .def_readonly("x_min", &BubbleControlMetrics::x_min)
        .def_readonly("x_max", &BubbleControlMetrics::x_max)
        .def_readonly("y_min", &BubbleControlMetrics::y_min)
        .def_readonly("y_max", &BubbleControlMetrics::y_max)

        .def_readonly("bubble_width", &BubbleControlMetrics::bubble_width)
        .def_readonly("bubble_height", &BubbleControlMetrics::bubble_height)
        .def_readonly("deformation_index", &BubbleControlMetrics::deformation_index)
        .def_readonly("aspect_ratio", &BubbleControlMetrics::aspect_ratio)

        .def_readonly("bubble_area", &BubbleControlMetrics::bubble_area)
        .def_readonly("area_ratio", &BubbleControlMetrics::area_ratio)

        .def_readonly("centroid_in_target", &BubbleControlMetrics::centroid_in_target)
        .def_readonly("reached_target_height", &BubbleControlMetrics::reached_target_height)

        .def_readonly("left_particle_in_target", &BubbleControlMetrics::left_particle_in_target)
        .def_readonly("right_particle_in_target", &BubbleControlMetrics::right_particle_in_target)
        .def_readonly("bottom_particle_in_target", &BubbleControlMetrics::bottom_particle_in_target)
        .def_readonly("top_particle_in_target", &BubbleControlMetrics::top_particle_in_target)

        .def_readonly(
            "all_extreme_particles_in_target",
            &BubbleControlMetrics::all_extreme_particles_in_target);

    py::class_<SphBubbleRisingHeat>(m, "bubble_rising_heat_from_sph_cpp")
        .def(
            py::init<int, int, bool, bool>(),
            py::arg("parallel_env") = 0,
            py::arg("episode_env") = 0,
            py::arg("reload_particles") = false,
            py::arg("write_output") = false)

        .def("cmake_test", &SphBubbleRisingHeat::cmakeTest)
        .def("debug_smoke_test", &SphBubbleRisingHeat::debugSmokeTest)

        .def("run_case", &SphBubbleRisingHeat::runCase, py::arg("target_time"))
        .def("step", &SphBubbleRisingHeat::step, py::arg("delta_time"))

        .def(
            "set_left_wall_segment_temperatures",
            &SphBubbleRisingHeat::setLeftWallSegmentTemperatures,
            py::arg("temperatures"),
            py::arg("enforce_mean") = true,
            py::arg("mean_temperature") = 1.0)

        .def(
            "set_left_wall_segment_actions",
            &SphBubbleRisingHeat::setLeftWallSegmentActions,
            py::arg("actions"),
            py::arg("amplitude") = 0.3,
            py::arg("mean_temperature") = 1.0)

        .def(
            "get_left_wall_segment_temperatures",
            &SphBubbleRisingHeat::getLeftWallSegmentTemperatures)

        .def("get_bubble_metrics", &SphBubbleRisingHeat::getBubbleMetrics)
        .def("get_bubble_metrics_dict", &SphBubbleRisingHeat::getBubbleMetricsDict)

        .def("get_flow_observation", &SphBubbleRisingHeat::getFlowObservation)
        .def("get_bubble_observation", &SphBubbleRisingHeat::getBubbleObservation)

        .def("has_reached_target_height", &SphBubbleRisingHeat::hasReachedTargetHeight)
        .def("is_bubble_in_target_region", &SphBubbleRisingHeat::isBubbleInTargetRegion)
        .def("is_whole_bubble_in_target_region", &SphBubbleRisingHeat::isWholeBubbleInTargetRegion)

        .def("set_output_interval", &SphBubbleRisingHeat::set_output_interval)
        .def("get_physical_time", &SphBubbleRisingHeat::get_physical_time)
        .def("get_number_of_iterations", &SphBubbleRisingHeat::get_number_of_iterations);
}