/**
 * @file 	2d_eulerian_flow_around_cylinder_LG.cpp
 * @brief 	This is the test file for the weakly compressible viscous flow around a cylinder coupling with the Laguerre Gauss kernel.
 * @details We consider a Eulerian flow passing by a cylinder in 2D.
 * @author 	Zhentong Wang and Xiangyu Hu
 */
#include "eulerian_fluid_dynamics.hpp" // eulerian classes for weakly compressible fluid only.
#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Set the file path to the data file.
//----------------------------------------------------------------------
std::string airfoil = "./input/NACA5515.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0;
Real DL = 3 * L;
Real DL1 = 2 * L;
Real DH = L;
Real airfoil_h = 0.1;
Real resolution_ref = 0.01; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(-DL1, -DH), Vec2d(DL, DH));
//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho0_f = 1.0;                                       /**< Density. */
Real U_f = 0.42;                                          /**< freestream velocity. */
Real c_f = 10.0 * U_f;                                   /**< Speed of sound. */
Real Re = 420.0;                                         /**< Reynolds number. */
Real mu_f = 1.0e-3;       /**< Dynamics viscosity. */
StdVec<Vecd> observation_location = {Vecd(0.2*L, 0.086*L)};
//----------------------------------------------------------------------
//	Define geometries and body shapes
//----------------------------------------------------------------------
class AirfoilModel : public MultiPolygonShape
{
  public:
    explicit AirfoilModel(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addAPolygonFromFile(airfoil, ShapeBooleanOps::add);
    }
};

std::vector<Vecd> createWaterBlockShape()
{
    std::vector<Vecd> water_block_shape;
    water_block_shape.push_back(Vecd(-DL1, -DH));
    water_block_shape.push_back(Vecd(-DL1, DH));
    water_block_shape.push_back(Vecd(DL, DH));
    water_block_shape.push_back(Vecd(DL, -DH));
    water_block_shape.push_back(Vecd(-DL1, -DH));

    return water_block_shape;
}

class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        MultiPolygon outer_boundary(createWaterBlockShape());
        add<MultiPolygonShape>(outer_boundary, "OuterBoundary");
        AirfoilModel import_model("InnerBody");
        subtract<AirfoilModel>(import_model);
    }
};

class FarFieldBoundary : public fluid_dynamics::NonReflectiveBoundaryCorrection
{
  public:
    explicit FarFieldBoundary(BaseInnerRelation &inner_relation)
        : fluid_dynamics::NonReflectiveBoundaryCorrection(inner_relation)
    {
        rho_farfield_ = rho0_f;
        sound_speed_ = c_f;
        vel_farfield_ = Vecd(U_f, 0.0);
    };
    virtual ~FarFieldBoundary(){};
};
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    // Tag for run particle relaxation for the initial body fitted distribution.
    sph_system.setRunParticleRelaxation(true);
    // Tag for computation start with relaxed body fitted particles distribution.
    sph_system.setReloadParticles(false);
    // Handle command line arguments and override the tags for particle relaxation and reload.
    sph_system.handleCommandlineOptions(ac, av);
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBody airfoil(sph_system, makeShared<AirfoilModel>("Airfoil"));
    //airfoil.defineBodyLevelSetShape()->cleanLevelSet(0.9)->writeLevelSet(io_environment);
    airfoil.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    airfoil.defineParticlesAndMaterial<SolidParticles, Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? airfoil.generateParticles<ParticleGeneratorReload>(io_environment, airfoil.getName())
        : airfoil.generateParticles<ParticleGeneratorLattice>();
    airfoil.addBodyStateForRecording<Real>("Density");

    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    water_block.defineComponentLevelSetShape("OuterBoundary");
    water_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? water_block.generateParticles<ParticleGeneratorReload>(io_environment, water_block.getName())
        : water_block.generateParticles<ParticleGeneratorLattice>();
    water_block.addBodyStateForRecording<Real>("Pressure");
    water_block.addBodyStateForRecording<Real>("Density");

    ObserverBody fluid_observer(sph_system, "FluidObserver");
    fluid_observer.generateParticles<ObserverParticleGenerator>(observation_location);
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //	Note that the same relation should be defined only once.
    //----------------------------------------------------------------------
    InnerRelation water_block_inner(water_block);
    ComplexRelation water_airfoil_complex(water_block, {&airfoil});
    ContactRelation airfoil_water_contact(airfoil, {&water_block});
    ContactRelation fluid_observer_contact(fluid_observer, {&water_block});
    //----------------------------------------------------------------------
    //	Run particle relaxation for body-fitted distribution if chosen.
    //----------------------------------------------------------------------
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation airfoil_inner(airfoil); // extra body topology only for particle relaxation
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        SimpleDynamics<RandomizeParticlePosition> random_airfoil_particles(airfoil);
        SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
        
        relax_dynamics::RelaxationStepInner relaxation_step_inner(airfoil_inner, true);
        relax_dynamics::RelaxationStepComplex relaxation_step_complex(water_airfoil_complex, "OuterBoundary", true);
        airfoil.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");
        water_block.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");

        BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
        ReloadParticleIO write_real_body_particle_reload_files(io_environment, sph_system.real_bodies_);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_airfoil_particles.exec(0.25);
        random_water_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_complex.SurfaceBounding().exec();

        airfoil.updateCellLinkedList();
        water_block.updateCellLinkedList();

        write_real_body_states.writeToFile(0);

        int ite_p = 0;
        while (ite_p < 1000)
        {
            relaxation_step_inner.exec();
            relaxation_step_complex.exec();
            ite_p += 1;
            if (ite_p % 200 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
                write_real_body_states.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process finish !" << std::endl;

        write_real_body_particle_reload_files.writeToFile(0);

        return 0;
    }

    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    InteractionWithUpdate<fluid_dynamics::EulerianIntegration1stHalfAcousticRiemannWithWall> pressure_relaxation(water_airfoil_complex);
    InteractionWithUpdate<fluid_dynamics::EulerianIntegration2ndHalfAcousticRiemannWithWall> density_relaxation(water_airfoil_complex);
    InteractionWithUpdate<KernelCorrectionMatrixComplex> kernel_correction_matrix(water_airfoil_complex);
    InteractionDynamics<KernelGradientCorrectionComplex> kernel_gradient_update(kernel_correction_matrix);
    SimpleDynamics<TimeStepInitialization> initialize_a_fluid_step(water_block);
    SimpleDynamics<NormalDirectionFromBodyShape> airfoil_normal_direction(airfoil);
    InteractionDynamics<fluid_dynamics::ViscousAccelerationWithWall> viscous_acceleration(water_airfoil_complex);
    SimpleDynamics<NormalDirectionFromBodyShape> water_block_normal_direction(water_block);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block, 0.5 / Dimensions);
    InteractionWithUpdate<FarFieldBoundary> variable_reset_in_boundary_condition(water_airfoil_complex.getInnerRelation());
    InteractionWithUpdate<fluid_dynamics::FreeSurfaceIndicationComplex> surface_indicator(water_airfoil_complex);
    InteractionDynamics<fluid_dynamics::SmearedSurfaceIndication> smeared_surface(water_airfoil_complex.getInnerRelation());
    InteractionDynamics<fluid_dynamics::VorticityInner> compute_vorticity(water_block_inner);
    //----------------------------------------------------------------------
    //	Compute the force exerted on solid body due to fluid pressure and viscosity
    //----------------------------------------------------------------------
    InteractionDynamics<solid_dynamics::PressureForceAccelerationFromFluid> pressure_force_on_solid(airfoil_water_contact);
    InteractionDynamics<solid_dynamics::ViscousForceFromFluid> viscous_force_on_solid(airfoil_water_contact);
    InteractionDynamics<solid_dynamics::AllForceAccelerationFromFluid> fluid_force_on_solid_update(airfoil_water_contact, viscous_force_on_solid);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_viscous_force_on_inserted_body(io_environment, viscous_force_on_solid, "TotalViscousForceOnSolid");
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_pressure_force_on_inserted_body(io_environment, pressure_force_on_solid, "TotalPressureForceOnSolid");
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_force_on_inserted_body(io_environment, fluid_force_on_solid_update, "TotalForceOnSolid");
    ReducedQuantityRecording<MaximumSpeed> write_maximum_speed(io_environment, water_block);
    ObservedQuantityRecording<Real>
        write_recorded_water_pressure("Pressure", io_environment, fluid_observer_contact);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    airfoil_normal_direction.exec();
    surface_indicator.exec();
    smeared_surface.exec();
    water_block_normal_direction.exec();
    variable_reset_in_boundary_condition.exec();
    kernel_correction_matrix.exec();
    kernel_gradient_update.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = 0;
    int screen_output_interval = 1000;
    Real end_time = 15;
    Real output_interval = 0.5; /**< time stamps for output. */
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    write_real_body_states.writeToFile(0);
    write_total_viscous_force_on_inserted_body.writeToFile(number_of_iterations);
    write_total_pressure_force_on_inserted_body.writeToFile(number_of_iterations);
    write_total_force_on_inserted_body.writeToFile(number_of_iterations);
    write_recorded_water_pressure.writeToFile(number_of_iterations);
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        while (integration_time < output_interval)
        {
            initialize_a_fluid_step.exec();
            Real dt = get_fluid_time_step_size.exec();
            viscous_acceleration.exec();
            pressure_relaxation.exec(dt);
            density_relaxation.exec(dt);

            integration_time += dt;
            GlobalStaticVariables::physical_time_ += dt;
            variable_reset_in_boundary_condition.exec();

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << GlobalStaticVariables::physical_time_
                          << "	dt = " << dt << "\n";
            }
            number_of_iterations++;
        }

        TickCount t2 = TickCount::now();

        compute_vorticity.exec();

        write_real_body_states.writeToFile();
        write_total_viscous_force_on_inserted_body.writeToFile(number_of_iterations);
        write_total_pressure_force_on_inserted_body.writeToFile(number_of_iterations);
        write_total_force_on_inserted_body.writeToFile(number_of_iterations);
        write_maximum_speed.writeToFile(number_of_iterations);
        write_recorded_water_pressure.writeToFile(number_of_iterations);
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }

    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

    return 0;
}
