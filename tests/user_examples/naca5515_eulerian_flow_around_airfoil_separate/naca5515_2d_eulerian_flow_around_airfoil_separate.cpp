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
std::string airfoil = "./input/NACA5515_5deg.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0;
Real DL = 5 * L;
Real DL1 = 2 * L;
Real DH = 3 * L;
Real resolution_ref = 0.004; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(-DL1, -DH), Vec2d(DL, DH));
//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho0_f = 1.0;                                       /**< Density. */
Real U_f = 1.0;                                          /**< freestream velocity. */
Real c_f = 10.0 * U_f;                                   /**< Speed of sound. */
Real Re = 420.0;                                         /**< Reynolds number. */
Real mu_f = rho0_f * U_f * L / Re;       /**< Dynamics viscosity. */
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

Vec2d waterblock_halfsize = Vec2d(0.5 * (DL + DL1), DH);
Vec2d waterblock_translation = Vec2d(0.5, 0.0);

class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addABox(Transform(waterblock_translation), waterblock_halfsize, ShapeBooleanOps::add);
        multi_polygon_.addAPolygonFromFile(airfoil, ShapeBooleanOps::sub);
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

class PressureObserverParticleGeneratorWing : public ObserverParticleGenerator
{
  public:
    explicit PressureObserverParticleGeneratorWing(SPHBody &sph_body) : ObserverParticleGenerator(sph_body)
    {
        std::fstream dataFile(airfoil);
        Vecd temp_point;
        Real temp1 = 0.0, temp2 = 0.0;
        if (dataFile.fail())
        {
            std::cout << "File can not open.\n"
                        << std::endl;
            ;
        }

        while (!dataFile.fail() && !dataFile.eof())
        {
            dataFile >> temp1 >> temp2;
            temp_point[0] = temp1;
            temp_point[1] = temp2;
            positions_.push_back(temp_point);
        }
        dataFile.close();
    }
};

class OutputObserverPositionAndPressure : public BaseIO,
                                            public ObservingAQuantity<Real>
{
public:
    OutputObserverPositionAndPressure(IOEnvironment& io_environment, BaseContactRelation& contact_relation, const std::string &airfoil_body_name)
        : BaseIO(io_environment),
        ObservingAQuantity<Real>(contact_relation, "Pressure"),
        observer_(contact_relation.getSPHBody()),
        base_particles_(observer_.getBaseParticles()),
        airfoil_body_name_(airfoil_body_name){};

    void writeToFile(size_t iteration_step = 0) override
    {
        this->exec();
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + "observer_position_and_pressure_ " + airfoil_body_name_ + convertPhysicalTimeToString(GlobalStaticVariables::physical_time_) + ".dat";
	    std::ofstream out_file(filefullpath.c_str(), std::ios::app);

        for (size_t i = 0; i != base_particles_.total_real_particles_; ++i)
        {
            out_file << "particle " << i << ": Position " << base_particles_.pos_[i][0] << " " << base_particles_.pos_[i][1] 
                << " Pressure " <<  (*this->interpolated_quantities_)[i] << " " << std::endl;
        }
        out_file.close();
    }

protected:
    SPHBody &observer_;
    BaseParticles &base_particles_;
    std::string airfoil_body_name_;

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
    sph_system.setRunParticleRelaxation(false);
    // Tag for computation start with relaxed body fitted particles distribution.
    sph_system.setReloadParticles(true);
    // Handle command line arguments and override the tags for particle relaxation and reload.
    sph_system.handleCommandlineOptions(ac, av);
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBody airfoil(sph_system, makeShared<AirfoilModel>("Airfoil"));
    //airfoil.defineAdaptationRatios(1.15, 2.0);
    //airfoil.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    airfoil.defineBodyLevelSetShape()->cleanLevelSet(1.0)->writeLevelSet(io_environment);
    airfoil.defineParticlesAndMaterial<SolidParticles, Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? airfoil.generateParticles<ParticleGeneratorReload>(io_environment, airfoil.getName())
        : airfoil.generateParticles<ParticleGeneratorLattice>();
    airfoil.addBodyStateForRecording<Real>("Density");

    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    //water_block.defineBodyLevelSetShape()->cleanLevelSet(1.0)->writeLevelSet(io_environment);
    water_block.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    water_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? water_block.generateParticles<ParticleGeneratorReload>(io_environment, water_block.getName())
        : water_block.generateParticles<ParticleGeneratorLattice>();
    water_block.addBodyStateForRecording<Real>("Pressure");
    water_block.addBodyStateForRecording<Real>("Density");

    ObserverBody fluid_observer(sph_system, "FluidObserver");
    fluid_observer.generateParticles<ObserverParticleGenerator>(observation_location);
    
    ObserverBody wing_pressure_observer(sph_system, "WingPressureObserver");
    wing_pressure_observer.generateParticles<PressureObserverParticleGeneratorWing>();
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
    ContactRelation wing_observer_water_contact(wing_pressure_observer, { &water_block });
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
        relax_dynamics::RelaxationStepInner relaxation_step_inner_water(water_block_inner, true);
        airfoil.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");
        water_block.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");

        BodyStatesRecordingToPlt write_real_body_states(io_environment, sph_system.real_bodies_);
        ReloadParticleIO write_real_body_particle_reload_files(io_environment, sph_system.real_bodies_);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_airfoil_particles.exec(0.25);
        random_water_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_inner_water.SurfaceBounding().exec();

        airfoil.updateCellLinkedList();
        water_block.updateCellLinkedList();

        write_real_body_states.writeToFile(0);

        int ite_p = 0;
        while (ite_p < 1000)
        {
            relaxation_step_inner.exec();
            relaxation_step_inner_water.exec();
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
    BodyStatesRecordingToPlt write_real_body_states(io_environment, sph_system.real_bodies_);
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_viscous_force_on_inserted_body(io_environment, viscous_force_on_solid, "TotalViscousForceOnSolid");
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_pressure_force_on_inserted_body(io_environment, pressure_force_on_solid, "TotalPressureForceOnSolid");
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_force_on_inserted_body(io_environment, fluid_force_on_solid_update, "TotalForceOnSolid");
    ReducedQuantityRecording<MaximumSpeed> write_maximum_speed(io_environment, water_block);
    ObservedQuantityRecording<Real>
        write_recorded_water_pressure("Pressure", io_environment, fluid_observer_contact);
    OutputObserverPositionAndPressure write_wing_pressure(io_environment, wing_observer_water_contact, "Wing");
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

    write_wing_pressure.writeToFile(number_of_iterations);

    TickCount t4 = TickCount::now();
    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

    return 0;
}
