/**
 * @file 	2d_eulerian_flow_around_cylinder_LG.cpp
 * @brief 	This is the test file for the weakly compressible viscous flow around a cylinder coupling with the Laguerre Gauss kernel.
 * @details We consider a Eulerian flow passing by a cylinder in 2D.
 * @author 	Zhentong Wang and Xiangyu Hu
 */
#include "sphinxsys.h"
#include "kernel_summation.h"
#include "kernel_summation.hpp"

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
    explicit WaterBlock(const std::string& shape_name) : ComplexShape(shape_name)
    {
        MultiPolygon outer_boundary(createWaterBlockShape());
        add<MultiPolygonShape>(outer_boundary, "OuterBoundary");
        AirfoilModel import_model("InnerBody");
        subtract<AirfoilModel>(import_model);
    }
};

class WaterOuter : public ComplexShape
{
  public:
    explicit WaterOuter(const std::string &shape_name) : ComplexShape(shape_name)
    {
        MultiPolygon outer_boundary(createWaterBlockShape());
        add<MultiPolygonShape>(outer_boundary, "OuterBoundaryShape");
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

class WingPressureObserver;
template <>
class ParticleGenerator<WingPressureObserver> : public ParticleGenerator<Observer>
{
  public:
    explicit ParticleGenerator(SPHBody &sph_body) : ParticleGenerator<Observer>(sph_body)
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

class OutputObserverPositionAndPressure : public BodyStatesRecording, public ObservingAQuantity<Real>
{
public:
    OutputObserverPositionAndPressure(BaseContactRelation& contact_relation)
        : BodyStatesRecording(contact_relation.getSPHBody()),
        ObservingAQuantity<Real>(contact_relation, "Pressure"),
        observer_(contact_relation.getSPHBody()),
        base_particles_(observer_.getBaseParticles()),
        pos_(*particles_->getVariableByName<Vecd>("Position")) {};
    virtual ~OutputObserverPositionAndPressure() {};

    virtual void writeWithFileName(const std::string &sequence) override
    {
        this->exec();
        std::string output_folder = "./output";
        std::string filefullpath = output_folder + "/" + "observer_position_and_pressure_ " + convertPhysicalTimeToString(GlobalStaticVariables::physical_time_) + ".dat";
        std::ofstream out_file(filefullpath.c_str(), std::ios::app);

        for (size_t i = 0; i != base_particles_.total_real_particles_; ++i)
        {
            out_file << "particle " << i << ": Position " << pos_[i][0] << " " << pos_[i][1]
                << " Pressure " << (*this->interpolated_quantities_)[i] << " " << std::endl;
        }
        out_file.close();
    }

protected:
    SPHBody& observer_;
    BaseParticles& base_particles_;
    std::string airfoil_body_name_;
    StdLargeVec<Vecd>& pos_;
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
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBody airfoil(sph_system, makeShared<AirfoilModel>("AirfoilNACA5515"));
    airfoil.defineBodyLevelSetShape()->cleanLevelSet(1.0)->writeLevelSet(sph_system);
    airfoil.defineMaterial<Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? airfoil.generateParticles<BaseParticles, Reload>(airfoil.getName())
        : airfoil.generateParticles<BaseParticles, Lattice>();
    airfoil.addBodyStateForRecording<Real>("Density");

    InverseShape<AirfoilModel> inversed_import("InversedAirfoil");
    LevelSetShape inversed_import_level_set(inversed_import, makeShared<SPHAdaptation>(resolution_ref));
    inversed_import_level_set.cleanLevelSet(1.0);
    WaterOuter water_shape("WaterShape");
    water_shape.initializeComponentLevelSetShapesByAdaptation(makeShared<SPHAdaptation>(resolution_ref), sph_system);
    water_shape.addAnLevelSetShape(&inversed_import_level_set);
       
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    water_block.defineComponentLevelSetShape("OuterBoundary");
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? water_block.generateParticles<BaseParticles, Reload>(water_block.getName())
        : water_block.generateParticles<BaseParticles, Lattice>();
    water_block.addBodyStateForRecording<Real>("Density");

    ObserverBody fluid_observer(sph_system, "FluidObserver");
    fluid_observer.generateParticles<BaseParticles, Observer>(observation_location);

    ObserverBody wing_pressure_observer(sph_system, "WingPressureObserver");
    wing_pressure_observer.generateParticles<BaseParticles, WingPressureObserver>();
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //	Note that the same relation should be defined only once.
    //----------------------------------------------------------------------
    InnerRelation water_block_inner(water_block);
    InnerRelation airfoil_inner(airfoil);
    ContactRelation water_block_contact(water_block, {&airfoil});
    ContactRelation airfoil_contact(airfoil, {&water_block});
    ComplexRelation water_wall_complex(water_block_inner, water_block_contact);
    ComplexRelation airfoil_water_complex(airfoil_inner, airfoil_contact);
    ContactRelation fluid_observer_contact(fluid_observer, {&water_block});
    ContactRelation wing_observer_water_contact(wing_pressure_observer, { &water_block });
    //----------------------------------------------------------------------
    //----------------------------------------------------------------------
    //	Run particle relaxation for body-fitted distribution if chosen.
    //----------------------------------------------------------------------
    if (sph_system.RunParticleRelaxation())
    {
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_inserted_body_particles(airfoil);
        SimpleDynamics<RandomizeParticlePosition> random_water_body_particles(water_block);
        //BodyStatesRecordingToVtp write_real_body_states(sph_system.real_bodies_);
        BodyStatesRecordingToPlt write_real_body_states(sph_system.real_bodies_);
        ReloadParticleIO write_real_body_particle_reload_files(sph_system.real_bodies_);
        RelaxationStepLevelSetCorrectionInner relaxation_step_inner(airfoil_inner);
        RelaxationStepWithComplexBounding relaxation_step_inner_water(water_block_inner, water_shape);

        InteractionDynamics<NablaWVLevelSetCorrectionInner> solid_zero_order_consistency(airfoil_inner);
        InteractionDynamics<NablaWVLevelSetCorrectionComplex> fluid_zero_order_consistency(ConstructorArgs(water_block_inner, std::string("OuterBoundary")), water_block_contact);
        airfoil.addBodyStateForRecording<Vecd>("KernelSummation");
        water_block.addBodyStateForRecording<Vecd>("KernelSummation");

        SimpleDynamics<NormalDirectionFromBodyShape> solid_normal_direction(airfoil);
        InteractionWithUpdate<FluidSurfaceIndicationByDistance> fluid_surface_indicator(water_block_contact);
        water_block.addBodyStateForRecording<int>("FluidContactIndicator");
        ReducedQuantityRecording<SurfaceKineticEnergy> write_water_surface_kinetic_energy(water_block);
        water_block.addBodyStateForRecording<Real>("ParticleEnergy");
        AvgSurfaceKineticEnergy write_average_surface_kinetic_energy(water_block);

        InteractionDynamics<LocalDisorderMeasure> local_disorder_measure(water_block_inner);
        water_block.addBodyStateForRecording<Real>("FirstDistance");
        water_block.addBodyStateForRecording<Real>("SecondDistance");
        water_block.addBodyStateForRecording<Real>("LocalDisorderMeasureParameter");
        GlobalDisorderMeasure write_global_disorder_measure(water_block);
    
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_inserted_body_particles.exec(0.25);
        random_water_body_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        relaxation_step_inner_water.SurfaceBounding().exec();
        airfoil.updateCellLinkedList();
        water_block.updateCellLinkedList();
        airfoil_water_complex.updateConfiguration();
        water_wall_complex.updateConfiguration();
        //----------------------------------------------------------------------
        //	First output before the simulation.
        //----------------------------------------------------------------------
        solid_zero_order_consistency.exec();
        fluid_zero_order_consistency.exec();
        solid_normal_direction.exec();
        fluid_surface_indicator.exec();
        local_disorder_measure.exec();
        write_water_surface_kinetic_energy.writeToFile();
        write_average_surface_kinetic_energy.writeToFile();
        write_global_disorder_measure.writeToFile();
        write_real_body_states.writeToFile();

        int ite_p = 0;
        TickCount t1 = TickCount::now();
        while (ite_p < 2000)
        {
            relaxation_step_inner.exec();
            relaxation_step_inner_water.exec();

            solid_normal_direction.exec();
            fluid_surface_indicator.exec();
            local_disorder_measure.exec();

            ite_p += 1;
            if (ite_p % 500 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
                write_real_body_states.writeToFile(ite_p);
            }

            airfoil_water_complex.updateConfiguration();
            water_wall_complex.updateConfiguration();

            write_water_surface_kinetic_energy.writeToFile(ite_p);
            write_average_surface_kinetic_energy.writeToFile(ite_p);
            write_global_disorder_measure.writeToFile(ite_p);
        }
        std::cout << "The physics relaxation process finish !" << std::endl;

        TickCount t4 = TickCount::now();
        TimeInterval tt;
        tt = t4 - t1;
        std::cout << "Total time for computation: " << tt.seconds() << " seconds." << std::endl;

        solid_zero_order_consistency.exec();
        fluid_zero_order_consistency.exec();
        write_real_body_states.writeToFile(ite_p);

        write_real_body_particle_reload_files.writeToFile(0);

        return 0;
    }
    //----------------------------------------------------------------------
    //	Define the main numerical methods used in the simulation.
    //	Note that there may be data dependence on the constructors of these methods.
    //----------------------------------------------------------------------
    InteractionDynamics<NablaWVLevelSetCorrectionInner> solid_zero_order_consistency(airfoil_inner);
    InteractionDynamics<NablaWVLevelSetCorrectionComplex> fluid_zero_order_consistency(ConstructorArgs(water_block_inner, std::string("OuterBoundary")), water_block_contact);
    //InteractionDynamics<NablaWVComplex> fluid_zero_order_consistency(water_inner, water_import_contact);
    airfoil.addBodyStateForRecording<Vecd>("KernelSummation");
    water_block.addBodyStateForRecording<Vecd>("KernelSummation");

    SimpleDynamics<NormalDirectionFromBodyShape> airfoil_normal_direction(airfoil);
    InteractionWithUpdate<FreeSurfaceIndicationComplex> surface_indicator(water_block_inner, water_block_contact);
    InteractionDynamics<SmearedSurfaceIndication> smeared_surface(water_block_inner);
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> airfoil_kernel_correction_matrix(airfoil_inner, airfoil_contact);
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> water_block_kernel_correction_matrix(water_block_inner, water_block_contact);
    InteractionDynamics<KernelGradientCorrectionComplex> kernel_gradient_update(water_block_inner, water_block_contact);
    
    InteractionWithUpdate<fluid_dynamics::EulerianIntegration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::EulerianIntegration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_block_contact);

    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_force(water_block_inner, water_block_contact);
    SimpleDynamics<NormalDirectionFromBodyShape> water_block_normal_direction(water_block);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block, 0.5);
    InteractionWithUpdate<FarFieldBoundary> variable_reset_in_boundary_condition(water_block_inner);
    //----------------------------------------------------------------------
    //	Compute the force exerted on solid body due to fluid pressure and viscosity
    //----------------------------------------------------------------------
    InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_from_fluid(airfoil_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_from_fluid(airfoil_contact);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    water_block.addBodyStateForRecording<Vecd>("Momentum");
    water_block.addBodyStateForRecording<Vecd>("MomentumChangeRate");
    water_block.addBodyStateForRecording<int>("Indicator");
    //BodyStatesRecordingToVtp write_real_body_states(sph_system.real_bodies_);
    BodyStatesRecordingToPlt write_real_body_states(sph_system.real_bodies_);


    //RegressionTestDynamicTimeWarping<ReducedQuantityRecording<QuantitySummation<Vecd>>>
    //    write_total_viscous_force_from_fluid(airfoil, "ViscousForceFromFluid");
    //ReducedQuantityRecording<QuantitySummation<Vecd>>
    //    write_total_pressure_force_from_fluid_body(airfoil, "PressureForceFromFluid");
    //ReducedQuantityRecording<MaximumSpeed> write_maximum_speed(water_block);

    ObservedQuantityRecording<Real>
        write_recorded_water_pressure("Pressure", fluid_observer_contact);
    OutputObserverPositionAndPressure write_wing_pressure(wing_observer_water_contact);
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
    airfoil_kernel_correction_matrix.exec();
    water_block_kernel_correction_matrix.exec();
    kernel_gradient_update.exec();

    solid_zero_order_consistency.exec();
    fluid_zero_order_consistency.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    size_t number_of_iterations = 0;
    int screen_output_interval = 1000;
    Real end_time = 15.0;
    Real output_interval = 0.5; /**< time stamps for output. */
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    write_real_body_states.writeToFile(number_of_iterations);
    write_recorded_water_pressure.writeToFile(number_of_iterations);
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        while (integration_time < output_interval)
        {
            Real dt = get_fluid_time_step_size.exec();
            viscous_force.exec();
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

        solid_zero_order_consistency.exec();
        fluid_zero_order_consistency.exec();
        write_real_body_states.writeToFile();

        viscous_force_from_fluid.exec();
        pressure_force_from_fluid.exec();
        //write_total_viscous_force_from_fluid.writeToFile(number_of_iterations);
        //write_total_pressure_force_from_fluid_body.writeToFile(number_of_iterations);
        //write_maximum_speed.writeToFile(number_of_iterations);

        write_recorded_water_pressure.writeToFile(number_of_iterations);
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }

    if (GlobalStaticVariables::physical_time_ > 14.0)
    {
        write_wing_pressure.writeToFile(number_of_iterations);
        solid_zero_order_consistency.exec();
        fluid_zero_order_consistency.exec();
        write_real_body_states.writeToFile(number_of_iterations);
    }

    TickCount t4 = TickCount::now();

    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

    //write_total_viscous_force_from_fluid.testResult();

    return 0;
}
