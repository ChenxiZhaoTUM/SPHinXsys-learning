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
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real D = 1.0;
Real DL1 = 25 * D;
Real DL = 50 * D;
Real DH = 27.5 * D;
Real resolution_ref = D / 10.0; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(0.0, -DH), Vec2d(DL1 + DL, DH));
Vec2d cylinder01_center(25.0 * D, 2.0 * D); /**< Location of the cylinder center. */
Vec2d cylinder02_center(25.0 * D, -2.0 * D); /**< Location of the cylinder center. */
Real cylinder_radius = D / 2.0;
//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho0_f = 1.0;                                       /**< Density. */
Real U_f = 1.0;                                          /**< freestream velocity. */
Real c_f = 10.0 * U_f;                                   /**< Speed of sound. */
Real Re = 100.0;                                         /**< Reynolds number. */
Real mu_f = rho0_f * U_f * D / Re;                       /**< Dynamics viscosity. */
//----------------------------------------------------------------------
//	Define geometries and body shapes
//----------------------------------------------------------------------
class Cylinder01 : public MultiPolygonShape
{
  public:
    explicit Cylinder01(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        /** Geometry definition. */
        multi_polygon_.addACircle(cylinder01_center, cylinder_radius, 100, ShapeBooleanOps::add);
    }
};

class Cylinder02 : public MultiPolygonShape
{
  public:
    explicit Cylinder02(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        /** Geometry definition. */
        multi_polygon_.addACircle(cylinder02_center, cylinder_radius, 100, ShapeBooleanOps::add);
    }
};


std::vector<Vecd> createWaterBlockShape()
{
    std::vector<Vecd> water_block_shape;
    water_block_shape.push_back(Vecd(0.0, -DH));
    water_block_shape.push_back(Vecd(0.0, DH));
    water_block_shape.push_back(Vecd(DL1 + DL, DH));
    water_block_shape.push_back(Vecd(DL1 + DL, -DH));
    water_block_shape.push_back(Vecd(0.0, -DH));

    return water_block_shape;
}

class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        MultiPolygon outer_boundary(createWaterBlockShape());
        add<MultiPolygonShape>(outer_boundary, "OuterBoundary");
        Cylinder01 cylinder01("inner01");
        subtract<Cylinder01>(cylinder01);
        Cylinder02 cylinder02("inner02");
        subtract<Cylinder02>(cylinder02);
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
    sph_system.setRunParticleRelaxation(false);
    // Tag for computation start with relaxed body fitted particles distribution.
    sph_system.setReloadParticles(true);
    // Handle command line arguments and override the tags for particle relaxation and reload.
    sph_system.handleCommandlineOptions(ac, av);
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBody cylinder01(sph_system, makeShared<Cylinder01>("cylinder01"));
    cylinder01.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    cylinder01.defineParticlesAndMaterial<SolidParticles, Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? cylinder01.generateParticles<ParticleGeneratorReload>(io_environment, cylinder01.getName())
        : cylinder01.generateParticles<ParticleGeneratorLattice>();
    cylinder01.addBodyStateForRecording<Real>("Density");

    SolidBody cylinder02(sph_system, makeShared<Cylinder02>("cylinder02"));
    cylinder02.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    cylinder02.defineParticlesAndMaterial<SolidParticles, Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? cylinder02.generateParticles<ParticleGeneratorReload>(io_environment, cylinder02.getName())
        : cylinder02.generateParticles<ParticleGeneratorLattice>();
    cylinder02.addBodyStateForRecording<Real>("Density");

    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    water_block.defineComponentLevelSetShape("OuterBoundary");
    water_block.defineParticlesAndMaterial<BaseParticles, WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? water_block.generateParticles<ParticleGeneratorReload>(io_environment, water_block.getName())
        : water_block.generateParticles<ParticleGeneratorLattice>();
    water_block.addBodyStateForRecording<Real>("Pressure");
    water_block.addBodyStateForRecording<Real>("Density");

    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //	Note that the same relation should be defined only once.
    //----------------------------------------------------------------------
    InnerRelation water_block_inner(water_block);
    ComplexRelation water_cylinder_complex(water_block, RealBodyVector{&cylinder01, &cylinder02});
    ContactRelation cylinder01_water_contact(cylinder01, {&water_block});
    ContactRelation cylinder02_water_contact(cylinder02, {&water_block});
    
    //----------------------------------------------------------------------
    //	Run particle relaxation for body-fitted distribution if chosen.
    //----------------------------------------------------------------------
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation cylinder01_inner(cylinder01); // extra body topology only for particle relaxation
        InnerRelation cylinder02_inner(cylinder02);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        SimpleDynamics<RandomizeParticlePosition> random_cylinder01_particles(cylinder01);
        SimpleDynamics<RandomizeParticlePosition> random_cylinder02_particles(cylinder02);
        SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
        
        relax_dynamics::RelaxationStepInner relaxation_step_inner_cylinder01(cylinder01_inner, true);
        relax_dynamics::RelaxationStepInner relaxation_step_inner_cylinder02(cylinder02_inner, true);
        relax_dynamics::RelaxationStepComplex relaxation_step_complex(water_cylinder_complex, "OuterBoundary", true);

        BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
        ReloadParticleIO write_real_body_particle_reload_files(io_environment, sph_system.real_bodies_);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_cylinder01_particles.exec(0.25);
        random_cylinder02_particles.exec(0.25);
        random_water_particles.exec(0.25);
        relaxation_step_inner_cylinder01.SurfaceBounding().exec();
        relaxation_step_inner_cylinder02.SurfaceBounding().exec();
        relaxation_step_complex.SurfaceBounding().exec();

        cylinder01.updateCellLinkedList();
        cylinder02.updateCellLinkedList();
        water_block.updateCellLinkedList();

        write_real_body_states.writeToFile(0);

        int ite_p = 0;
        while (ite_p < 1000)
        {
            relaxation_step_inner_cylinder01.exec();
            relaxation_step_inner_cylinder02.exec();
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
    InteractionWithUpdate<fluid_dynamics::EulerianIntegration1stHalfAcousticRiemannWithWall> pressure_relaxation(water_cylinder_complex);
    InteractionWithUpdate<fluid_dynamics::EulerianIntegration2ndHalfAcousticRiemannWithWall> density_relaxation(water_cylinder_complex);
    InteractionWithUpdate<KernelCorrectionMatrixComplex> kernel_correction_matrix(water_cylinder_complex);
    InteractionDynamics<KernelGradientCorrectionComplex> kernel_gradient_update(kernel_correction_matrix);
    SimpleDynamics<TimeStepInitialization> initialize_a_fluid_step(water_block);
    SimpleDynamics<NormalDirectionFromBodyShape> cylinder01_normal_direction(cylinder01);
    SimpleDynamics<NormalDirectionFromBodyShape> cylinder02_normal_direction(cylinder02);
    InteractionDynamics<fluid_dynamics::ViscousAccelerationWithWall> viscous_acceleration(water_cylinder_complex);
    SimpleDynamics<NormalDirectionFromBodyShape> water_block_normal_direction(water_block);
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block, 0.5 / Dimensions);
    InteractionWithUpdate<FarFieldBoundary> variable_reset_in_boundary_condition(water_cylinder_complex.getInnerRelation());
    InteractionWithUpdate<fluid_dynamics::FreeSurfaceIndicationComplex> surface_indicator(water_cylinder_complex);
    InteractionDynamics<fluid_dynamics::SmearedSurfaceIndication> smeared_surface(water_cylinder_complex.getInnerRelation());
    InteractionDynamics<fluid_dynamics::VorticityInner> compute_vorticity(water_block_inner);
    //----------------------------------------------------------------------
    //	Compute the force exerted on solid body due to fluid pressure and viscosity
    //----------------------------------------------------------------------
    InteractionDynamics<solid_dynamics::PressureForceAccelerationFromFluid> pressure_force_on_cylinder01(cylinder01_water_contact);
    InteractionDynamics<solid_dynamics::PressureForceAccelerationFromFluid> pressure_force_on_cylinder02(cylinder02_water_contact);
    InteractionDynamics<solid_dynamics::ViscousForceFromFluid> viscous_force_on_cylinder01(cylinder01_water_contact);
    InteractionDynamics<solid_dynamics::ViscousForceFromFluid> viscous_force_on_cylinder02(cylinder02_water_contact);
    InteractionDynamics<solid_dynamics::AllForceAccelerationFromFluid> fluid_force_on_cylinder01_update(cylinder01_water_contact, viscous_force_on_cylinder01);
    InteractionDynamics<solid_dynamics::AllForceAccelerationFromFluid> fluid_force_on_cylinder02_update(cylinder02_water_contact, viscous_force_on_cylinder02);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_viscous_force_on_cylinder01(io_environment, viscous_force_on_cylinder01, "TotalViscousForceOnCylinder01");
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_pressure_force_on_cylinder01(io_environment, pressure_force_on_cylinder01, "TotalPressureForceOnCylinder01");
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_force_on_cylinder01(io_environment, fluid_force_on_cylinder01_update, "TotalForceOnCylinder01");
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_viscous_force_on_cylinder02(io_environment, viscous_force_on_cylinder02, "TotalViscousForceOnCylinder02");
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_pressure_force_on_cylinder02(io_environment, pressure_force_on_cylinder02, "TotalPressureForceOnCylinder02");
    ReducedQuantityRecording<solid_dynamics::TotalForceFromFluid>
        write_total_force_on_cylinder02(io_environment, fluid_force_on_cylinder02_update, "TotalForceOnCylinder02");
    ReducedQuantityRecording<MaximumSpeed> write_maximum_speed(io_environment, water_block);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    cylinder01_normal_direction.exec();
    cylinder02_normal_direction.exec();
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
    Real end_time = 250;
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
    write_total_viscous_force_on_cylinder01.writeToFile(number_of_iterations);
    write_total_viscous_force_on_cylinder02.writeToFile(number_of_iterations);
    write_total_pressure_force_on_cylinder01.writeToFile(number_of_iterations);
    write_total_pressure_force_on_cylinder02.writeToFile(number_of_iterations);
    write_total_force_on_cylinder01.writeToFile(number_of_iterations);
    write_total_force_on_cylinder02.writeToFile(number_of_iterations);
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

            write_total_viscous_force_on_cylinder01.writeToFile(number_of_iterations);
            write_total_viscous_force_on_cylinder02.writeToFile(number_of_iterations);
            write_total_pressure_force_on_cylinder01.writeToFile(number_of_iterations);
            write_total_pressure_force_on_cylinder02.writeToFile(number_of_iterations);
            write_total_force_on_cylinder01.writeToFile(number_of_iterations);
            write_total_force_on_cylinder02.writeToFile(number_of_iterations);
        }

        TickCount t2 = TickCount::now();

        compute_vorticity.exec();

        write_real_body_states.writeToFile();
        write_maximum_speed.writeToFile(number_of_iterations);

        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }

    TickCount t4 = TickCount::now();
    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

    return 0;
}
