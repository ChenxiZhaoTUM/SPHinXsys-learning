/**
 * @file 	elastic_gate.cpp
 * @brief 	2D elastic gate deformation due to dam break force.
 * @details This is the one of the basic test cases, also the first case for
 * 			understanding SPH method for fluid-structure-interaction (FSI) simulation.
 * @author 	Luhui Han, Chi Zhang and Xiangyu Hu
 */
#include "sphinxsys.h" //	SPHinXsys Library.
using namespace SPH;   //	Namespace cite here.
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 3.5 - 2.0;                        /**< Tank length. */
Real DH = 2.5 - 1.0;                        /**< Tank height. */
Real Dam_H = 1.35 - 1.0;                     /**< Water block height. */
Real Gate_width = 0.0095;                  /**< Width of the gate. */
Real Gate_length = 0.570;
Real Gate_height = 1.20 - 0.85;
Real Gate_xDis = 1.0 - 0.5;

Real resolution_ref = Gate_width / 2.0; /**< Initial reference particle spacing. */
Real BW = resolution_ref * 4.0;         /**< Extending width for BCs. */
/** The offset that the rubber gate shifted above the tank. */
Real dp_s = 0.5 * resolution_ref;

/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vec2d(-BW, -BW), Vec2d(DL + BW, DH + BW));

// observer location
StdVec<Vecd> observation_location = {Vecd(Gate_xDis + 0.3102, Gate_height + (Gate_width / 4 + 0.3102) * tan(10.0/360.0 * 2 * Pi))};
//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real rho0_f = 1000.0;                         /**< Reference density of fluid. */
Real mu_f = 1.0E-6;
Real U_f = 5.0;                            /**< Characteristic velocity. */
Real c_f = 10 * U_f; /**< Reference sound speed. */
//----------------------------------------------------------------------
//	Material parameters of the elastic gate.
//----------------------------------------------------------------------
Real rho0_s = 7000;   /**< Reference density of gate. */
Real poisson = 0.48; /**< Poisson ratio. */
Real Youngs_modulus = 0.8E10;
//Real Youngs_modulus = 1.6E10;
//Real gravity_g = 9.81;
//----------------------------------------------------------------------
//	Cases-dependent geometries
//----------------------------------------------------------------------
class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        /** Geometry definition. */
        std::vector<Vecd> water_block_shape;
        water_block_shape.push_back(Vecd(0.0, 0.0));
        water_block_shape.push_back(Vecd(0.0, Dam_H));
        water_block_shape.push_back(Vecd(DL, Dam_H));
        water_block_shape.push_back(Vecd(DL, 0.0));
        water_block_shape.push_back(Vecd(0.0, 0.0));
        multi_polygon_.addAPolygon(water_block_shape, ShapeBooleanOps::add);
    }
};
//----------------------------------------------------------------------
//	Wall cases-dependent geometries.
//----------------------------------------------------------------------
class WallBoundary : public MultiPolygonShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        /** Geometry definition. */
        std::vector<Vecd> outer_wall_shape;
        outer_wall_shape.push_back(Vecd(-BW, -BW));
        outer_wall_shape.push_back(Vecd(-BW, DH + BW));
        outer_wall_shape.push_back(Vecd(DL + BW, DH + BW));
        outer_wall_shape.push_back(Vecd(DL + BW, -BW));
        outer_wall_shape.push_back(Vecd(-BW, -BW));

        std::vector<Vecd> inner_wall_shape;
        inner_wall_shape.push_back(Vecd(0.0, 0.0));
        inner_wall_shape.push_back(Vecd(0.0, DH));
        inner_wall_shape.push_back(Vecd(DL, DH));
        inner_wall_shape.push_back(Vecd(DL, 0.0));
        inner_wall_shape.push_back(Vecd(0.0, 0.0));

        multi_polygon_.addAPolygon(outer_wall_shape, ShapeBooleanOps::add);
        multi_polygon_.addAPolygon(inner_wall_shape, ShapeBooleanOps::sub);
    }
};
//----------------------------------------------------------------------
//	create a gate shape
//----------------------------------------------------------------------
class Panel : public MultiPolygonShape
{
  public:
    explicit Panel(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        std::vector<Vecd> gate_shape;
        gate_shape.push_back(Vecd(Gate_xDis, Gate_height));
        gate_shape.push_back(Vecd(Gate_xDis + Gate_length, Gate_height));
        gate_shape.push_back(Vecd(Gate_xDis + Gate_length, Gate_height + Gate_width));
        gate_shape.push_back(Vecd(Gate_xDis, Gate_height + Gate_width));
        gate_shape.push_back(Vecd(Gate_xDis, Gate_height));

        multi_polygon_.addAPolygon(gate_shape, ShapeBooleanOps::add);
    }
};

class InitialVelocityCondition : public BaseLocalDynamics<SPHBody>
{
  private:
    Vec2d *vel_;
    Vec2d initial_velocity_;

  public:
    InitialVelocityCondition(SPHBody &body, Vec2d initial_velocity)
        : BaseLocalDynamics<SPHBody>(body),
          vel_(this->particles_->template registerStateVariable<Vec2d>("Velocity")),
          initial_velocity_(std::move(initial_velocity)){};
    inline void update(size_t index_i, [[maybe_unused]] Real dt = 0.0)
    {
        vel_[index_i] = initial_velocity_;
    }
};

//class DisControlGeometry : public BodyPartByParticle
//{
//  public:
//    DisControlGeometry(SPHBody &body, const std::string &body_part_name)
//        : BodyPartByParticle(body, body_part_name)
//    {
//        TaggingParticleMethod tagging_particle_method = std::bind(&DisControlGeometry::tagManually, this, _1);
//        tagParticles(tagging_particle_method);
//    };
//    virtual ~DisControlGeometry(){};
//
//  private:
//    void tagManually(size_t index_i)
//    {
//        Vecd pos_before_rotation = Rotation2d(-10.0/360.0*2*Pi) * base_particles_.ParticlePositions()[index_i];
//        if (pos_before_rotation[0] < Gate_xDis + 25 * 0.001 && pos_before_rotation[0] > Gate_xDis + (25 + 495) * 0.001)
//        {
//            body_part_particles_.push_back(index_i);
//        }
//    };
//};
//
//class ControlDisplacement : public thin_structure_dynamics::ConstrainShellBodyRegion
//{
//  public:
//    ControlDisplacement(BodyPartByParticle &body_part)
//        : ConstrainShellBodyRegion(body_part),
//          angular_vel_(particles_->getVariableDataByName<Vecd>("AngularVelocity")){};
//    virtual ~ControlDisplacement(){};
//
//  protected:
//    Vecd *angular_vel_;
//
//    void update(size_t index_i, Real dt = 0.0)
//    {
//        angular_vel_[index_i] = Vecd::Zero();
//    };
//};
//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    water_block.defineClosure<WeaklyCompressibleFluid, Viscosity>(ConstructArgs(rho0_f, c_f), mu_f);
    water_block.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineMaterial<Solid>();
    wall_boundary.generateParticles<BaseParticles, Lattice>();

    SolidBody gate(sph_system, makeShared<Panel>("Panel"));
    gate.defineAdaptationRatios(1.15, 2.0);
    gate.defineMaterial<SaintVenantKirchhoffSolid>(rho0_s, Youngs_modulus, poisson);
    gate.generateParticles<BaseParticles, Lattice>();

    ObserverBody gate_observer(sph_system, "Observer");
    gate_observer.defineAdaptationRatios(1.15, 2.0);
    gate_observer.generateParticles<ObserverParticles>(observation_location);
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation water_block_inner(water_block);
    ContactRelation water_block_contact(water_block, RealBodyVector{&wall_boundary, &gate});
    InnerRelation gate_inner(gate);
    ContactRelation gate_water_contact(gate, {&water_block});
    ContactRelation gate_observer_contact(gate_observer, {&gate});
    //----------------------------------------------------------------------
    // Combined relations built from basic relations
    //----------------------------------------------------------------------
    ComplexRelation water_block_complex(water_block_inner, water_block_contact);
    //----------------------------------------------------------------------
    // Define the numerical methods used in the simulation.
    // Note that there may be data dependence on the sequence of constructions.
    // Generally, the geometric models or simple objects without data dependencies,
    // such as gravity, should be initiated first.
    // Then the major physical particle dynamics model should be introduced.
    // Finally, the auxiliary models such as time step estimator, initial condition,
    // boundary condition and other constraints should be defined.
    // For typical fluid-structure interaction, we first define structure dynamics,
    // Then fluid dynamics and the corresponding coupling dynamics.
    // The coupling with multi-body dynamics will be introduced at last.
    //----------------------------------------------------------------------
    Transform transform2d(Rotation2d(10.0/360.0*2*Pi));
    SimpleDynamics<TranslationAndRotation> gate_offset_position(gate, transform2d);
    /*Gravity gravity(Vecd(0.0, -gravity_g));
    SimpleDynamics<GravityForce<Gravity>> constant_gravity(gate, gravity);*/
    SimpleDynamics<InitialVelocityCondition> panel_velocity(gate, Vec2d(0.0, -4.0));
    InteractionWithUpdate<LinearGradientCorrectionMatrixInner> gate_corrected_configuration(gate_inner); 
    SimpleDynamics<NormalDirectionFromBodyShape> gate_normal_direction(gate);
    Dynamics1Level<solid_dynamics::Integration1stHalfPK2> gate_stress_relaxation_first_half(gate_inner);
    Dynamics1Level<solid_dynamics::Integration2ndHalf> gate_stress_relaxation_second_half(gate_inner);
    ReduceDynamics<solid_dynamics::AcousticTimeStep> gate_computing_time_step_size(gate);
    SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> gate_update_normal(gate);

    /*DisControlGeometry dis_control_geometry(gate, "DisControlGeometry");
    SimpleDynamics<ControlDisplacement> dis_control(dis_control_geometry);*/

    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> corrected_configuration_fluid(InteractArgs(water_block_inner, 0.5), water_block_contact);
    Dynamics1Level<fluid_dynamics::Integration1stHalfCorrectionWithWallRiemann> pressure_relaxation(water_block_inner, water_block_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplexFreeSurface> update_density_by_summation(water_block_inner, water_block_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_block_contact);

    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex>
        free_stream_surface_indicator(water_block_inner, water_block_contact);
    /** Impose transport velocity formulation. */
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>>
        transport_velocity_correction(water_block_inner, water_block_contact);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step_size(water_block, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step_size(water_block);
    //----------------------------------------------------------------------
    //	Algorithms of FSI.
    //----------------------------------------------------------------------
    InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_from_fluid(gate_water_contact);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> fluid_pressure_force_on_gate(gate_water_contact);
    solid_dynamics::AverageVelocityAndAcceleration average_velocity_and_acceleration(gate);
    //----------------------------------------------------------------------
    //	Define the configuration related particles dynamics.
    //----------------------------------------------------------------------
    ParticleSorting particle_sorting(water_block);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    //BodyStatesRecordingToPlt write_real_body_states_to_plt(sph_system);
    BodyStatesRecordingToVtp write_real_body_states_to_vtp(sph_system);
    write_real_body_states_to_vtp.addToWrite<Real>(water_block, "Pressure");
    write_real_body_states_to_vtp.addToWrite<Vecd>(gate, "Velocity");
    ObservedQuantityRecording<Vecd> write_beam_tip_displacement("Position", gate_observer_contact);
    // TODO: observing position is not as good observing displacement.
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    gate_offset_position.exec();
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    wall_boundary_normal_direction.exec();
    gate_normal_direction.exec();
    gate_corrected_configuration.exec();
    panel_velocity.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    int number_of_iterations = 0;
    int screen_output_interval = 100;
    Real end_time = 0.06;
    Real output_interval = 0.001;
    Real dt = 0.0;   /**< Default acoustic time step sizes. */
    Real dt_s = 0.0; /**< Default acoustic time step sizes for solid. */
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    write_real_body_states_to_vtp.writeToFile();
    write_beam_tip_displacement.writeToFile();
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (physical_time < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < output_interval)
        {
            Real Dt = get_fluid_advection_time_step_size.exec();
            free_stream_surface_indicator.exec();
            update_density_by_summation.exec();
            corrected_configuration_fluid.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();

            /** FSI for viscous force. */
            viscous_force_from_fluid.exec();
            /** Update normal direction at elastic body surface. */
            gate_update_normal.exec();

            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt - relaxation_time);

                /** Fluid relaxation and force computation. */
                pressure_relaxation.exec(dt);
                fluid_pressure_force_on_gate.exec();
                density_relaxation.exec(dt);
                /** Solid dynamics time stepping. */
                Real dt_s_sum = 0.0;
                average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    dt_s = gate_computing_time_step_size.exec();
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;
                    gate_stress_relaxation_first_half.exec(dt_s);
                    gate_stress_relaxation_second_half.exec(dt_s);
                    dt_s_sum += dt_s;
                }
                average_velocity_and_acceleration.update_averages_.exec(dt);
                
                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
            }

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << physical_time
                          << "	Dt = " << Dt << "	dt = " << dt << "	dt_s = " << dt_s << "\n";
            }
            number_of_iterations++;

            /** Update cell linked list and configuration. */
            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                particle_sorting.exec();
            }
            water_block.updateCellLinkedList();
            gate.updateCellLinkedList();
            water_block_complex.updateConfiguration();
            gate_water_contact.updateConfiguration();
            /** Output the observed data. */
            write_beam_tip_displacement.writeToFile(number_of_iterations);
        }
        TickCount t2 = TickCount::now();
        write_real_body_states_to_vtp.writeToFile();
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();
    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;
    return 0;
}
