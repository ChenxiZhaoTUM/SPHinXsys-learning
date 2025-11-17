#include "sphinxsys.h" // SPHinXsys Library.
#include "bidirectional_buffer.h"
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "pressure_boundary.h"

using namespace SPH;

//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real resolution_ref = 0.01;   // particle spacing
Real BW = resolution_ref * 3; // boundary width
Real DL = 1.0;              // tank length
Real DH = 0.7;                // tank height
Real DW = 0.5;                // tank width
Real LL = DL;                // liquid length
Real LH = 0.5;                // liquid height
Real LW = DW;                // liquid width
Real tube_radius = 0.05;
Real tube_outer_radius = 0.05 + BW;
Vecd tube_transition = Vecd(0.0, 0.25, 0.0);
Vecd buffer_halfsize = Vecd(2.0 * resolution_ref, tube_radius, tube_radius);
Vecd buffer_translation = Vecd(-0.5 * DL - BW + 2.0 * resolution_ref, 0.25, 0.0);
Vecd buffer_right_translation = Vecd(0.5 * DL + BW - 2.0 * resolution_ref, 0.25, 0.0);
//----------------------------------------------------------------------
//	Material properties of the fluid.
//----------------------------------------------------------------------
Real U_f = 10.0;                     /**< Characteristic velocity. */
Real c_f = 10.0 * U_f * 1.5;              /**< Speed of sound. */

//Real rho0_inner = 80.0;
Real rho0_inner = 100.0;
Real mu_inner = 1.3e-5;
Real Cp_inner = 2200.0 / 10000;
Real k_inner = 0.03;
Real diffusion_coff_inner = k_inner / (Cp_inner * rho0_inner);
Real initial_temperature_inner = 20.0;

Real rho0_outer = 1025.0;
//Real mu_outer = 1.1e-2;
Real mu_outer = 0.5;
Real Cp_outer = 4000.0 / 10000;
Real k_outer = 0.6;
Real diffusion_coff_outer = k_outer / (Cp_outer * rho0_outer);
Real initial_temperature_outer = 0.0;

Real rho0_tube = 7850.0;
//Real Youngs_modulus_tube = 2.0e11;
Real Youngs_modulus_tube = 1.0e9;
//Real poisson_tube = 0.3;
Real poisson_tube = 0.4;
Real Cp_tube = 470.0 / 10000;
Real k_tube = 50.0;
Real diffusion_coff_tube = k_tube / (Cp_tube * rho0_tube);
Real initial_temperature_tube = 0.0;
Real physical_viscosity = (2 * tube_radius) / (DL + 2 * BW) / 4 * sqrt(rho0_tube * Youngs_modulus_tube) * (2 * tube_radius);

//----------------------------------------------------------------------
//	Geometric elements used in shape modeling.
//----------------------------------------------------------------------
class TubeBlock : public ComplexShape
{
public:
    explicit TubeBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1.0, 0, 0), tube_outer_radius,
            0.5 * (DL+ 2 * BW), int(20), tube_transition);

        subtract<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1.0, 0, 0), tube_radius,
            0.5 * (DL+ 2 * BW), int(20), tube_transition);
    }

};

class InnerFluidBlock : public ComplexShape
{
  public:
    explicit InnerFluidBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1.0, 0, 0), tube_radius,
            0.5 * (DL+ 2 * BW), int(20), tube_transition);
    }
};

class OuterFluidBlock : public ComplexShape
{
  public:
    explicit OuterFluidBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Vecd halfsize_water(0.5 * LL, 0.5 * LH, 0.5 * LW);
        Transform translation_water(Vecd(0.0, 0.5 * LH, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_water), halfsize_water);

        subtract<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1.0, 0, 0), tube_outer_radius,
            0.5 * DL, int(20), tube_transition);
    }
};

class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        Vecd halfsize_outer(0.5 * DL + BW, 0.5 * DH + BW, 0.5 * DW + BW);
        Vecd halfsize_inner(0.5 * DL, 0.5 * DH, 0.5 * DW);
        Transform translation_wall(Vecd(0.0, 0.5 * DH, 0.0));
        add<TransformShape<GeometricShapeBox>>(Transform(translation_wall), halfsize_outer);
        subtract<TransformShape<GeometricShapeBox>>(Transform(translation_wall), halfsize_inner);
        
        subtract<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1.0, 0, 0), tube_outer_radius,
            0.5 * BW, int(20), Vecd(-0.5 * DL  - 0.5 * BW, 0.25, 0.0));
        subtract<TriangleMeshShapeCylinder>(SimTK::UnitVec3(1.0, 0, 0), tube_outer_radius,
            0.5 * BW, int(20), Vecd(0.5 * DL  + 0.5 * BW, 0.25, 0.0));
    }
};
//----------------------------------------------------------------------
//	Inflow velocity
//----------------------------------------------------------------------
//struct InflowVelocity
//{
//    Real u_ref_, t_ref_;
//    AlignedBoxShape &aligned_box_;
//    Vecd halfsize_;
//
//    template <class BoundaryConditionType>
//    InflowVelocity(BoundaryConditionType &boundary_condition)
//        : u_ref_(3.0), t_ref_(2.0),
//          aligned_box_(boundary_condition.getAlignedBox()),
//          halfsize_(aligned_box_.HalfSize()) {}
//
//    Vecd operator()(Vecd &position, Vecd &velocity, Real current_time)
//    {
//        Vecd target_velocity = Vecd::Zero();
//        Real u_ave = 0.5 * u_ref_ * (1.0 - cos(Pi * current_time / t_ref_));
//        target_velocity[0] = 1.5 * u_ave * (1.0 - (position[1] * position[1] + position[2] * position[2]) / tube_radius / tube_radius);
//        return target_velocity;
//    }
//};

struct InflowVelocity
{
    Real u_ave;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ave(0.0) {}

    Vecd operator()(Vecd &position, Vecd &velocity, Real current_time)
    {
        Vecd target_velocity = velocity;

        u_ave = 0.2339;
        Real a[8] = {-0.0176, -0.0657, -0.0280, 0.0068, 0.0075, 0.0115, 0.0040, 0.0035};
        Real b[8] = {0.1205, 0.0171, -0.0384, -0.0152, -0.0122, 0.0002, 0.0033, 0.0060};
        Real w = 2 * Pi / 1;
        for (size_t i = 0; i < 8; i++)
        {
            u_ave = SMAX(u_ave + a[i] * cos(w * (i + 1) * current_time) + b[i] * sin(w * (i + 1) * current_time),
                        0.0);
        }
            
        target_velocity[0] = SMAX(1.5 * u_ave * (1.0 - (position[1] * position[1] + position[2] * position[2]) / tube_radius / tube_radius),
                                  0.);
        target_velocity[1] = 0.0;
        target_velocity[2] = 0.0;

        return target_velocity;
    }
};

struct LeftInflowPressure
{
    template <class BoundaryConditionType>
    LeftInflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real p, Real curent_time)
    {
        return p;
    }
};

struct RightOutflowPressure
{
    template <class BoundaryConditionType>
    RightOutflowPressure(BoundaryConditionType &boundary_condition) {}

    Real operator()(Real p, Real curent_time)
    {
        /*constant pressure*/
        Real pressure = 0.0;
        return pressure;
    }
};

//----------------------------------------------------------------------
//	Define external excitation.
//----------------------------------------------------------------------
Real f = 0.3;
Real omega = 2 * Pi * f;
Real A = 0.1;
Real gravity_g = 9.81;                    /**< Gravity force of fluid. */

class VariableGravity : public Gravity
{
public:
	VariableGravity() : Gravity(Vecd(0.0, 0.0, 0.0)) {};
	Vecd InducedAcceleration(const Vecd &position, Real physical_time) const
	{
		Vecd acceleration = Vecd::Zero();

		if (physical_time < 0.5)
		{
			acceleration[0] = A * omega * cos(omega * physical_time) / 0.5;
			acceleration[1] = - gravity_g;
		}
		else
		{
			acceleration[0] = - A * omega * omega * sin(omega * physical_time);
			acceleration[1] = - gravity_g;
		}
		
		return acceleration;
	}
};

//----------------------------------------------------------------------
//	Boundary constrain
//----------------------------------------------------------------------
class BoundaryGeometry : public BodyPartByParticle
{
  public:
    BoundaryGeometry(SPHBody &body, const std::string &body_part_name, Real constrain_len)
        : BodyPartByParticle(body, body_part_name), constrain_len_(constrain_len)
    {
        TaggingParticleMethod tagging_particle_method = std::bind(&BoundaryGeometry::tagManually, this, _1);
        tagParticles(tagging_particle_method);
    };
    virtual ~BoundaryGeometry(){};

  private:
      Real constrain_len_;

    void tagManually(size_t index_i)
    {
        if (base_particles_.ParticlePositions()[index_i][0] < -0.5 * DL - BW + constrain_len_ 
            || base_particles_.ParticlePositions()[index_i][0] > 0.5 * DL + BW - constrain_len_)
        {
            body_part_particles_.push_back(index_i);
        }
    };
};

//----------------------------------------------------------------------
//	Heat transfer initial conditions.
//----------------------------------------------------------------------
std::string diffusion_species_name = "Phi";

class InnerDiffusionInitialCondition : public LocalDynamics
{
  public:
    explicit InnerDiffusionInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name))
	{};

    void update(size_t index_i, Real dt)
    {
        phi_[index_i] = initial_temperature_inner;
    };

  protected:
    Real *phi_;
};

class OuterDiffusionInitialCondition : public LocalDynamics
{
  public:
    explicit OuterDiffusionInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name))
	{};

    void update(size_t index_i, Real dt)
    {
        phi_[index_i] = initial_temperature_outer;
    };

  protected:
    Real *phi_;
};

class TubeDiffusionInitialCondition : public LocalDynamics
{
  public:
    explicit TubeDiffusionInitialCondition(SPHBody &sph_body)
        : LocalDynamics(sph_body),
          phi_(particles_->registerStateVariable<Real>(diffusion_species_name))
	{};

    void update(size_t index_i, Real dt)
    {
        phi_[index_i] = initial_temperature_tube;
    };

  protected:
    Real *phi_;
};

using DiffusionBodyRelaxation = MultiPhaseDiffusionBodyRelaxationComplex<
    IsotropicThermalDiffusion, KernelGradientInner, KernelGradientContact>;

//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up an SPHSystem.
    //----------------------------------------------------------------------
    BoundingBox system_domain_bounds(Vecd(- 0.5 * DL - BW, -BW, -0.5 * DW - BW), Vecd(0.5 * DL + BW, DH + BW, 0.5 * DW + BW));
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    /** Tag for run particle relaxation for the initial body fitted distribution. */
    sph_system.setRunParticleRelaxation(false);
    /** Tag for computation start with relaxed body fitted particles distribution. */
    sph_system.setReloadParticles(true);
    sph_system.handleCommandlineOptions(ac, av);
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating bodies with corresponding materials and particles.
    //----------------------------------------------------------------------
    SolidBody tube(sph_system, makeShared<TubeBlock>("TubeBlock"));
    tube.defineBodyLevelSetShape();
    tube.defineClosure<SaintVenantKirchhoffSolid, IsotropicThermalDiffusion>(ConstructArgs(rho0_tube, Youngs_modulus_tube, poisson_tube), 
        ConstructArgs(diffusion_species_name, k_tube, rho0_tube, Cp_tube));
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? tube.generateParticles<BaseParticles, Reload>(tube.getName())
        : tube.generateParticles<BaseParticles, Lattice>();

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineBodyLevelSetShape();
    wall_boundary.defineMaterial<Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_boundary.generateParticles<BaseParticles, Reload>(wall_boundary.getName())
        : wall_boundary.generateParticles<BaseParticles, Lattice>();

    FluidBody fluid_in(sph_system, makeShared<InnerFluidBlock>("InnerFluidBlock"));
    fluid_in.defineBodyLevelSetShape();
    fluid_in.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicThermalDiffusion>(ConstructArgs(rho0_inner, c_f), mu_inner, 
        ConstructArgs(diffusion_species_name, k_inner, rho0_inner, Cp_inner));
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? fluid_in.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, fluid_in.getName())
        : fluid_in.generateParticles<BaseParticles, Lattice>();

    FluidBody fluid_out(sph_system, makeShared<OuterFluidBlock>("OuterFluidBlock"));
    fluid_out.defineBodyLevelSetShape();
    fluid_out.defineClosure<WeaklyCompressibleFluid, Viscosity, IsotropicThermalDiffusion>(ConstructArgs(rho0_outer, c_f), mu_outer, 
        ConstructArgs(diffusion_species_name, k_outer, rho0_outer, Cp_outer));
    fluid_out.generateParticles<BaseParticles, Lattice>();

    InnerRelation inner_fluid_body_inner(fluid_in);
    ContactRelation inner_fluid_body_tube_contact(fluid_in, {&tube});
    ComplexRelation inner_fluid_body_tube_complex(inner_fluid_body_inner, inner_fluid_body_tube_contact);

    InnerRelation outer_fluid_body_inner(fluid_out);
    ContactRelation outer_fluid_body_contacts(fluid_out, {&wall_boundary, &tube});
    ContactRelation outer_fluid_body_diffusion_contacts(fluid_out, {&tube});
    ComplexRelation outer_fluid_body_complex(outer_fluid_body_inner, outer_fluid_body_contacts);

    InnerRelation tube_inner(tube);
    ContactRelation tube_contacts(tube, {&fluid_in, &fluid_out});

    InnerRelation wall_boundary_inner(wall_boundary);

    if (sph_system.RunParticleRelaxation())
    {
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_tube_particles(tube);
        SimpleDynamics<RandomizeParticlePosition> random_wall_particles(wall_boundary);
        SimpleDynamics<RandomizeParticlePosition> random_inner_fluid_particles(fluid_in);
        BodyStatesRecordingToVtp write_bodies(sph_system);
        ReloadParticleIO write_particle_reload_files({&tube, &wall_boundary, &fluid_in});
        RelaxationStepInner relaxation_step_tube_inner(tube_inner);
        RelaxationStepInner relaxation_step_wall_boundary_inner(wall_boundary_inner);
        RelaxationStepInner relaxation_step_fluid_in_inner(inner_fluid_body_inner);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_tube_particles.exec(0.25);
        random_wall_particles.exec(0.25);
        random_inner_fluid_particles.exec(0.25);
        relaxation_step_tube_inner.SurfaceBounding().exec();
        relaxation_step_wall_boundary_inner.SurfaceBounding().exec();
        relaxation_step_fluid_in_inner.SurfaceBounding().exec();
        write_bodies.writeToFile();

        int ite_p = 0;
        while (ite_p < 1000)
        {
            relaxation_step_tube_inner.exec();
            relaxation_step_wall_boundary_inner.exec();
            relaxation_step_fluid_in_inner.exec();
            ite_p += 1;
            if (ite_p % 200 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
                write_bodies.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of inserted body finish !" << std::endl;
        write_particle_reload_files.writeToFile();
        return 0;
    }
    
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
    SimpleDynamics<NormalDirectionFromBodyShape> tube_normal_direction(tube);
    SimpleDynamics<NormalDirectionFromBodyShape> wall_normal_direction(wall_boundary);
    //----------------------------------------------------------------------
    // Pipe flow
    //----------------------------------------------------------------------
    InteractionDynamics<NablaWVComplex> inner_kernel_summation(inner_fluid_body_inner, inner_fluid_body_tube_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> inner_boundary_indicator(inner_fluid_body_inner, inner_fluid_body_tube_contact);

    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> inner_pressure_relaxation(inner_fluid_body_inner, inner_fluid_body_tube_contact);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallNoRiemann> inner_density_relaxation(inner_fluid_body_inner, inner_fluid_body_tube_contact);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> inner_viscous_force(inner_fluid_body_inner, inner_fluid_body_tube_contact);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> inner_transport_velocity_correction(inner_fluid_body_inner, inner_fluid_body_tube_contact);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_inner_fluid_advection_time_step(fluid_in, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_inner_fluid_acoustic_time_step(fluid_in);

    BodyAlignedBoxByCell left_emitter(fluid_in, makeShared<AlignedBoxShape>(xAxis, Transform(Vecd(buffer_translation)), buffer_halfsize));
    fluid_dynamics::BidirectionalBuffer<LeftInflowPressure> left_bidirection_buffer(left_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell right_emitter(fluid_in, makeShared<AlignedBoxShape>(xAxis,Transform(Rotation3d(Pi, Vecd(0., 1.0, 0.)), Vecd(buffer_right_translation)), buffer_halfsize));
    fluid_dynamics::BidirectionalBuffer<RightOutflowPressure> right_bidirection_buffer(right_emitter, in_outlet_particle_buffer);
    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> inner_update_density_by_summation(inner_fluid_body_inner, inner_fluid_body_tube_contact);
    SimpleDynamics<fluid_dynamics::PressureCondition<LeftInflowPressure>> left_inflow_pressure_condition(left_emitter);
    SimpleDynamics<fluid_dynamics::PressureCondition<RightOutflowPressure>> right_inflow_pressure_condition(right_emitter);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> inflow_velocity_condition(left_emitter);
    
    //----------------------------------------------------------------------
    // Sloshing
    //----------------------------------------------------------------------
    VariableGravity variable_gravity;
    SimpleDynamics<GravityForce<VariableGravity>> initialize_outer_fluid_step(fluid_out, variable_gravity);

    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> outer_pressure_relaxation(outer_fluid_body_inner, outer_fluid_body_contacts);
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallNoRiemann> outer_density_relaxation(outer_fluid_body_inner, outer_fluid_body_contacts);
    InteractionWithUpdate<fluid_dynamics::DensitySummationComplexFreeSurface> outer_update_density_by_summation(outer_fluid_body_inner, outer_fluid_body_contacts);
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> 
		outer_viscous_force(outer_fluid_body_inner, outer_fluid_body_contacts);

    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_outer_fluid_advection_time_step(fluid_out, U_f);
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_outer_fluid_acoustic_time_step(fluid_out);

    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> outer_kernel_correction_matrix(outer_fluid_body_inner, outer_fluid_body_contacts);
    InteractionDynamics<KernelGradientCorrectionInner> outer_kernel_gradient_update(outer_fluid_body_inner);
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseInner<Vecd, FixedDampingRate>>>
        outer_fluid_damping(0.2, outer_fluid_body_inner, "Velocity", mu_outer);

    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex> outer_free_surface_indicator(outer_fluid_body_inner, outer_fluid_body_contacts);
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>> outer_transport_velocity_correction(outer_fluid_body_inner, outer_fluid_body_contacts);
    
    //----------------------------------------------------------------------
    //	Solid dynamics for tube
    //----------------------------------------------------------------------
    InteractionWithUpdate<LinearGradientCorrectionMatrixInner> tube_corrected_configuration(tube_inner);
    Dynamics1Level<solid_dynamics::Integration1stHalfPK2> tube_stress_relaxation_first_half(tube_inner);
    Dynamics1Level<solid_dynamics::Integration2ndHalf> tube_stress_relaxation_second_half(tube_inner);
    ReduceDynamics<solid_dynamics::AcousticTimeStep> tube_computing_time_step_size(tube);
    SimpleDynamics<solid_dynamics::UpdateElasticNormalDirection> tube_update_normal(tube);
    /** Exert constrain on wall. */
    BoundaryGeometry tube_boundary_geometry(tube, "TubeBoundaryGeometry", resolution_ref * 4);
    SimpleDynamics<FixBodyPartConstraint> tube_constrain_holder(tube_boundary_geometry);
    DampingWithRandomChoice<InteractionSplit<DampingPairwiseInner<Vecd, FixedDampingRate>>>
        tube_velocity_damping(0.2, tube_inner, "Velocity", physical_viscosity);
    //----------------------------------------------------------------------
    //	FSI
    //----------------------------------------------------------------------
    InteractionWithUpdate<solid_dynamics::ViscousForceFromFluid> viscous_force_on_tube(tube_contacts);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(inner_density_relaxation)>> pressure_force_on_tube(tube_contacts);
    solid_dynamics::AverageVelocityAndAcceleration tube_average_velocity_and_acceleration(tube);

    //----------------------------------------------------------------------
    //	Heat transfer
    //----------------------------------------------------------------------
    DiffusionBodyRelaxation inner_temperature_relaxation(inner_fluid_body_inner, inner_fluid_body_tube_contact);
    DiffusionBodyRelaxation outer_temperature_relaxation(outer_fluid_body_inner, outer_fluid_body_diffusion_contacts);
    DiffusionBodyRelaxation tube_temperature_relaxation(tube_inner, tube_contacts);
    GetDiffusionTimeStepSize inner_diffusion_time_step_size(fluid_in);
    GetDiffusionTimeStepSize outer_diffusion_time_step_size(fluid_out);
    GetDiffusionTimeStepSize tube_diffusion_time_step_size(tube);
    SimpleDynamics<InnerDiffusionInitialCondition> setup_inner_diffusion_initial_condition(fluid_in);
    SimpleDynamics<OuterDiffusionInitialCondition> setup_outer_diffusion_initial_condition(fluid_out);
    SimpleDynamics<TubeDiffusionInitialCondition> setup_tube_diffusion_initial_condition(tube);

    //----------------------------------------------------------------------
    //	Define the configuration related particles dynamics.
    //----------------------------------------------------------------------
    ParticleSorting inner_particle_sorting(fluid_in);
    ParticleSorting outer_particle_sorting(fluid_out);
    //----------------------------------------------------------------------
    //	Define the methods for I/O operations and observations of the simulation.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_real_body_states(sph_system);
    write_real_body_states.addToWrite<int>(fluid_in, "BufferParticleIndicator");
    write_real_body_states.addToWrite<Vecd>(tube, "NormalDirection");
    write_real_body_states.addToWrite<Real>(fluid_out, "Pressure");
    write_real_body_states.addDerivedVariableRecording<SimpleDynamics<VonMisesStress>>(tube);
    write_real_body_states.addDerivedVariableRecording<SimpleDynamics<Displacement>>(tube);

    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    inner_fluid_body_tube_complex.updateConfiguration();
    outer_fluid_body_complex.updateConfiguration();
    outer_fluid_body_diffusion_contacts.updateConfiguration();
    tube_contacts.updateConfiguration();

    inner_boundary_indicator.exec();
    left_bidirection_buffer.tag_buffer_particles.exec();
    right_bidirection_buffer.tag_buffer_particles.exec();
    outer_free_surface_indicator.exec();

    /** computing surface normal direction for the wall. */
    tube_normal_direction.exec();
    wall_normal_direction.exec();

    outer_kernel_correction_matrix.exec();
    outer_kernel_gradient_update.exec();
    tube_corrected_configuration.exec();

    setup_inner_diffusion_initial_condition.exec();
    setup_outer_diffusion_initial_condition.exec();
    setup_tube_diffusion_initial_condition.exec();
    //----------------------------------------------------------------------
    //	Setup for time-stepping control
    //----------------------------------------------------------------------
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    Real end_time = 10;
    Real output_interval = end_time / 1000.0; /**< time stamps for output,WriteToFile*/
    int number_of_iterations = 0;
    int screen_output_interval = 40;
    Real dt = 0.0;
    Real dt_s = 0.0;
    //----------------------------------------------------------------------
    //	Statistics for CPU time
    //----------------------------------------------------------------------
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    //----------------------------------------------------------------------
    //	First output before the main loop.
    //----------------------------------------------------------------------
    write_real_body_states.writeToFile();
    //----------------------------------------------------------------------
    //	Main loop starts here.
    //----------------------------------------------------------------------
    while (physical_time < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < output_interval)
        {
            initialize_outer_fluid_step.exec();

            Real Dt = SMIN(get_inner_fluid_advection_time_step.exec(), get_outer_fluid_advection_time_step.exec());
            inner_update_density_by_summation.exec();
            inner_viscous_force.exec();
            inner_transport_velocity_correction.exec();
            
            outer_update_density_by_summation.exec();
            outer_viscous_force.exec();
            outer_transport_velocity_correction.exec();

            /** FSI for viscous force. */
            viscous_force_on_tube.exec();

            size_t inner_ite_dt = 0;
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_inner_fluid_acoustic_time_step.exec(), get_outer_fluid_acoustic_time_step.exec(),
                    inner_diffusion_time_step_size.exec(), outer_diffusion_time_step_size.exec(), tube_diffusion_time_step_size.exec(),
                    Dt);

                // pressure relax
                inner_pressure_relaxation.exec(dt);
                // outer damping
                if (physical_time < 0.5)
				{
					outer_fluid_damping.exec(dt);
				}
                outer_pressure_relaxation.exec(dt);
                /** FSI for pressure force. */
                pressure_force_on_tube.exec();

                // inner pressure boundary
                inner_kernel_summation.exec();
                left_inflow_pressure_condition.exec(dt);
                right_inflow_pressure_condition.exec(dt);
                inflow_velocity_condition.exec();

                // density relax
                inner_density_relaxation.exec(dt);
                outer_density_relaxation.exec(dt);

                // heat transfer
                inner_temperature_relaxation.exec(dt);
                outer_temperature_relaxation.exec(dt);
                tube_temperature_relaxation.exec(dt);

                // tube deforms
                Real dt_s_sum = 0.0;
                tube_average_velocity_and_acceleration.initialize_displacement_.exec();
                while (dt_s_sum < dt)
                {
                    dt_s = tube_computing_time_step_size.exec();
                    if (dt - dt_s_sum < dt_s)
                        dt_s = dt - dt_s_sum;

                    tube_stress_relaxation_first_half.exec(dt_s);

                    tube_constrain_holder.exec(dt_s);
                    tube_velocity_damping.exec(dt_s);
                    tube_constrain_holder.exec(dt_s);

                    tube_stress_relaxation_second_half.exec(dt_s);
                    dt_s_sum += dt_s;
                }
                tube_average_velocity_and_acceleration.update_averages_.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
                inner_ite_dt++;
            }

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9)
                          << "N=" << number_of_iterations
                          << "	Time = " << physical_time
                          << "	Dt = " << Dt << "	dt = " << dt << "	dt_s = " << dt_s << "\n";
            }
            number_of_iterations++;

            left_bidirection_buffer.injection.exec();
            right_bidirection_buffer.injection.exec();

            left_bidirection_buffer.deletion.exec();
            right_bidirection_buffer.deletion.exec();

            /** Water block configuration and periodic condition. */
            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                inner_particle_sorting.exec();
                outer_particle_sorting.exec();
            }
            fluid_in.updateCellLinkedList();
            inner_fluid_body_tube_complex.updateConfiguration();

            fluid_out.updateCellLinkedList();
            outer_fluid_body_complex.updateConfiguration();
            outer_fluid_body_diffusion_contacts.updateConfiguration();

            tube.updateCellLinkedList();
            tube_normal_direction.exec();
            tube_contacts.updateConfiguration();

            inner_boundary_indicator.exec();
            left_bidirection_buffer.tag_buffer_particles.exec();
            right_bidirection_buffer.tag_buffer_particles.exec();
            outer_free_surface_indicator.exec();
        }
        TickCount t2 = TickCount::now();

        write_real_body_states.writeToFile();
        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }

    TickCount t4 = TickCount::now();
    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds() << " seconds." << std::endl;

    return 0;
}
