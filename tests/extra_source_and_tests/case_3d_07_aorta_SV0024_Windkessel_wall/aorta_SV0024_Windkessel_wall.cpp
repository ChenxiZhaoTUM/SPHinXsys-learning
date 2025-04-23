/**
 * @file 	aorta_SV0024_Windkessel_wall.cpp
 * @brief 
 */
#include "sphinxsys.h"
#include "bidirectional_buffer.h"
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "pressure_boundary.h"
#include "particle_generation_and_detection.h"
#include "windkessel_bc.h"
#include "hemodynamic_indices.h"

/**
 * @brief Namespace cite here.
 */
using namespace SPH;

std::string full_path_to_file = "./input/aorta_0154_0001.stl";

/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vecd(-6.0E-2, -3.0E-2, -1.0E-2), Vecd(6.0E-2, 8.0E-2, 20.0E-2));

Real rho0_f = 1060.0;                   
Real mu_f = 0.0035;
Real U_f = 3.0;
Real c_f = 10.0*U_f;

Real dp_0 = 0.06E-2;
Real wall_resolution = 0.5 * dp_0;
Real scaling = 1.0E-2;
Vecd translation(0.0, 0.0, 0.0);

// buffer locations
struct RotationResult
{
    Vec3d axis;
    Real angle;
};

RotationResult RotationCalculator(Vecd target_normal, Vecd standard_direction)
{
    target_normal.normalize();

    Vec3d axis = standard_direction.cross(target_normal);
    Real angle = std::acos(standard_direction.dot(target_normal));

    if (axis.norm() < 1e-6)
    {
        if (standard_direction.dot(target_normal) < 0)
        {
            axis = Vec3d(1, 0, 0);
            angle = M_PI;
        }
        else
        {
            axis = Vec3d(0, 0, 1);
            angle = 0;
        }
    }
    else
    {
        axis.normalize();
    }

    return {axis, angle};
}

Vecd standard_direction(1, 0, 0);

// inlet (-1.015, 4.519, 2.719), (0.100, 0.167, 0.981), 1.366
Real A_in = 5.9825 * scaling * scaling;
Real radius_inlet = 1.366 * scaling;
Vec3d inlet_half = Vec3d(2.0 * dp_0, 1.6 * scaling, 1.6 * scaling);
Vec3d inlet_vector(0.100, 0.167, 0.981);
Vec3d inlet_normal = inlet_vector.normalized();
Vec3d inlet_center = Vec3d(-1.015, 4.519, 2.719) * scaling;
Vec3d inlet_cut_translation = inlet_center - inlet_normal * (2.0 * dp_0);
Vec3d inlet_blood_cut_translation = inlet_center - inlet_normal * (1.0 * dp_0);
Vec3d inlet_buffer_translation = inlet_center + inlet_normal * (2.0 * dp_0 + 1.0 * dp_0);
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, standard_direction);
Rotation3d inlet_emitter_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);
Rotation3d inlet_disposer_rotation(inlet_rotation_result.angle + Pi, inlet_rotation_result.axis);

// outlet1 (-0.947, 4.322, 10.110), (0.607, 0.373, 0.702), 0.280
Real A_out1 = 0.2798 * scaling * scaling;
Vec3d outlet_1_half = Vec3d(2.0 * dp_0, 0.5 * scaling, 0.5 * scaling);
Vec3d outlet_1_vector(0.607, 0.373, 0.702);
Vec3d outlet_1_normal = outlet_1_vector.normalized();
Vec3d outlet_1_center = Vec3d(-0.947, 4.322, 10.110) * scaling;
Vec3d outlet_1_cut_translation = outlet_1_center + outlet_1_normal * (2.0 * dp_0);
Vec3d outlet_1_blood_cut_translation = outlet_1_center + outlet_1_normal * (1.0 * dp_0);
Vec3d outlet_1_buffer_translation = outlet_1_center - outlet_1_normal * (2.0 * dp_0 + 1.0 * dp_0);
RotationResult outlet_1_rotation_result = RotationCalculator(outlet_1_normal, standard_direction);
Rotation3d outlet_1_disposer_rotation(outlet_1_rotation_result.angle, outlet_1_rotation_result.axis);
Rotation3d outlet_1_emitter_rotation(outlet_1_rotation_result.angle + Pi, outlet_1_rotation_result.axis);

// outlet2 (-2.650, 3.069, 10.911), (-0.099, 0.048, 0.994), 0.253
Real A_out2 = 0.2043 * scaling * scaling;
Vec3d outlet_2_half = Vec3d(2.0 * dp_0, 1.0 * scaling, 1.0 * scaling);
Vec3d outlet_2_vector(-0.099, 0.048, 0.994);
Vec3d outlet_2_normal = outlet_2_vector.normalized();
Vec3d outlet_2_center = Vec3d(-2.650, 3.069, 10.911) * scaling;
Vec3d outlet_2_cut_translation = outlet_2_center + outlet_2_normal * (2.0 * dp_0);
Vec3d outlet_2_blood_cut_translation = outlet_2_center + outlet_2_normal * (1.0 * dp_0);
Vec3d outlet_2_buffer_translation = outlet_2_center - outlet_2_normal * (2.0 * dp_0 + 1.0 * dp_0);
RotationResult outlet_2_rotation_result = RotationCalculator(outlet_2_normal, standard_direction);
Rotation3d outlet_2_disposer_rotation(outlet_2_rotation_result.angle, outlet_2_rotation_result.axis);
Rotation3d outlet_2_emitter_rotation(outlet_2_rotation_result.angle + Pi, outlet_2_rotation_result.axis);

// outlet3 (-2.946, 1.789, 10.007), (-0.146, -0.174, 0.974), 0.356
Real A_out3 = 0.3721 * scaling * scaling;
Real radius_outlet3 = 0.356 * scaling;
Vec3d outlet_3_half = Vec3d(2.0 * dp_0, 0.4 * scaling, 0.4 * scaling);
Vec3d outlet_3_vector(-0.146, -0.174, 0.974);
Vec3d outlet_3_normal = outlet_3_vector.normalized();
Vec3d outlet_3_center = Vec3d(-2.946, 1.789, 10.007) * scaling;
Vec3d outlet_3_cut_translation = outlet_3_center + outlet_3_normal * (2.0 * dp_0);
Vec3d outlet_3_blood_cut_translation = outlet_3_center + outlet_3_normal * (1.0 * dp_0);
Vec3d outlet_3_buffer_translation = outlet_3_center - outlet_3_normal * (2.0 * dp_0 + 1.0 * dp_0);
RotationResult outlet_3_rotation_result = RotationCalculator(outlet_3_normal, standard_direction);
Rotation3d outlet_3_disposer_rotation(outlet_3_rotation_result.angle, outlet_3_rotation_result.axis);
Rotation3d outlet_3_emitter_rotation(outlet_3_rotation_result.angle + Pi, outlet_3_rotation_result.axis);

// outlet4 (-1.052, 1.152, 9.669), (0.568, 0.428, 0.703), 0.395
Real A_out4 = 0.4567 * scaling * scaling;
Vec3d outlet_4_half = Vec3d(2.0 * dp_0, 1.0 * scaling, 1.0 * scaling);
Vec3d outlet_4_vector(0.568, 0.428, 0.703);
Vec3d outlet_4_normal = outlet_4_vector.normalized();
Vec3d outlet_4_center = Vec3d(-1.052, 1.152, 9.669) * scaling;
Vec3d outlet_4_cut_translation = outlet_4_center + outlet_4_normal * (2.0 * dp_0);
Vec3d outlet_4_blood_cut_translation = outlet_4_center + outlet_4_normal * (1.0 * dp_0);
Vec3d outlet_4_buffer_translation = outlet_4_center - outlet_4_normal * (2.0 * dp_0 + 1.0 * dp_0);
RotationResult outlet_4_rotation_result = RotationCalculator(outlet_4_normal, standard_direction);
Rotation3d outlet_4_disposer_rotation(outlet_4_rotation_result.angle, outlet_4_rotation_result.axis);
Rotation3d outlet_4_emitter_rotation(outlet_4_rotation_result.angle + Pi, outlet_4_rotation_result.axis);

// outlet5 (-1.589, -0.797, 0.247), (-0.033, 0.073, -0.997), 0.921
Real A_out5 = 2.6756 * scaling * scaling;
Vec3d outlet_5_half = Vec3d(2.0 * dp_0, 1.5 * scaling, 1.5 * scaling);
Vec3d outlet_5_vector(-0.033, 0.073, -0.997);
Vec3d outlet_5_normal = outlet_5_vector.normalized();
Vec3d outlet_5_center = Vec3d(-1.589, -0.797, 0.247) * scaling;
Vec3d outlet_5_cut_translation = outlet_5_center + outlet_5_normal * (2.0 * dp_0);
Vec3d outlet_5_blood_cut_translation = outlet_5_center + outlet_5_normal * (1.0 * dp_0);
Vec3d outlet_5_buffer_translation = outlet_5_center - outlet_5_normal * (2.0 * dp_0 + 1.0 * dp_0);
RotationResult outlet_5_rotation_result = RotationCalculator(outlet_5_normal, standard_direction);
Rotation3d outlet_5_disposer_rotation(outlet_5_rotation_result.angle, outlet_5_rotation_result.axis);
Rotation3d outlet_5_emitter_rotation(outlet_5_rotation_result.angle + Pi, outlet_5_rotation_result.axis);


class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        /** Geometry definition. */
        add<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
    }
};

class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        /** Geometry definition. */
        add<ExtrudeShape<TriangleMeshShapeSTL>>(5 * wall_resolution, full_path_to_file, translation, scaling);
        subtract<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
    }
};

struct InflowVelocity
{
    Real u_ave, interval_;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ave(0.0), interval_(0.66) {}

    Vecd operator()(Vecd &position, Vecd &velocity, Real current_time)
    {
        Vecd target_velocity = velocity;
        int n = static_cast<int>(current_time / interval_);
        Real t_in_cycle = current_time - n * interval_;

        u_ave = 5.0487;
        Real a[8] = {4.5287, -4.3509, -5.8551, -1.5063, 1.2800, 0.9012, 0.0855, -0.0480};
        Real b[8] = {-8.0420, -6.2637, 0.7465, 3.5239, 1.6283, -0.1306, -0.2738, -0.0449};

        Real w = 2 * Pi / 1.0;
        for (size_t i = 0; i < 8; i++)
        {
            u_ave = u_ave + a[i] * cos(w * (i + 1) * t_in_cycle) + b[i] * sin(w * (i + 1) * t_in_cycle);
        }
            
        //target_velocity[0] = SMAX(2.0 * u_ave * (1.0 - (position[1] * position[1] + position[2] * position[2]) / radius_inlet / radius_inlet),
        //                          1.0e-2);

        target_velocity[0] = 2.0 * u_ave * (1.0 - (position[1] * position[1] + position[2] * position[2]) / radius_inlet / radius_inlet);

        target_velocity[1] = 0.0;
        target_velocity[2] = 0.0;

        return target_velocity;
    }
};

/**
 * @brief 	Main program starts here.
 */
int main(int ac, char *av[])
{
    /**
     * @brief Build up -- a SPHSystem --
     */
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(true);       // Tag for computation with save particles distribution
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    /**
     * @brief Material property, particles and body creation of fluid.
     */
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineClosure<WeaklyCompressibleFluid, Viscosity>(ConstructArgs(rho0_f, c_f), mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    {
        water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName());
    }
    else
    {
        water_block.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
        water_block.generateParticles<BaseParticles, Lattice>();
    }

    /**
     * @brief 	Particle and body creation of wall boundary.
     */
    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("Wall"));
    wall_boundary.defineAdaptationRatios(1.15, dp_0/wall_resolution);
    wall_boundary.defineMaterial<Solid>();
    if (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
    {
        wall_boundary.generateParticles<BaseParticles, Reload>(wall_boundary.getName());
    }
    else
    {
        wall_boundary.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
        wall_boundary.generateParticles<BaseParticles, Lattice>();
    }

    /** topology */
    InnerRelation water_block_inner(water_block);
    ContactRelation water_block_contact(water_block, {&wall_boundary});
    ContactRelation wall_contact(wall_boundary, {&water_block});
    ComplexRelation water_block_complex(water_block_inner, water_block_contact);
    if (sph_system.RunParticleRelaxation())
    {
        /** body topology only for particle relaxation */
        InnerRelation water_block_inner(water_block);
        InnerRelation wall_boundary_inner(wall_boundary);

        /** cut wall */
        /*BodyAlignedCylinderByCell inlet_detection_cylinder(wall_boundary,
                                                           makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_half[1], inlet_half[0]));*/
        BodyAlignedBoxByCell inlet_detection_box(wall_boundary,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_half));
        BodyAlignedBoxByCell outlet01_detection_box(wall_boundary,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_emitter_rotation), Vec3d(outlet_1_cut_translation)), outlet_1_half));
        BodyAlignedBoxByCell outlet02_detection_box(wall_boundary,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_emitter_rotation), Vec3d(outlet_2_cut_translation)), outlet_2_half));
        /*BodyAlignedCylinderByCell outlet03_detection_cylinder(wall_boundary,
                                                           makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(outlet_3_emitter_rotation), Vec3d(outlet_3_cut_translation)), outlet_3_half[1], outlet_3_half[0]));*/
        BodyAlignedBoxByCell outlet03_detection_box(wall_boundary,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_emitter_rotation), Vec3d(outlet_3_cut_translation)), outlet_3_half));
        BodyAlignedBoxByCell outlet04_detection_box(wall_boundary,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_emitter_rotation), Vec3d(outlet_4_cut_translation)), outlet_4_half));
        BodyAlignedBoxByCell outlet05_detection_box(wall_boundary,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_emitter_rotation), Vec3d(outlet_5_cut_translation)), outlet_5_half));
        //RealBody test_body_in(
        //    sph_system, makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_half[1], inlet_half[0], "TestBodyIn"));
        //test_body_in.generateParticles<BaseParticles, Lattice>();
        
        /** cut blood */
        BodyAlignedBoxByCell inlet_blood_detection_box(water_block,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_blood_cut_translation)), inlet_half));
        BodyAlignedBoxByCell outlet01_blood_detection_box(water_block,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_emitter_rotation), Vec3d(outlet_1_blood_cut_translation)), outlet_1_half));
        BodyAlignedBoxByCell outlet02_blood_detection_box(water_block,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_emitter_rotation), Vec3d(outlet_2_blood_cut_translation)), outlet_2_half));
        BodyAlignedBoxByCell outlet03_blood_detection_box(water_block,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_emitter_rotation), Vec3d(outlet_3_blood_cut_translation)), outlet_3_half));
        BodyAlignedBoxByCell outlet04_blood_detection_box(water_block,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_emitter_rotation), Vec3d(outlet_4_blood_cut_translation)), outlet_4_half));
        BodyAlignedBoxByCell outlet05_blood_detection_box(water_block,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_emitter_rotation), Vec3d(outlet_5_blood_cut_translation)), outlet_5_half));
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** Random reset the insert body particle position. */
        SimpleDynamics<RandomizeParticlePosition> water_block_random_inserted_body_particles(water_block);
        SimpleDynamics<RandomizeParticlePosition> wall_boundary_random_inserted_body_particles(wall_boundary);

        // here, need a class to switch particles in aligned box to ghost particles (not real particles)
        //SimpleDynamics<DeleteParticlesInCylinder> inlet_particles_detection(inlet_detection_cylinder);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet01_particles_detection(outlet01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet02_particles_detection(outlet02_detection_box);
        //SimpleDynamics<DeleteParticlesInCylinder> outlet03_particles_detection(outlet03_detection_cylinder);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet03_particles_detection(outlet03_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet04_particles_detection(outlet04_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet05_particles_detection(outlet05_detection_box);

        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> inlet_blood_particles_detection(inlet_blood_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet01_blood_particles_detection(outlet01_blood_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet02_blood_particles_detection(outlet02_blood_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet03_blood_particles_detection(outlet03_blood_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet04_blood_particles_detection(outlet04_blood_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet05_blood_particles_detection(outlet05_blood_detection_box);

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp water_block_write_inserted_body_to_vtp(water_block);
        BodyStatesRecordingToVtp wall_boundary_write_inserted_body_to_vtp(wall_boundary);
        BodyStatesRecordingToVtp write_all_bodies_to_vtp({sph_system});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files({ &wall_boundary, &water_block });
        /** A  Physics relaxation step. */
        RelaxationStepInner water_block_relaxation_step_inner(water_block_inner);
        RelaxationStepInner wall_boundary_relaxation_step_inner(wall_boundary_inner);

        ParticleSorting particle_sorting_wall(wall_boundary);
        ParticleSorting particle_sorting_water(water_block);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        water_block_random_inserted_body_particles.exec(0.25);
        wall_boundary_random_inserted_body_particles.exec(0.25);
        water_block_relaxation_step_inner.SurfaceBounding().exec();
        wall_boundary_relaxation_step_inner.SurfaceBounding().exec();
        water_block_write_inserted_body_to_vtp.writeToFile(0);
        wall_boundary_write_inserted_body_to_vtp.writeToFile(0);
        //----------------------------------------------------------------------
        //	Relax particles of the insert body.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 3000)
        {
            water_block_relaxation_step_inner.exec();
            wall_boundary_relaxation_step_inner.exec();
            ite_p += 1;
            if (ite_p % 100 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";

                if (ite_p % 500 == 0)
                {
                    water_block_write_inserted_body_to_vtp.writeToFile(ite_p);
                    wall_boundary_write_inserted_body_to_vtp.writeToFile(ite_p);
                }
            }
        }
        std::cout << "The physics relaxation process of inserted body finish !" << std::endl;

        inlet_particles_detection.exec();
        particle_sorting_wall.exec();
        wall_boundary.updateCellLinkedList();
        outlet01_particles_detection.exec();
        particle_sorting_wall.exec();
        wall_boundary.updateCellLinkedList();
        outlet02_particles_detection.exec();
        particle_sorting_wall.exec();
        wall_boundary.updateCellLinkedList();
        outlet03_particles_detection.exec();
        particle_sorting_wall.exec();
        wall_boundary.updateCellLinkedList();
        outlet04_particles_detection.exec();
        particle_sorting_wall.exec();
        wall_boundary.updateCellLinkedList();
        outlet05_particles_detection.exec();
        particle_sorting_wall.exec();
        wall_boundary.updateCellLinkedList();

        inlet_blood_particles_detection.exec();
        particle_sorting_water.exec();
        water_block.updateCellLinkedList();
        outlet01_blood_particles_detection.exec();
        particle_sorting_water.exec();
        water_block.updateCellLinkedList();
        outlet02_blood_particles_detection.exec();
        particle_sorting_water.exec();
        water_block.updateCellLinkedList();
        outlet03_blood_particles_detection.exec();
        particle_sorting_water.exec();
        water_block.updateCellLinkedList();
        outlet04_blood_particles_detection.exec();
        particle_sorting_water.exec();
        water_block.updateCellLinkedList();
        outlet05_blood_particles_detection.exec();
        particle_sorting_water.exec();
        water_block.updateCellLinkedList();

        /** Output results. */
        write_all_bodies_to_vtp.writeToFile(ite_p);
        write_particle_reload_files.writeToFile(0);

        return 0;
    }
    /**
     * @brief 	Define all numerical methods which are used in this case.
     */
    /**
     * @brief 	Methods used for time stepping.
     */
    SimpleDynamics<NormalDirectionFromBodyShape> wall_boundary_normal_direction(wall_boundary);
    InteractionWithUpdate<LinearGradientCorrectionMatrixComplex> kernel_correction_complex(water_block_inner, water_block_contact);
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_block_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex>
        free_stream_surface_indicator(water_block_inner, water_block_contact);
    /** Pressure relaxation algorithm without Riemann solver for viscous flows. */
    Dynamics1Level<fluid_dynamics::Integration1stHalfCorrectionWithWallRiemann> pressure_relaxation(water_block_inner, water_block_contact);
    /** Pressure relaxation algorithm by using position verlet time stepping. */
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_block_contact);
    /* Time step size without considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AdvectionViscousTimeStep> get_fluid_advection_time_step_size(water_block, U_f);
    /** Time step size with considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AcousticTimeStep> get_fluid_time_step_size(water_block);
    /** Computing viscous acceleration. */
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_block_contact);
    /** Impose transport velocity. */
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>>
        transport_velocity_correction(water_block_inner, water_block_contact);

    // bidirectional buffer
    //BodyAlignedCylinderByCell inlet_emitter(water_block, makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_buffer_translation)), inlet_half[1], inlet_half[0]));
    //fluid_dynamics::NonPrescribedPressureBidirectionalBufferArb<AlignedCylinderShape> inflow_injection(inlet_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell inlet_emitter(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_buffer_translation)), inlet_half));
    fluid_dynamics::BidirectionalBuffer<fluid_dynamics::NonPrescribedPressure> inflow_injection(inlet_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_1(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_emitter_rotation), Vec3d(outlet_1_buffer_translation)), outlet_1_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_1(outflow_emitter_1, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_2(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_emitter_rotation), Vec3d(outlet_2_buffer_translation)), outlet_2_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_2(outflow_emitter_2, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_3(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_emitter_rotation), Vec3d(outlet_3_buffer_translation)), outlet_3_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_3(outflow_emitter_3, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_4(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_emitter_rotation), Vec3d(outlet_4_buffer_translation)), outlet_4_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_4(outflow_emitter_4, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_5(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_emitter_rotation), Vec3d(outlet_5_buffer_translation)), outlet_5_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_5(outflow_emitter_5, in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_block_contact);
    //SimpleDynamics<fluid_dynamics::InflowVelocityConditionArb<InflowVelocity, AlignedCylinderShape>> emitter_buffer_inflow_condition(inlet_emitter);
    SimpleDynamics<fluid_dynamics::InflowVelocityCondition<InflowVelocity>> emitter_buffer_inflow_condition(inlet_emitter);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition1(outflow_emitter_1);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition2(outflow_emitter_2);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition3(outflow_emitter_3);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition4(outflow_emitter_4);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition5(outflow_emitter_5);

    ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_inlet_transient_flow_rate(inlet_emitter, A_in);
    ReduceDynamics<fluid_dynamics::SectionTransientMassFlowRate> compute_inlet_transient_mass_flow_rate(inlet_emitter, A_in);
    //ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_outlet_01_transient_flow_rate(outflow_emitter_1, Pi * pow(0.280*scaling, 2));
    //ReduceDynamics<fluid_dynamics::SectionTransientMassFlowRate> compute_outlet_01_transient_mass_flow_rate(outflow_emitter_1, Pi * pow(0.280*scaling, 2));
    //ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_outlet_02_transient_flow_rate(outflow_emitter_2, Pi * pow(0.253*scaling, 2));
    //ReduceDynamics<fluid_dynamics::SectionTransientMassFlowRate> compute_outlet_02_transient_mass_flow_rate(outflow_emitter_2, Pi * pow(0.253*scaling, 2));
    //ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_outlet_03_transient_flow_rate(outflow_emitter_3, Pi * pow(0.356*scaling, 2));
    //ReduceDynamics<fluid_dynamics::SectionTransientMassFlowRate> compute_outlet_03_transient_mass_flow_rate(outflow_emitter_3, Pi * pow(0.356*scaling, 2));
    //ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_outlet_04_transient_flow_rate(outflow_emitter_4, Pi * pow(0.395*scaling, 2));
    //ReduceDynamics<fluid_dynamics::SectionTransientMassFlowRate> compute_outlet_04_transient_mass_flow_rate(outflow_emitter_4, Pi * pow(0.395*scaling, 2));
    //ReduceDynamics<fluid_dynamics::SectionTransientFlowRate> compute_outlet_05_transient_flow_rate(outflow_emitter_5, Pi * pow(0.921*scaling, 2));
    //ReduceDynamics<fluid_dynamics::SectionTransientMassFlowRate> compute_outlet_05_transient_mass_flow_rate(outflow_emitter_5, Pi * pow(0.921*scaling, 2));

    InteractionWithUpdate<solid_dynamics::WallShearStress> viscous_force_from_fluid(wall_contact);
    SimpleDynamics<solid_dynamics::FirstLayerFromFluid> solid_first_layer(wall_boundary, water_block);
    SimpleDynamics<solid_dynamics::HemodynamicIndiceCalculation> hemodynamic_indice_calculation(wall_boundary, 0.66);
    InteractionWithUpdate<solid_dynamics::PressureForceFromFluid<decltype(density_relaxation)>> pressure_force_on_wall(wall_contact);

    /**
     * @brief Output.
     */
    //----------------------------------------------------------------------
    //	Define the configuration related particles dynamics.
    //----------------------------------------------------------------------
    /** Output the body states. */
    ParticleSorting particle_sorting(water_block);
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<Real>(water_block, "DensityChangeRate");
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "PositionDivergence");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");
    body_states_recording.addToWrite<int>(wall_boundary, "SolidFirstLayerIndicator");
    body_states_recording.addToWrite<Vecd>(wall_boundary, "NormalDirection");
    body_states_recording.addToWrite<Vecd>(wall_boundary, "WallShearStress");
    body_states_recording.addToWrite<Real>(wall_boundary, "TimeAveragedWallShearStress");
    body_states_recording.addToWrite<Real>(wall_boundary, "OscillatoryShearIndex");

    /**
     * @brief Setup geometry and initial conditions.
     */
    sph_system.initializeSystemCellLinkedLists(); 
    sph_system.initializeSystemConfigurations();
    free_stream_surface_indicator.exec();
    inflow_injection.tag_buffer_particles.exec();
    outflow_injection_1.tag_buffer_particles.exec();
    outflow_injection_2.tag_buffer_particles.exec();
    outflow_injection_3.tag_buffer_particles.exec();
    outflow_injection_4.tag_buffer_particles.exec();
    outflow_injection_5.tag_buffer_particles.exec();
    wall_boundary_normal_direction.exec();
    solid_first_layer.exec();
    kernel_correction_complex.exec();
    
    /**
     * @brief 	Basic parameters.
     */
    Real &physical_time = *sph_system.getSystemVariableDataByName<Real>("PhysicalTime");
    size_t number_of_iterations = 0.0;
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 3.5;   /**< End time. */
    Real Output_Time = 0.01; /**< Time stamps for output of body states. */
    Real dt = 0.0;          /**< Default acoustic time step sizes. */
    /** statistics for computing CPU time. */
    TickCount t1 = TickCount::now();
    TimeInterval interval;
    TimeInterval interval_computing_time_step;
    TimeInterval interval_computing_pressure_relaxation;
    TimeInterval interval_updating_configuration;
    TickCount time_instance;
    Real accumulated_time = 0.006;
    int updateP_n = 0;

    /** Output the start states of bodies. */
    body_states_recording.writeToFile(0);

    outflow_pressure_condition1.getTargetPressure()->setWindkesselParams(7.13E07, 8.26E-10, 1.20E09, accumulated_time);
    outflow_pressure_condition2.getTargetPressure()->setWindkesselParams(7.13E07, 8.26E-10, 1.20E09, accumulated_time);
    outflow_pressure_condition3.getTargetPressure()->setWindkesselParams(6.02E07, 9.79E-10, 1.01E09, accumulated_time);
    outflow_pressure_condition4.getTargetPressure()->setWindkesselParams(6.89E07, 8.55E-10, 1.16E09, accumulated_time);
    outflow_pressure_condition5.getTargetPressure()->setWindkesselParams(9.80E06, 6.02E-09, 1.65E08, accumulated_time);

    /**
     * @brief 	Main loop starts here.
    */
    while (physical_time < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < Output_Time)
        {  
            time_instance = TickCount::now();
            Real Dt = get_fluid_advection_time_step_size.exec();          
            update_fluid_density.exec();
            kernel_correction_complex.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();

            /** FSI for viscous force. */
            viscous_force_from_fluid.exec();
            hemodynamic_indice_calculation.exec(Dt);

            interval_computing_time_step += TickCount::now() - time_instance;
            /** Dynamics including pressure relaxation. */
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);

                pressure_relaxation.exec(dt);
                /** FSI for pressure force. */
                pressure_force_on_wall.exec();

                kernel_summation.exec();

                emitter_buffer_inflow_condition.exec();  

                // windkessel model implementation
                if (physical_time >= updateP_n * accumulated_time)
                {
                    outflow_pressure_condition1.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition2.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition3.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition4.getTargetPressure()->updateNextPressure();
                    outflow_pressure_condition5.getTargetPressure()->updateNextPressure();
                   
                    ++updateP_n;
                }

                outflow_pressure_condition1.exec(dt); 
                outflow_pressure_condition2.exec(dt);
                outflow_pressure_condition3.exec(dt); 
                outflow_pressure_condition4.exec(dt); 
                outflow_pressure_condition5.exec(dt);    

                density_relaxation.exec(dt);

                relaxation_time += dt;
                integration_time += dt;
                physical_time += dt;
            }
            interval_computing_pressure_relaxation += TickCount::now() - time_instance;

            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9)
                          << "N=" << number_of_iterations
                          << "	Time = " << physical_time
                          << "	Dt = " << Dt << "	dt = " << dt << "\n";
            }
            number_of_iterations++;

            /** Update cell linked list and configuration. */
            time_instance = TickCount::now();

            /** Water block configuration and periodic condition. */
            inflow_injection.injection.exec();
            outflow_injection_1.injection.exec();
            outflow_injection_2.injection.exec();
            outflow_injection_3.injection.exec();
            outflow_injection_4.injection.exec();
            outflow_injection_5.injection.exec();

            inflow_injection.deletion.exec();
            outflow_injection_1.deletion.exec();
            outflow_injection_2.deletion.exec();
            outflow_injection_3.deletion.exec();
            outflow_injection_4.deletion.exec();
            outflow_injection_5.deletion.exec();

            if (number_of_iterations % 100 == 0 && number_of_iterations != 1)
            {
                particle_sorting.exec();
            }

            water_block.updateCellLinkedList();
            water_block_complex.updateConfiguration();
            wall_contact.updateConfiguration();

            interval_updating_configuration += TickCount::now() - time_instance;
            free_stream_surface_indicator.exec();

            inflow_injection.tag_buffer_particles.exec();
            outflow_injection_1.tag_buffer_particles.exec();
            outflow_injection_2.tag_buffer_particles.exec();
            outflow_injection_3.tag_buffer_particles.exec();
            outflow_injection_4.tag_buffer_particles.exec();
            outflow_injection_5.tag_buffer_particles.exec();
        }
        TickCount t2 = TickCount::now();
        body_states_recording.writeToFile();
        compute_inlet_transient_flow_rate.exec();
        compute_inlet_transient_mass_flow_rate.exec();

        TickCount t3 = TickCount::now();
        interval += t3 - t2;
    }
    TickCount t4 = TickCount::now();
    
    TimeInterval tt;
    tt = t4 - t1 - interval;
    std::cout << "Total wall time for computation: " << tt.seconds()
              << " seconds." << std::endl;
    std::cout << std::fixed << std::setprecision(9) << "interval_computing_time_step ="
              << interval_computing_time_step.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9) << "interval_computing_pressure_relaxation = "
              << interval_computing_pressure_relaxation.seconds() << "\n";
    std::cout << std::fixed << std::setprecision(9) << "interval_updating_configuration = "
              << interval_updating_configuration.seconds() << "\n";

    return 0;
}
