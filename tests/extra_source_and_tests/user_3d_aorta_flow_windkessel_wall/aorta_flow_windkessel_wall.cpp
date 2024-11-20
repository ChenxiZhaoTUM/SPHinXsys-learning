/**
 * @file 	pulsatile_Aorta_flow.cpp
 * @brief 	3D pulsatile Aorta flow example
 * @details This is the one of the basic test cases for pressure boundary condition and bidirectional buffer.
 * @author 	Shuoguo Zhang and Xiangyu Hu
 */
/**
 * @brief 	SPHinXsys Library.
 */
#include "sphinxsys.h" 
#include "density_correciton.h"
#include "density_correciton.hpp"
#include "pressure_boundary.h"
#include "bidirectional_buffer.h"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "arbitrary_shape_buffer.h"
#include "arbitrary_shape_buffer_3d.h"

/**
 * @brief Namespace cite here.
 */
using namespace SPH;

std::string full_path_to_file = "./input/aorta_blood_domain.stl";

/** Domain bounds of the system. */
BoundingBox system_domain_bounds(Vecd(-6.0E-2, -4.0E-2, -2.0E-2), Vecd(3.0E-2, 10.0E-2, 15.0E-2));

Real rho0_f = 1060.0;
Real mu_f = 0.00355;
Real U_f = 2.0;
Real c_f = 10.0*U_f;

Real dp_0 = 0.06E-2;
Real wall_resolution = 0.5 * dp_0;
Real scaling = 1.0E-2;
Vecd translation(0.0, 0.0, 0.0);
StdVec<Vecd> observer_location = {Vecd(-1.24E-2, 4.41E-2, 5.18E-2)};

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

// inlet (-0.9768, 4.6112, 3.0052), (0.1000, 0.1665, 0.9810)
Real A_in = 5.9765 * scaling * scaling;
Vec3d inlet_half = Vec3d(2.0 * dp_0, 1.8 * scaling, 1.8 * scaling);
Vec3d inlet_vector(0.1000, 0.1665, 0.9810);
Vec3d inlet_normal = inlet_vector.normalized();
Vec3d inlet_center = Vec3d(-0.9768, 4.6112, 3.0052) * scaling - inlet_normal * (2.0 * dp_0);
Vec3d inlet_cut_translation = inlet_center - inlet_normal * (2.7 * dp_0);
Vec3d inlet_buffer_translation = inlet_center + inlet_normal * (2.0 * dp_0);
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, standard_direction);
Rotation3d inlet_emitter_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);
Rotation3d inlet_disposer_rotation(inlet_rotation_result.angle + Pi, inlet_rotation_result.axis);

// outlet1 (-1.2562, 4.4252, 10.0148), (0.6420, 0.4110, 0.6472)
Real A_out1 = 0.2688 * scaling * scaling;
Vec3d outlet_1_half = Vec3d(2.0 * dp_0, 1.0 * scaling, 1.0 * scaling);
Vec3d outlet_1_vector(0.6420, 0.4110, 0.6472);
Vec3d outlet_1_normal = outlet_1_vector.normalized();
Vec3d outlet_1_center = Vec3d(-1.2562, 4.4252, 10.0148) * scaling + outlet_1_normal * (2.0 * dp_0);
Vec3d outlet_1_cut_translation = outlet_1_center + outlet_1_normal * (1.5 * dp_0);
Vec3d outlet_1_buffer_translation = outlet_1_center - outlet_1_normal * (2.0 * dp_0);
RotationResult outlet_1_rotation_result = RotationCalculator(outlet_1_normal, standard_direction);
Rotation3d outlet_1_disposer_rotation(outlet_1_rotation_result.angle, outlet_1_rotation_result.axis);
Rotation3d outlet_1_emitter_rotation(outlet_1_rotation_result.angle + Pi, outlet_1_rotation_result.axis);

// outlet2 (-2.6303, 3.0594, 10.6919), (-0.0988, 0.0485, 0.9939)
Real A_out2 = 0.2230 * scaling * scaling;
Vec3d outlet_2_half = Vec3d(2.0 * dp_0, 1.0 * scaling, 1.0 * scaling);
Vec3d outlet_2_vector(-0.0988, 0.0485, 0.9939);
Vec3d outlet_2_normal = outlet_2_vector.normalized();
Vec3d outlet_2_center = Vec3d(-2.6303, 3.0594, 10.6919) * scaling + outlet_2_normal * (2.0 * dp_0);
Vec3d outlet_2_cut_translation = outlet_2_center + outlet_2_normal * (1.5 * dp_0);
Vec3d outlet_2_buffer_translation = outlet_2_center - outlet_2_normal * (2.0 * dp_0);
RotationResult outlet_2_rotation_result = RotationCalculator(outlet_2_normal, standard_direction);
Rotation3d outlet_2_disposer_rotation(outlet_2_rotation_result.angle, outlet_2_rotation_result.axis);
Rotation3d outlet_2_emitter_rotation(outlet_2_rotation_result.angle + Pi, outlet_2_rotation_result.axis);

// outlet3 (-2.8585, 1.8357, 9.8034), (-0.1471, -0.1813, 0.9724)
Real A_out3 = 0.3948 * scaling * scaling;
Vec3d outlet_3_half = Vec3d(2.0 * dp_0, 0.4 * scaling, 0.4 * scaling);
Vec3d outlet_3_vector(-0.1471, -0.1813, 0.9724);
Vec3d outlet_3_normal = outlet_3_vector.normalized();
Vec3d outlet_3_center = Vec3d(-2.8585, 1.8357, 9.8034) * scaling + outlet_3_normal * (2.0 * dp_0);
Vec3d outlet_3_cut_translation = outlet_3_center + outlet_3_normal * (1.5 * dp_0);
Vec3d outlet_3_buffer_translation = outlet_3_center - outlet_3_normal * (2.0 * dp_0);
RotationResult outlet_3_rotation_result = RotationCalculator(outlet_3_normal, standard_direction);
Rotation3d outlet_3_disposer_rotation(outlet_3_rotation_result.angle, outlet_3_rotation_result.axis);
Rotation3d outlet_3_emitter_rotation(outlet_3_rotation_result.angle + Pi, outlet_3_rotation_result.axis);

// outlet4 (-1.0946, 1.0386, 9.5016), (0.5675, 0.4280, 0.7034)
Real A_out4 = 0.5134 * scaling * scaling;
Vec3d outlet_4_half = Vec3d(2.0 * dp_0, 1.0 * scaling, 1.0 * scaling);
Vec3d outlet_4_vector(0.5675, 0.4280, 0.7034);
Vec3d outlet_4_normal = outlet_4_vector.normalized();
Vec3d outlet_4_center = Vec3d(-1.0946, 1.0386, 9.5016) * scaling + outlet_4_normal * (2.0 * dp_0);
Vec3d outlet_4_cut_translation = outlet_4_center + outlet_4_normal * (1.5 * dp_0);
Vec3d outlet_4_buffer_translation = outlet_4_center - outlet_4_normal * (2.0 * dp_0);
RotationResult outlet_4_rotation_result = RotationCalculator(outlet_4_normal, standard_direction);
Rotation3d outlet_4_disposer_rotation(outlet_4_rotation_result.angle, outlet_4_rotation_result.axis);
Rotation3d outlet_4_emitter_rotation(outlet_4_rotation_result.angle + Pi, outlet_4_rotation_result.axis);

// outlet5 (-1.6791, -0.8069, 0.5017), (0.0327, -0.0729, 0.9968)
Real A_out5 = 2.67 * scaling * scaling;
Vec3d outlet_5_half = Vec3d(2.0 * dp_0, 2.0 * scaling, 2.0 * scaling);
Vec3d outlet_5_vector(-0.0327, 0.0729, -0.9968);
Vec3d outlet_5_normal = outlet_5_vector.normalized();
Vec3d outlet_5_center = Vec3d(-1.6791, -0.8069, 0.5017) * scaling + outlet_5_normal * (2.0 * dp_0);
Vec3d outlet_5_cut_translation = outlet_5_center + outlet_5_normal * (1.5 * dp_0);
Vec3d outlet_5_buffer_translation = outlet_5_center - outlet_5_normal * (2.0 * dp_0);
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
        add<ExtrudeShape<TriangleMeshShapeSTL>>(4 * wall_resolution, full_path_to_file, translation, scaling);
        subtract<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
    }
};

struct InflowVelocity
{
    Real u_ave;

    template <class BoundaryConditionType>
    InflowVelocity(BoundaryConditionType &boundary_condition)
        : u_ave(0.0) {}

    Vecd operator()(Vecd &position, Vecd &velocity)
    {
        Vecd target_velocity = velocity;
        Real run_time = GlobalStaticVariables::physical_time_;

        u_ave = 0.3782;
        Real a[8] = {-0.1812,0.1276,-0.08981,0.04347,-0.05412,0.02642,0.008946,-0.009005};
        Real b[8] = {-0.07725,0.01466,0.004295,-0.06679,0.05679,-0.01878,0.01869,-0.01888};
        for (size_t i = 0; i < 8; i++)
        {
            u_ave = u_ave + a[i] * cos(8.302 * (i + 1) * run_time) + b[i] * sin(8.302 * (i + 1) * run_time);
        }
            
        target_velocity[0] = u_ave;
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
    /** Tag for run particle relaxation for the initial body fitted distribution. */
    sph_system.setRunParticleRelaxation(false);
    /** Tag for computation start with relaxed body fitted particles distribution. */
    sph_system.setReloadParticles(true);
    /** handle command line arguments. */
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    /**
     * @brief Material property, particles and body creation of fluid.
     */
    FluidBody water_block(sph_system, makeShared<WaterBlock>("WaterBody"));
    water_block.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
    //water_block.defineBodyLevelSetShape()->correctLevelSetSign();
    water_block.defineMaterial<WeaklyCompressibleFluid>(rho0_f, c_f, mu_f);
    ParticleBuffer<ReserveSizeFactor> in_outlet_particle_buffer(0.5);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? water_block.generateParticlesWithReserve<BaseParticles, Reload>(in_outlet_particle_buffer, water_block.getName())
        : water_block.generateParticles<BaseParticles, Lattice>();
    /**
     * @brief 	Particle and body creation of wall boundary.
     */
    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("Wall"));
    wall_boundary.defineAdaptationRatios(1.15, 2.0);
    wall_boundary.defineBodyLevelSetShape()->writeLevelSet(sph_system);
    //wall_boundary.defineBodyLevelSetShape();
    wall_boundary.defineMaterial<Solid>();
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? wall_boundary.generateParticles<BaseParticles, Reload>(wall_boundary.getName())
        : wall_boundary.generateParticles<BaseParticles, Lattice>();

    ObserverBody velocity_observer(sph_system, "VelocityObserver");
    velocity_observer.generateParticles<ObserverParticles>(observer_location);
    /** topology */
    InnerRelation water_block_inner(water_block);
    ContactRelation water_block_contact(water_block, {&wall_boundary});
    ContactRelation velocity_observer_contact(velocity_observer, {&water_block});
    ComplexRelation water_block_complex(water_block_inner, water_block_contact);
    if (sph_system.RunParticleRelaxation())
    {
        /** body topology only for particle relaxation */
        InnerRelation water_block_inner(water_block);
        InnerRelation wall_boundary_inner(wall_boundary);
        BodyAlignedCylinderByCell inlet_detection_cylinder(wall_boundary,
                                                           makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_half[1], inlet_half[0]));
        BodyAlignedBoxByCell outlet01_detection_box(wall_boundary,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_emitter_rotation), Vec3d(outlet_1_cut_translation)), outlet_1_half));
        BodyAlignedBoxByCell outlet02_detection_box(wall_boundary,
                                                makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_emitter_rotation), Vec3d(outlet_2_cut_translation)), outlet_2_half));
        BodyAlignedBoxByCell outlet03_detection_box(wall_boundary,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_emitter_rotation), Vec3d(outlet_3_cut_translation)), outlet_3_half));
        BodyAlignedBoxByCell outlet04_detection_box(wall_boundary,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_emitter_rotation), Vec3d(outlet_4_cut_translation)), outlet_4_half));
        BodyAlignedBoxByCell outlet05_detection_box(wall_boundary,
                                                    makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_emitter_rotation), Vec3d(outlet_5_cut_translation)), outlet_5_half));
        RealBody test_body_in(
            sph_system, makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_cut_translation)), inlet_half[1], inlet_half[0], "TestBodyIn"));
        test_body_in.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_1(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_emitter_rotation), Vec3d(outlet_1_cut_translation)), outlet_1_half, "TestBodyOut01"));
        test_body_out_1.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_2(
        sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_emitter_rotation), Vec3d(outlet_2_cut_translation)), outlet_2_half, "TestBodyOut02"));
        test_body_out_2.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_3(
            sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_emitter_rotation), Vec3d(outlet_3_cut_translation)), outlet_3_half, "TestBodyOut03"));
        test_body_out_3.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_4(
            sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_emitter_rotation), Vec3d(outlet_4_cut_translation)), outlet_4_half, "TestBodyOut04"));
        test_body_out_4.generateParticles<BaseParticles, Lattice>();

        RealBody test_body_out_5(
            sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_emitter_rotation), Vec3d(outlet_5_cut_translation)), outlet_5_half, "TestBodyOut05"));
        test_body_out_5.generateParticles<BaseParticles, Lattice>();
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        /** Random reset the insert body particle position. */
        SimpleDynamics<RandomizeParticlePosition> water_block_random_inserted_body_particles(water_block);
        SimpleDynamics<RandomizeParticlePosition> wall_boundary_random_inserted_body_particles(wall_boundary);

        // here, need a class to switch particles in aligned box to ghost particles (not real particles)
        SimpleDynamics<DeleteParticlesInCylinder> inlet_particles_detection(inlet_detection_cylinder);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet01_particles_detection(outlet01_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet02_particles_detection(outlet02_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet03_particles_detection(outlet03_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet04_particles_detection(outlet04_detection_box);
        SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet05_particles_detection(outlet05_detection_box);

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp water_block_write_inserted_body_to_vtp(water_block);
        BodyStatesRecordingToVtp wall_boundary_write_inserted_body_to_vtp(wall_boundary);
        BodyStatesRecordingToVtp write_all_bodies_to_vtp({sph_system});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files({ &wall_boundary, &water_block });
        /** A  Physics relaxation step. */
        RelaxationStepInner water_block_relaxation_step_inner(water_block_inner);
        RelaxationStepInner wall_boundary_relaxation_step_inner(wall_boundary_inner);
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
        while (ite_p < 200)
        {
            water_block_relaxation_step_inner.exec();
            wall_boundary_relaxation_step_inner.exec();
            ite_p += 1;
            if (ite_p % 100 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the inserted body N = " << ite_p << "\n";
                water_block_write_inserted_body_to_vtp.writeToFile(ite_p);
                wall_boundary_write_inserted_body_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of inserted body finish !" << std::endl;

        inlet_particles_detection.exec();
        wall_boundary.updateCellLinkedListWithParticleSort(100);
        outlet01_particles_detection.exec();
        wall_boundary.updateCellLinkedListWithParticleSort(100);
        outlet02_particles_detection.exec();
        wall_boundary.updateCellLinkedListWithParticleSort(100);
        outlet03_particles_detection.exec();
        wall_boundary.updateCellLinkedListWithParticleSort(100);
        outlet04_particles_detection.exec();
        wall_boundary.updateCellLinkedListWithParticleSort(100);
        outlet05_particles_detection.exec();
        wall_boundary.updateCellLinkedListWithParticleSort(100);

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
    InteractionDynamics<NablaWVComplex> kernel_summation(water_block_inner, water_block_contact);
    InteractionWithUpdate<SpatialTemporalFreeSurfaceIndicationComplex>
        free_stream_surface_indicator(water_block_inner, water_block_contact);
    /** Pressure relaxation algorithm without Riemann solver for viscous flows. */
    Dynamics1Level<fluid_dynamics::Integration1stHalfWithWallRiemann> pressure_relaxation(water_block_inner, water_block_contact);
    /** Pressure relaxation algorithm by using position verlet time stepping. */
    Dynamics1Level<fluid_dynamics::Integration2ndHalfWithWallRiemann> density_relaxation(water_block_inner, water_block_contact);
    /* Time step size without considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AdvectionTimeStepSize> get_fluid_advection_time_step_size(water_block, U_f);
    /** Time step size with considering sound wave speed. */
    ReduceDynamics<fluid_dynamics::AcousticTimeStepSize> get_fluid_time_step_size(water_block);
    /** Computing viscous acceleration. */
    InteractionWithUpdate<fluid_dynamics::ViscousForceWithWall> viscous_acceleration(water_block_inner, water_block_contact);
    /** Impose transport velocity. */
    InteractionWithUpdate<fluid_dynamics::TransportVelocityCorrectionComplex<BulkParticles>>
        transport_velocity_correction(water_block_inner, water_block_contact);

    // disposer
    BodyAlignedCylinderByCell inlet_disposer(water_block, makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_disposer_rotation), Vec3d(inlet_buffer_translation)), inlet_half[1], inlet_half[0]));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionArb<AlignedCylinderShape>> inlet_disposer_outflow_deletion(inlet_disposer);
    BodyAlignedBoxByCell disposer_1(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_disposer_rotation), Vec3d(outlet_1_buffer_translation)), outlet_1_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_1(disposer_1, "out01");
    BodyAlignedBoxByCell disposer_2(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_disposer_rotation), Vec3d(outlet_2_buffer_translation)), outlet_2_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_2(disposer_2, "out02");
    BodyAlignedBoxByCell disposer_3(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_disposer_rotation), Vec3d(outlet_3_buffer_translation)), outlet_3_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_3(disposer_3, "out03");
    BodyAlignedBoxByCell disposer_4(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_disposer_rotation), Vec3d(outlet_4_buffer_translation)), outlet_4_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_4(disposer_4, "out04");
    BodyAlignedBoxByCell disposer_5(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_disposer_rotation), Vec3d(outlet_5_buffer_translation)), outlet_5_half));
    SimpleDynamics<fluid_dynamics::DisposerOutflowDeletionWithWindkessel> disposer_outflow_deletion_5(disposer_5, "out05");

    // bidirectional buffer
    BodyAlignedCylinderByCell inlet_emitter(water_block, makeShared<AlignedCylinderShape>(xAxis, Transform(Rotation3d(inlet_emitter_rotation), Vec3d(inlet_buffer_translation)), inlet_half[1], inlet_half[0]));
    fluid_dynamics::NonPrescribedPressureBidirectionalBufferArb<AlignedCylinderShape> inlet_emitter_inflow_injection(inlet_emitter, in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_1(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_1_emitter_rotation), Vec3d(outlet_1_buffer_translation)), outlet_1_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_1(outflow_emitter_1, "out01", in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_2(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_2_emitter_rotation), Vec3d(outlet_2_buffer_translation)), outlet_2_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_2(outflow_emitter_2, "out02", in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_3(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_3_emitter_rotation), Vec3d(outlet_3_buffer_translation)), outlet_3_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_3(outflow_emitter_3, "out03", in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_4(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_4_emitter_rotation), Vec3d(outlet_4_buffer_translation)), outlet_4_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_4(outflow_emitter_4, "out04", in_outlet_particle_buffer);
    BodyAlignedBoxByCell outflow_emitter_5(water_block, makeShared<AlignedBoxShape>(xAxis, Transform(Rotation3d(outlet_5_emitter_rotation), Vec3d(outlet_5_buffer_translation)), outlet_5_half));
    fluid_dynamics::WindkesselOutletBidirectionalBuffer outflow_injection_5(outflow_emitter_5, "out05", in_outlet_particle_buffer);

    InteractionWithUpdate<fluid_dynamics::DensitySummationPressureComplex> update_fluid_density(water_block_inner, water_block_contact);
    SimpleDynamics<fluid_dynamics::InflowVelocityConditionArb<InflowVelocity, AlignedCylinderShape>> emitter_buffer_inflow_condition(inlet_emitter);
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition1(outflow_emitter_1, "out01");
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition2(outflow_emitter_2, "out02");
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition3(outflow_emitter_3, "out03");
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition4(outflow_emitter_4, "out04");
    SimpleDynamics<fluid_dynamics::WindkesselBoundaryCondition> outflow_pressure_condition5(outflow_emitter_5, "out05");

    /**
     * @brief Output.
     */
    //----------------------------------------------------------------------
    //	Define the configuration related particles dynamics.
    //----------------------------------------------------------------------
    /** Output the body states. */
    BodyStatesRecordingToVtp body_states_recording(sph_system);
    body_states_recording.addToWrite<Real>(water_block, "Pressure");
    body_states_recording.addToWrite<int>(water_block, "Indicator");
    body_states_recording.addToWrite<Real>(water_block, "PositionDivergence");
    body_states_recording.addToWrite<Real>(water_block, "Density");
    body_states_recording.addToWrite<int>(water_block, "BufferParticleIndicator");

    ObservedQuantityRecording<Vecd>
        write_centerline_velocity("Velocity", velocity_observer_contact);
    /**
     * @brief Setup geometry and initial conditions.
     */
    sph_system.initializeSystemCellLinkedLists(); 
    sph_system.initializeSystemConfigurations();
    free_stream_surface_indicator.exec();
    inlet_emitter_inflow_injection.tag_buffer_particles.exec();
    outflow_injection_1.tag_buffer_particles.exec();
    outflow_injection_2.tag_buffer_particles.exec();
    outflow_injection_3.tag_buffer_particles.exec();
    outflow_injection_4.tag_buffer_particles.exec();
    outflow_injection_5.tag_buffer_particles.exec();

    wall_boundary_normal_direction.exec();
    
    /**
     * @brief 	Basic parameters.
     */
    size_t number_of_iterations = 0.0;
    int screen_output_interval = 100;
    int observation_sample_interval = screen_output_interval * 2;
    Real end_time = 2.0;   /**< End time. */
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
    write_centerline_velocity.writeToFile(number_of_iterations);

    // why -average_Q??
    outflow_pressure_condition1.getTargetPressure()->setWindkesselParams(1.18E8, 7.7E-10, 1.84E9, accumulated_time, 0.0000098);
    outflow_pressure_condition2.getTargetPressure()->setWindkesselParams(1.04E8, 8.74E-10, 1.63E9, accumulated_time, 0.00001);
    outflow_pressure_condition3.getTargetPressure()->setWindkesselParams(1.18E8, 7.7E-10, 1.84E9, accumulated_time, 0.0000068);
    outflow_pressure_condition4.getTargetPressure()->setWindkesselParams(9.7E7, 9.34E-10, 1.52E9, accumulated_time, 0.0000118);
    outflow_pressure_condition5.getTargetPressure()->setWindkesselParams(1.88E7, 4.82E-9, 2.95E8, accumulated_time, 0.000096);

    //outflow_pressure_condition1.getTargetPressure()->setWindkesselParams(1.18E8, 7.7E-10, 1.84E9, accumulated_time, 0);
    //outflow_pressure_condition2.getTargetPressure()->setWindkesselParams(1.04E8, 8.74E-10, 1.63E9, accumulated_time, 0);
    //outflow_pressure_condition3.getTargetPressure()->setWindkesselParams(1.18E8, 7.7E-10, 1.84E9, accumulated_time, 0);
    //outflow_pressure_condition4.getTargetPressure()->setWindkesselParams(9.7E7, 9.34E-10, 1.52E9, accumulated_time, 0);
    //outflow_pressure_condition5.getTargetPressure()->setWindkesselParams(1.88E7, 4.82E-9, 2.95E8, accumulated_time, 0);
    /**
     * @brief 	Main loop starts here.
    */
    while (GlobalStaticVariables::physical_time_ < end_time)
    {
        Real integration_time = 0.0;
        /** Integrate time (loop) until the next output time. */
        while (integration_time < Output_Time)
        {  
            time_instance = TickCount::now();
            Real Dt = get_fluid_advection_time_step_size.exec();          
            update_fluid_density.exec();
            viscous_acceleration.exec();
            transport_velocity_correction.exec();
            interval_computing_time_step += TickCount::now() - time_instance;
            /** Dynamics including pressure relaxation. */
            time_instance = TickCount::now();
            Real relaxation_time = 0.0;
            while (relaxation_time < Dt)
            {
                dt = SMIN(get_fluid_time_step_size.exec(), Dt);
                pressure_relaxation.exec(dt);
                kernel_summation.exec();

                emitter_buffer_inflow_condition.exec();  

                // windkessel model implementation

                if (GlobalStaticVariables::physical_time_ >= updateP_n * accumulated_time)
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
                GlobalStaticVariables::physical_time_ += dt;
            }
            interval_computing_pressure_relaxation += TickCount::now() - time_instance;
            if (number_of_iterations % screen_output_interval == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "N=" << number_of_iterations << "	Time = "
                          << GlobalStaticVariables::physical_time_
                          << "	Dt = " << Dt << "	dt = " << dt << "\n";

                if (number_of_iterations % observation_sample_interval == 0 && number_of_iterations != sph_system.RestartStep())
                {
                    write_centerline_velocity.writeToFile(number_of_iterations);
                }
            }
            number_of_iterations++;
            /** Update cell linked list and configuration. */
            time_instance = TickCount::now();
            /** Water block configuration and periodic condition. */
            inlet_emitter_inflow_injection.injection.exec();
            outflow_injection_1.injection.exec();
            outflow_injection_2.injection.exec();
            outflow_injection_3.injection.exec();
            outflow_injection_4.injection.exec();
            outflow_injection_5.injection.exec();
            inlet_disposer_outflow_deletion.exec();
            disposer_outflow_deletion_1.exec();
            disposer_outflow_deletion_2.exec();
            disposer_outflow_deletion_3.exec();
            disposer_outflow_deletion_4.exec();
            disposer_outflow_deletion_5.exec();

            water_block.updateCellLinkedList();
            water_block_contact.updateConfiguration();
            water_block_complex.updateConfiguration();
            interval_updating_configuration += TickCount::now() - time_instance;
            free_stream_surface_indicator.exec();

            inlet_emitter_inflow_injection.tag_buffer_particles.exec();
            outflow_injection_1.tag_buffer_particles.exec();
            outflow_injection_2.tag_buffer_particles.exec();
            outflow_injection_3.tag_buffer_particles.exec();
            outflow_injection_4.tag_buffer_particles.exec();
            outflow_injection_5.tag_buffer_particles.exec();
        }
        TickCount t2 = TickCount::now();
        body_states_recording.writeToFile();  
        velocity_observer_contact.updateConfiguration();
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
