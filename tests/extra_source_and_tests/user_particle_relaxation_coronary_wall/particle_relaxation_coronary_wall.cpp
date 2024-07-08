/**
 * @file 	particle_relaxation_single_resolution.cpp
 * @brief 	This is the test of using levelset to generate particles with single resolution and relax particles.
 * @details We use this case to test the particle generation and relaxation for a complex geometry.
 *			Before particle generation, we clean the sharp corners of the model.
 * @author 	Yongchuan Yu and Xiangyu Hu
 */

#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Setting for the first geometry.
//	To use this, please commenting the setting for the second geometry.
//----------------------------------------------------------------------
// std::string full_path_to_file = "./input/SPHinXsys.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
/*Vec3d domain_lower_bound(-2.3, -0.1, -0.3);
Vec3d domain_upper_bound(2.3, 4.5, 0.3);
Vecd translation(0.0, 0.0, 0.0);
Real scaling = 1.0; */
//----------------------------------------------------------------------
//	Setting for the second geometry.
//	To use this, please commenting the setting for the first geometry.
//----------------------------------------------------------------------
std::string full_path_to_file = "./input/normal_fluid_repair.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real scaling = pow(10, -3);
Vec3d domain_lower_bound(-375.0 * scaling, 100.0 * scaling, -340 * scaling);
Vec3d domain_upper_bound(-100.0 * scaling, 360.0 * scaling, 0.0 * scaling);
//----------------------------------------------------------------------
//	Below are common parts for the two test geometries.
//----------------------------------------------------------------------
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = (domain_upper_bound[0] - domain_lower_bound[0]) / 200.0;  // 1.375 * pow(10, -3)
//----------------------------------------------------------------------
//	define the imported model.
//----------------------------------------------------------------------
class SolidBodyFromMesh : public ComplexShape
{
  public:
    explicit SolidBodyFromMesh(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<ExtrudeShape<TriangleMeshShapeSTL>>(4.0 * dp_0, full_path_to_file, translation, scaling);
        subtract<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
    }
};

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

// inlet: R=41.7567, (-203.6015, 204.1509, -135.3577), (0.2987, 0.1312, 0.9445)
Vec3d inlet_half = Vec3d(2.0 * dp_0, 43.0 * scaling, 43.0 * scaling);
Vec3d inlet_normal(-0.2987, -0.1312, -0.9445);
Vec3d inlet_translation = Vec3d(-203.6015, 204.1509, -135.3577) * scaling + inlet_normal * 2.0 * dp_0;
Vec3d inlet_standard_direction(1, 0, 0);
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, inlet_standard_direction);
Rotation3d inlet_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);

// outlet main: R=36.1590, (-172.2628, 205.9036, -19.8868), (0.2678, 0.3191, -0.9084)
// intersection occurs!!!
Vec3d outlet_half_main = Vec3d(2.0 * dp_0, 45.0 * scaling, 45.0 * scaling);
Vec3d outlet_normal_main(-0.2678, -0.3191, 0.9084);
Vec3d outlet_translation_main = Vec3d(-172.2628, 205.9036, -19.8868) * scaling + outlet_normal_main * 2.0 * dp_0;
Vec3d outlet_standard_direction_main(1, 0, 0);
RotationResult outlet_rotation_result_main = RotationCalculator(outlet_normal_main, outlet_standard_direction_main);
Rotation3d outlet_rotation_main(outlet_rotation_result_main.angle, outlet_rotation_result_main.axis);

// outlet x_pos 01: R=2.6964, (-207.4362, 136.7848, -252.6892), (0.636, 0.771, -0.022)
Vec3d outlet_half_left_01 = Vec3d(2.0 * dp_0, 9.0 * scaling, 9.0 * scaling);
Vec3d outlet_normal_left_01(-0.636, -0.771, 0.022);
Vec3d outlet_translation_left_01 = Vec3d(-207.4362, 136.7848, -252.6892) * scaling + outlet_normal_left_01 * 2.0 * dp_0;
Vec3d outlet_standard_direction_left_01(1, 0, 0);
RotationResult outlet_rotation_result_left_01 = RotationCalculator(outlet_normal_left_01, outlet_standard_direction_left_01);
Rotation3d outlet_rotation_left_01(outlet_rotation_result_left_01.angle, outlet_rotation_result_left_01.axis);

// outlet x_pos 02: R=2.8306, (-193.2735, 337.4625, -270.2884), (-0.6714, 0.3331, -0.6620)
Vec3d outlet_half_left_02 = Vec3d(2.0 * dp_0, 10.0 * scaling, 10.0 * scaling);
Vec3d outlet_normal_left_02(-0.6714, 0.3331, -0.6620);
Vec3d outlet_translation_left_02 = Vec3d(-193.2735, 337.4625, -270.2884) * scaling + outlet_normal_left_02 * 2.0 * dp_0;
Vec3d outlet_standard_direction_left_02(1, 0, 0);
RotationResult outlet_rotation_result_left_02 = RotationCalculator(outlet_normal_left_02, outlet_standard_direction_left_02);
Rotation3d outlet_rotation_left_02(outlet_rotation_result_left_02.angle, outlet_rotation_result_left_02.axis);

// outlet x_pos 03: R=2.2804, (-165.5566, 326.1601, -139.9323), (0.6563, -0.6250, 0.4226)
Vec3d outlet_half_left_03 = Vec3d(2.0 * dp_0, 9.0 * scaling, 9.0 * scaling);
Vec3d outlet_normal_left_03(-0.6563, 0.6250, -0.4226);
Vec3d outlet_translation_left_03 = Vec3d(-165.5566, 326.1601, -139.9323) * scaling + outlet_normal_left_03 * 2.0 * dp_0;
Vec3d outlet_standard_direction_left_03(1, 0, 0);
RotationResult outlet_rotation_result_left_03 = RotationCalculator(outlet_normal_left_03, outlet_standard_direction_left_03);
Rotation3d outlet_rotation_left_03(outlet_rotation_result_left_03.angle, outlet_rotation_result_left_03.axis);

// outlet x_neg_front 01: R=2.6437, (-307.8, 312.1402, -333.2), (-0.185, -0.967, -0.176)
Vec3d outlet_half_rightF_01 = Vec3d(2.0 * dp_0, 10.0 * scaling, 10.0 * scaling);
Vec3d outlet_normal_rightF_01(-0.185, -0.967, -0.176);
Vec3d outlet_translation_rightF_01 = Vec3d(-307.8, 312.1402, -333.2) * scaling + outlet_normal_rightF_01 * 2.0 * dp_0;
Vec3d outlet_standard_direction_rightF_01(1, 0, 0);
RotationResult outlet_rotation_result_rightF_01 = RotationCalculator(outlet_normal_rightF_01, outlet_standard_direction_rightF_01);
Rotation3d outlet_rotation_rightF_01(outlet_rotation_result_rightF_01.angle, outlet_rotation_result_rightF_01.axis);

// outlet x_neg_front 02: R=1.5424, (-369.1252, 235.2617, -193.7022), (-0.501, 0.059, -0.863)
Vec3d outlet_half_rightF_02 = Vec3d(2.0 * dp_0, 8.0 * scaling, 8.0 * scaling);
Vec3d outlet_normal_rightF_02(-0.501, 0.059, -0.863);
Vec3d outlet_translation_rightF_02 = Vec3d(-369.1252, 235.2617, -193.7022) * scaling + outlet_normal_rightF_02 * 2.0 * dp_0;
Vec3d outlet_standard_direction_rightF_02(1, 0, 0);
RotationResult outlet_rotation_result_rightF_02 = RotationCalculator(outlet_normal_rightF_02, outlet_standard_direction_rightF_02);
Rotation3d outlet_rotation_rightF_02(outlet_rotation_result_rightF_02.angle, outlet_rotation_result_rightF_02.axis);

// outlet x_neg_behind 01: R=1.5743, (-268.3522, 116.0357, -182.4896), (0.325, -0.086, -0.942)
Vec3d outlet_half_rightB_01 = Vec3d(2.0 * dp_0, 8.0 * scaling, 8.0 * scaling);
Vec3d outlet_normal_rightB_01(0.325, -0.086, -0.942);
Vec3d outlet_translation_rightB_01 = Vec3d(-268.3522, 116.0357, -182.4896) * scaling + outlet_normal_rightB_01 * 2.0 * dp_0;
Vec3d outlet_standard_direction_rightB_01(1, 0, 0);
RotationResult outlet_rotation_result_rightB_01 = RotationCalculator(outlet_normal_rightB_01, outlet_standard_direction_rightB_01);
Rotation3d outlet_rotation_rightB_01(outlet_rotation_result_rightB_01.angle, outlet_rotation_result_rightB_01.axis);

// outlet x_neg_behind 02: R=1.8204, (-329.0846, 180.5258, -274.3232), (-0.1095, 0.9194, -0.3777)
Vec3d outlet_half_rightB_02 = Vec3d(2.0 * dp_0, 9.0 * scaling, 9.0 * scaling);
Vec3d outlet_normal_rightB_02(-0.1095, 0.9194, -0.3777);
Vec3d outlet_translation_rightB_02 = Vec3d(-329.0846, 180.5258, -274.3232) * scaling + outlet_normal_rightB_02 * 2.0 * dp_0;
Vec3d outlet_standard_direction_rightB_02(1, 0, 0);
RotationResult outlet_rotation_result_rightB_02 = RotationCalculator(outlet_normal_rightB_02, outlet_standard_direction_rightB_02);
Rotation3d outlet_rotation_rightB_02(outlet_rotation_result_rightB_02.angle, outlet_rotation_result_rightB_02.axis);

// outlet x_neg_behind 03: R=1.5491, (-342.1711, 197.1107, -277.8681), (0.1992, 0.5114, -0.8361)
Vec3d outlet_half_rightB_03 = Vec3d(2.0 * dp_0, 8.0 * scaling, 8.0 * scaling);
Vec3d outlet_normal_rightB_03(0.1992, 0.5114, -0.8361);
Vec3d outlet_translation_rightB_03 = Vec3d(-342.1711, 197.1107, -277.8681) * scaling + outlet_normal_rightB_03 * 2.0 * dp_0;
Vec3d outlet_standard_direction_rightB_03(1, 0, 0);
RotationResult outlet_rotation_result_rightB_03 = RotationCalculator(outlet_normal_rightB_03, outlet_standard_direction_rightB_03);
Rotation3d outlet_rotation_rightB_03(outlet_rotation_result_rightB_03.angle, outlet_rotation_result_rightB_03.axis);

// outlet x_neg_behind 04: R=2.1598, (-362.0112, 200.5693, -253.8417), (0.3694, 0.6067, -0.7044)
Vec3d outlet_half_rightB_04 = Vec3d(2.0 * dp_0, 9.0 * scaling, 9.0 * scaling);
Vec3d outlet_normal_rightB_04(0.3694, 0.6067, -0.7044);
Vec3d outlet_translation_rightB_04 = Vec3d(-362.0112, 200.5693, -253.8417) * scaling + outlet_normal_rightB_04 * 2.0 * dp_0;
Vec3d outlet_standard_direction_rightB_04(1, 0, 0);
RotationResult outlet_rotation_result_rightB_04 = RotationCalculator(outlet_normal_rightB_04, outlet_standard_direction_rightB_04);
Rotation3d outlet_rotation_rightB_04(outlet_rotation_result_rightB_04.angle, outlet_rotation_result_rightB_04.axis);

//-----------------------------------------------------------------------------------------------------------
//	Main program starts here.
//-----------------------------------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up -- a SPHSystem
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.setRunParticleRelaxation(false);
    sph_system.setReloadParticles(true);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    RealBody imported_model(sph_system, makeShared<SolidBodyFromMesh>("SolidBodyFromMesh"));
    // level set shape is used for particle relaxation
    imported_model.defineBodyLevelSetShape()->correctLevelSetSign()
        ->cleanLevelSet()->writeLevelSet(sph_system);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? imported_model.generateParticles<BaseParticles, Reload>(imported_model.getName())
        : imported_model.generateParticles<BaseParticles, Lattice>();

    /*RealBody test_body_in(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half, "TestBodyIn"));
    test_body_in.generateParticles<BaseParticles, Lattice>();
    BodyAlignedBoxByCell inlet_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half));*/

    /*RealBody test_body_out_main(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_main), Vec3d(outlet_translation_main)), outlet_half_main, "TestBodyOutMain"));
    test_body_out_main.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_main_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_main), Vec3d(outlet_translation_main)), outlet_half_main));

    /*RealBody test_body_out_left01(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_left_01), Vec3d(outlet_translation_left_01)), outlet_half_left_01, "TestBodyOutLeft01"));
    test_body_out_left01.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_left01_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_left_01), Vec3d(outlet_translation_left_01)), outlet_half_left_01));

    /*RealBody test_body_out_left02(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_left_02), Vec3d(outlet_translation_left_02)), outlet_half_left_02, "TestBodyOutLeft02"));
    test_body_out_left02.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_left02_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_left_02), Vec3d(outlet_translation_left_02)), outlet_half_left_02));

    /*RealBody test_body_out_left03(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_left_03), Vec3d(outlet_translation_left_03)), outlet_half_left_03, "TestBodyOutLeft03"));
    test_body_out_left03.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_left03_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_left_03), Vec3d(outlet_translation_left_03)), outlet_half_left_03));

    /*RealBody test_body_out_rightF01(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightF_01), Vec3d(outlet_translation_rightF_01)), outlet_half_rightF_01, "TestBodyOutRightF01"));
    test_body_out_rightF01.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightF01_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightF_01), Vec3d(outlet_translation_rightF_01)), outlet_half_rightF_01));

    /*RealBody test_body_out_rightF02(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightF_02), Vec3d(outlet_translation_rightF_02)), outlet_half_rightF_02, "TestBodyOutRightF02"));
    test_body_out_rightF02.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightF02_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightF_02), Vec3d(outlet_translation_rightF_02)), outlet_half_rightF_02));
    
    /*RealBody test_body_out_rightB01(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightB_01), Vec3d(outlet_translation_rightB_01)), outlet_half_rightB_01, "TestBodyOutRightB01"));
    test_body_out_rightB01.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightB01_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightB_01), Vec3d(outlet_translation_rightB_01)), outlet_half_rightB_01));

    /*RealBody test_body_out_rightB02(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightB_02), Vec3d(outlet_translation_rightB_02)), outlet_half_rightB_02, "TestBodyOutRightB02"));
    test_body_out_rightB02.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightB02_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightB_02), Vec3d(outlet_translation_rightB_02)), outlet_half_rightB_02));

    /*RealBody test_body_out_rightB03(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightB_03), Vec3d(outlet_translation_rightB_03)), outlet_half_rightB_03, "TestBodyOutRightB03"));
    test_body_out_rightB03.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightB03_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightB_03), Vec3d(outlet_translation_rightB_03)), outlet_half_rightB_03));

    /*RealBody test_body_out_rightB04(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightB_04), Vec3d(outlet_translation_rightB_04)), outlet_half_rightB_04, "TestBodyOutRightB04"));
    test_body_out_rightB04.generateParticles<BaseParticles, Lattice>();*/
    BodyAlignedBoxByCell outlet_rightB04_detection_box(imported_model, 
        makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_rotation_rightB_04), Vec3d(outlet_translation_rightB_04)), outlet_half_rightB_04));

    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    if (sph_system.RunParticleRelaxation())
    {
        InnerRelation imported_model_inner(imported_model);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_imported_model_particles(imported_model);
        /** A  Physics relaxation step. */
        RelaxationStepLevelSetCorrectionInner relaxation_step_inner(imported_model_inner);

        //SimpleDynamics<AlignedBoxParticlesDetection> inlet_particles_detection(inlet_detection_box, xAxis);
        SimpleDynamics<AlignedBoxParticlesDetection> outlet_main_particles_detection(outlet_main_detection_box, xAxis);
        SimpleDynamics<AlignedBoxParticlesDetection> outlet_left01_particles_detection(outlet_left01_detection_box, xAxis);
        SimpleDynamics<AlignedBoxParticlesDetection> outlet_left02_particles_detection(outlet_left02_detection_box, xAxis);
        SimpleDynamics<AlignedBoxParticlesDetection> outlet_left03_particles_detection(outlet_left03_detection_box, xAxis);
        SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightF01_particles_detection(outlet_rightF01_detection_box, xAxis);
        SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightF02_particles_detection(outlet_rightF02_detection_box, xAxis);
        SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightB01_particles_detection(outlet_rightB01_detection_box, xAxis);
        SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightB02_particles_detection(outlet_rightB02_detection_box, xAxis);
        SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightB03_particles_detection(outlet_rightB03_detection_box, xAxis);
        SimpleDynamics<AlignedBoxParticlesDetection> outlet_rightB04_particles_detection(outlet_rightB04_detection_box, xAxis);

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_imported_model_to_vtp({imported_model});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(imported_model);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_imported_model_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        write_imported_model_to_vtp.writeToFile(0.0);
        imported_model.updateCellLinkedList();
        // write_cell_linked_list.writeToFile(0.0);
        //----------------------------------------------------------------------
        //	Particle relaxation time stepping start here.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 500)
        {
            relaxation_step_inner.exec();
            ite_p += 1;
            if (ite_p % 50 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";
                write_imported_model_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of imported model finish !" << std::endl;

        //inlet_particles_detection.exec();
        outlet_main_particles_detection.exec();
        outlet_left01_particles_detection.exec();
        outlet_left02_particles_detection.exec();
        outlet_left03_particles_detection.exec();
        outlet_rightF01_particles_detection.exec();
        outlet_rightF02_particles_detection.exec();
        outlet_rightB01_particles_detection.exec();
        outlet_rightB02_particles_detection.exec();
        outlet_rightB03_particles_detection.exec();
        outlet_rightB04_particles_detection.exec();

        write_imported_model_to_vtp.writeToFile(ite_p);
        write_particle_reload_files.writeToFile(0);
        return 0;
    }

    BodyStatesRecordingToVtp write_body_states(sph_system);
    write_body_states.writeToFile();
    return 0;
}
