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
std::string full_path_to_file = "./input/carotid_fluid_geo.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real scaling = 1.0;
Vec3d domain_lower_bound(-7.0 * scaling, -4.0 * scaling, -35.0 * scaling);
Vec3d domain_upper_bound(20.0 * scaling, 12.0 * scaling, 30.0 * scaling);
//----------------------------------------------------------------------
//	Below are common parts for the two test geometries.
//----------------------------------------------------------------------
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 0.2;

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

// test buffer location
Real length = 8.0;
Real height = 4.0;
Vec3d test_half = Vec3d(length / 2, length / 2, height / 2);
Vec3d normal_test(-0.3134, 0.0009, 0.9496);
Vec3d center_location = Vec3d(-2.6733,-0.09, 21.7933);
Vec3d move_vector =  height / 2 * normal_test;
Vec3d test_translation = center_location + move_vector;
Vec3d standard_direction(0, 0, 1);
RotationResult test_rotation_result = RotationCalculator(normal_test, standard_direction);
Rotation3d test_rotation(test_rotation_result.angle, test_rotation_result.axis);


// inlet
Vec3d inlet_half = Vec3d(4.0, 4.0, 4.0);
Vec3d inlet_translation = Vec3d(2.26, 5.30, -30.97 - 4.0);
Vec3d normal_inlet(-0.1072, 0.0459, -0.9936);
Vec3d inlet_standard_direction(0, 0, 1);
RotationResult inlet_rotation_result = RotationCalculator(normal_inlet, inlet_standard_direction);
Rotation3d inlet_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);

//class InletBodyForSub : public ComplexShape
//{
//  public:
//    explicit InletBodyForSub(const std::string& shape_name) : ComplexShape(shape_name)
//    {
//        add<TransformShape<GeometricShapeBox>>(Transform(Rotation3d(test_rotation), Vec3d(test_translation)), test_half);
//    }
//};


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

        /*InletBodyForSub inlet_for_sub("InletForSub");
        subtract<InletBodyForSub>(inlet_for_sub);*/

        //add<TransformShape<GeometricShapeBox>>(Transform(Rotation3d(test_rotation), Vec3d(test_translation)), test_half);
        //cout << "angle = " << inlet_rotation_result.angle << endl;
        //cout << "axis = " << inlet_rotation_result.axis << endl;

        add<TransformShape<GeometricShapeBox>>(Transform(Rotation3d(Real(1.0), inlet_rotation_result.axis), Vec3d(test_translation)), test_half);
        //add<AlignedBoxShape>(Transform(Rotation3d(test_rotation), Vec3d(test_translation)), test_half);
    }
};
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
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    RealBody imported_model(sph_system, makeShared<SolidBodyFromMesh>("SolidBodyFromMesh"));
    // level set shape is used for particle relaxation
    imported_model.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
    //imported_model.defineBodyLevelSetShape()->writeLevelSet(sph_system);
    imported_model.defineParticlesAndMaterial();
    imported_model.generateParticles<Lattice>();


    // geo test
    SolidBody test_body(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(test_rotation), Vec3d(test_translation)), test_half, "TestBody"));
    test_body.defineParticlesAndMaterial<SolidParticles, Solid>();
    test_body.generateParticles<Lattice>();

    //----------------------------------------------------------------------
    //	Define simple file input and outputs functions.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_body_states(sph_system.real_bodies_);

    write_body_states.writeToFile();


    //MeshRecordingToPlt write_cell_linked_list(sph_system, imported_model.getCellLinkedList());
    if (sph_system.RunParticleRelaxation())
    {
        //----------------------------------------------------------------------
        //	Define body relation map.
        //	The contact map gives the topological connections between the bodies.
        //	Basically the the range of bodies to build neighbor particle lists.
        //  Generally, we first define all the inner relations, then the contact relations.
        //  At last, we define the complex relaxations by combining previous defined
        //  inner and contact relations.
        //----------------------------------------------------------------------
        InnerRelation imported_model_inner(imported_model);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_imported_model_particles(imported_model);
        /** A  Physics relaxation step. */
        RelaxationStepLevelSetCorrectionInner relaxation_step_inner(imported_model_inner);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_imported_model_particles.exec(0.25);
        relaxation_step_inner.SurfaceBounding().exec();
        write_body_states.writeToFile(0.0);
        imported_model.updateCellLinkedList();
        //write_cell_linked_list.writeToFile(0.0);
        //----------------------------------------------------------------------
        //	Particle relaxation time stepping start here.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 1000)
        {
            relaxation_step_inner.exec();
            ite_p += 1;
            if (ite_p % 100 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";
                write_body_states.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of imported model finish !" << std::endl;

    }
    return 0;
}
