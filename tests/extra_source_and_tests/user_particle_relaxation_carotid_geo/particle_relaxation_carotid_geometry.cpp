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
//	Setting for the geometry.
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
Real dp_0 = 0.1;

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

// inlet R=2.9293, (1.5611, 5.8559, -30.8885), (0.1034, -0.0458, 0.9935) 
Vec3d inlet_half = Vec3d(3.0, 3.0, 0.35);
Vec3d inlet_translation = Vec3d(1.5921, 5.8422, -30.5904);
Vec3d inlet_normal(0.1034, -0.0458, 0.9935);
Vec3d inlet_standard_direction(0, 0, 1);
RotationResult inlet_rotation_result = RotationCalculator(inlet_normal, inlet_standard_direction);
Rotation3d inlet_rotation(inlet_rotation_result.angle, inlet_rotation_result.axis);

//outlet1 R=1.9416, (-2.6975, -0.4330, 21.7855), (-0.3160, -0.0009, 0.9488)
Vec3d outlet_01_half = Vec3d(2.0, 2.0, 0.35);
Vec3d outlet_01_translation = Vec3d(-2.6027, -0.4327, 21.5009);
Vec3d outlet_01_normal(-0.3160, -0.0009, 0.9488);
Vec3d outlet_01_standard_direction(0, 0, 1);
RotationResult outlet_01_rotation_result = RotationCalculator(outlet_01_normal, outlet_01_standard_direction);
Rotation3d outlet_01_rotation(outlet_01_rotation_result.angle, outlet_01_rotation_result.axis);

//outlet2 R=1.2760, (9.0465, 1.02552, 18.6363), (-0.0417, 0.0701, 0.9967)
Vec3d outlet_02_half = Vec3d(1.5, 1.5, 0.35);
Vec3d outlet_02_translation = Vec3d(9.0590, 1.0045, 18.3373);
Vec3d outlet_02_normal(-0.0417, 0.0701, 0.9967);
Vec3d outlet_02_standard_direction(0, 0, 1);
RotationResult outlet_02_rotation_result = RotationCalculator(outlet_02_normal, outlet_02_standard_direction);
Rotation3d outlet_02_rotation(outlet_02_rotation_result.angle, outlet_02_rotation_result.axis);
//----------------------------------------------------------------------
//	define the imported model.
//----------------------------------------------------------------------
class SolidBodyFromMesh : public ComplexShape
{
  public:
    explicit SolidBodyFromMesh(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
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
    imported_model.generateParticles<BaseParticles, Lattice>();

    // geo test
    SolidBody test_body_in(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(inlet_rotation), Vec3d(inlet_translation)), inlet_half, "TestBodyIn"));
    test_body_in.generateParticles<BaseParticles, Lattice>();

    SolidBody test_body_out01(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_01_rotation), Vec3d(outlet_01_translation)), outlet_01_half, "TestBodyOut01"));
    test_body_out01.generateParticles<BaseParticles, Lattice>();

    SolidBody test_body_out02(
        sph_system, makeShared<AlignedBoxShape>(Transform(Rotation3d(outlet_02_rotation), Vec3d(outlet_02_translation)), outlet_02_half, "TestBodyOut02"));
    test_body_out02.generateParticles<BaseParticles, Lattice>();

    //----------------------------------------------------------------------
    //	Define simple file input and outputs functions.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_body_states(sph_system);

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
