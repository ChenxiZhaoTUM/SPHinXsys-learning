/**
 * @file 	carotid_VIPO_wall.cpp
 * @brief 	Carotid artery with solid wall, imposed velocity inlet and pressure outlet condition.
 */

#include "sphinxsys.h"
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "particle_generation_and_detection.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_blood_file = "./input/fluid_cylinder_12.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real length_scale = 1.0;
Vec3d domain_lower_bound(-1.0 * length_scale, -2.5 * length_scale, -2.5 * length_scale);
Vec3d domain_upper_bound(13.0 * length_scale, 2.5 * length_scale, 2.5 * length_scale);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 3.0/30.0;
Vecd buffer_half = Vecd(2.0 * dp_0, 3.0 * length_scale, 3.0 * length_scale);
//----------------------------------------------------------------------
//	Define case dependent body shapes.
//----------------------------------------------------------------------
class WallBoundary : public ComplexShape
{
  public:
    explicit WallBoundary(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<ExtrudeShape<TriangleMeshShapeSTL>>(4.0 * dp_0, full_path_to_blood_file, translation, length_scale);
        subtract<TriangleMeshShapeSTL>(full_path_to_blood_file, translation, length_scale);
    }
};
//-----------------------------------------------------------------------------------------------------------
//	Main program starts here.
//-----------------------------------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem with global controls.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------

    SolidBody wall_boundary(sph_system, makeShared<WallBoundary>("WallBoundary"));
    wall_boundary.defineBodyLevelSetShape()->correctLevelSetSign()->writeLevelSet(sph_system);
    wall_boundary.defineMaterial<Solid>();
    wall_boundary.generateParticles<BaseParticles, Lattice>();
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    InnerRelation wall_inner(wall_boundary);

    BodyAlignedBoxByCell inlet_detection_box(wall_boundary,
                                             makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(-2.0 * dp_0, 0.0, 0.0)), buffer_half));
    BodyAlignedBoxByCell outlet_detection_box(wall_boundary,
                                              makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(12.0 * length_scale + 2.0 * dp_0, 0.0, 0.0)), buffer_half));

    RealBody test_body_in(sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(-2.0 * dp_0, 0.0, 0.0)), buffer_half), "TestBodyIn");
    test_body_in.generateParticles<BaseParticles, Lattice>();
    RealBody test_body_out(sph_system, makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(12.0 * length_scale + 2.0 * dp_0, 0.0, 0.0)), buffer_half), "TestBodyout");
    test_body_out.generateParticles<BaseParticles, Lattice>();

    using namespace relax_dynamics;
    SimpleDynamics<RandomizeParticlePosition> random_particles(wall_boundary);
    RelaxationStepInner relaxation_step_inner(wall_inner);

    // here, need a class to switch particles in aligned box to ghost particles (not real particles)
    SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
    SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_particles_detection(outlet_detection_box);

    /** Write the body state to Vtp file. */
    BodyStatesRecordingToVtp write_state_to_vtp(sph_system);
    /** Write the particle reload files. */
    ReloadParticleIO write_particle_reload_files({ &wall_boundary });
    ParticleSorting particle_sorting_wall(wall_boundary);
    //----------------------------------------------------------------------
    //	Physics relaxation starts here.
    //----------------------------------------------------------------------
    random_particles.exec(0.25);
    relaxation_step_inner.SurfaceBounding().exec();
    write_state_to_vtp.writeToFile(0.0);
    //----------------------------------------------------------------------
    // From here the time stepping begins.
    //----------------------------------------------------------------------
    int ite = 0;
    int relax_step = 4000;
    while (ite < relax_step)
    {
        relaxation_step_inner.exec();
        ite++;
        if (ite % 500 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite << "\n";
            write_state_to_vtp.writeToFile(ite);
        }
    }

    inlet_particles_detection.exec();
    particle_sorting_wall.exec();
    wall_boundary.updateCellLinkedList();
    outlet_particles_detection.exec();
    particle_sorting_wall.exec();
    wall_boundary.updateCellLinkedList();

    std::cout << "The physics relaxation process of wall particles finish !" << std::endl;

    write_state_to_vtp.writeToFile(ite);
    write_particle_reload_files.writeToFile(0);

    return 0;
}
