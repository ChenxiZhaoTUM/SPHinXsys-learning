/**
 * @file 	airfoil_2d.cpp
 * @brief 	This is the test of using levelset to generate body fitted SPH particles.
 * @details	We use this case to test the particle generation and relaxation with a complex geometry (2D).
 *			Before the particles are generated, we clean the sharp corners and other unresolvable surfaces.
 * @author 	Yongchuan Yu and Xiangyu Hu
 */
#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Set the file path to the data file.
//----------------------------------------------------------------------
std::string naca0015_geo = "./input/NACA0015.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 0.6;
Real DH = 0.3;
Real resolution_ref = 0.01; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(-DL, -DH), Vec2d(DL, DH));
//----------------------------------------------------------------------
//	import model as a complex shape
//----------------------------------------------------------------------
Vec2d naca0015_translation = Vec2d(-0.48, 0.0);
class ImportModel : public MultiPolygonShape
{
  public:
    explicit ImportModel(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addAPolygonFromFile(naca0015_geo, ShapeBooleanOps::add, naca0015_translation);
    }
};

Vec2d waterblock_halfsize = Vec2d(DL, DH);
Vec2d waterblock_translation = Vec2d(0.0, 0.0);
class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addABox(Transform(waterblock_translation), waterblock_halfsize, ShapeBooleanOps::add);
        multi_polygon_.addAPolygonFromFile(naca0015_geo, ShapeBooleanOps::sub, naca0015_translation);
    }
};

//----------------------------------------------------------------------
//	Main program starts here.
//----------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up -- a SPHSystem
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, resolution_ref);
    sph_system.setRunParticleRelaxation(true); // tag to run particle relaxation when no commandline option
#ifdef BOOST_AVAILABLE
    sph_system.handleCommandlineOptions(ac, av);
#endif
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    RealBody naca0015(sph_system, makeShared<ImportModel>("NACA0015"));
    // naca0015.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    naca0015.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(io_environment);
    naca0015.defineParticlesAndMaterial();
    naca0015.generateParticles<ParticleGeneratorLattice>();
    naca0015.addBodyStateForRecording<Real>("Density");

    RealBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    water_block.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(io_environment);
    water_block.defineParticlesAndMaterial();
    water_block.generateParticles<ParticleGeneratorLattice>();
    water_block.addBodyStateForRecording<Real>("Density");
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation naca0015_inner(naca0015);
    InnerRelation water_inner(water_block);
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    SimpleDynamics<RandomizeParticlePosition> random_naca0015_particles(naca0015);
    SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
    relax_dynamics::RelaxationStepInner relaxation_step_inner(naca0015_inner, true);
    relax_dynamics::RelaxationStepInner relaxation_step_inner_water(water_inner, true);
    //----------------------------------------------------------------------
    //	Define simple file input and outputs functions.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
    //MeshRecordingToPlt cell_linked_list_recording(io_environment, naca0015.getCellLinkedList());
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    random_naca0015_particles.exec(0.25);
    random_water_particles.exec(0.25);
    relaxation_step_inner.SurfaceBounding().exec();
    relaxation_step_inner_water.SurfaceBounding().exec();
    naca0015.updateCellLinkedList();
    water_block.updateCellLinkedList();
    //----------------------------------------------------------------------
    //	First output before the simulation.
    //----------------------------------------------------------------------
    write_real_body_states.writeToFile();
    //cell_linked_list_recording.writeToFile();
    //----------------------------------------------------------------------
    //	Particle relaxation time stepping start here.
    //----------------------------------------------------------------------
    int ite_p = 0;
    while (ite_p < 2000)
    {
        relaxation_step_inner.exec();
        relaxation_step_inner_water.exec();
        ite_p += 1;
        if (ite_p % 100 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
            write_real_body_states.writeToFile(ite_p);
        }
    }
    std::cout << "The physics relaxation process finish !" << std::endl;

    return 0;
}