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
std::string airfoil_flap_front = "./input/airfoil_flap_front.dat";
std::string airfoil_wing = "./input/airfoil_wing.dat";
std::string airfoil_flap_rear = "./input/airfoil_flap_rear.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 1.25;             /**< airfoil length rear part. */
Real DL1 = 0.25;            /**< airfoil length front part. */
Real DH = 0.25;             /**< airfoil height. */
Real resolution_ref = 0.005; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(-DL1, -DH), Vec2d(DL, DH));
//----------------------------------------------------------------------
//	import model as a complex shape
//----------------------------------------------------------------------
class ImportModel : public MultiPolygonShape
{
  public:
    explicit ImportModel(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addAPolygonFromFile(airfoil_flap_front, ShapeBooleanOps::add);
        multi_polygon_.addAPolygonFromFile(airfoil_wing, ShapeBooleanOps::add);
        multi_polygon_.addAPolygonFromFile(airfoil_flap_rear, ShapeBooleanOps::add);
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
    RealBody airfoil(sph_system, makeShared<ImportModel>("AirFoil"));
    //airfoil.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    airfoil.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(io_environment);
    airfoil.defineParticlesAndMaterial();
    airfoil.generateParticles<ParticleGeneratorLattice>();
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation airfoil_inner(airfoil);
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    SimpleDynamics<RandomizeParticlePosition> random_airfoil_particles(airfoil);
    relax_dynamics::RelaxationStepInner relaxation_step_inner(airfoil_inner, true);
    //----------------------------------------------------------------------
    //	Define simple file input and outputs functions.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp input_body_recording_to_vtp(io_environment, airfoil);
    MeshRecordingToPlt cell_linked_list_recording(io_environment, airfoil.getCellLinkedList());
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    random_airfoil_particles.exec(0.25);
    relaxation_step_inner.SurfaceBounding().exec();
    airfoil.updateCellLinkedList();
    //----------------------------------------------------------------------
    //	First output before the simulation.
    //----------------------------------------------------------------------
    input_body_recording_to_vtp.writeToFile();
    cell_linked_list_recording.writeToFile();
    //----------------------------------------------------------------------
    //	Particle relaxation time stepping start here.
    //----------------------------------------------------------------------
    int ite_p = 0;
    while (ite_p < 2000)
    {
        relaxation_step_inner.exec();
        ite_p += 1;
        if (ite_p % 100 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
            input_body_recording_to_vtp.writeToFile(ite_p);
        }
    }
    std::cout << "The physics relaxation process finish !" << std::endl;

    return 0;
}