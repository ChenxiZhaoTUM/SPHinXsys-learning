/**
 * @file 	airfoil_2d.cpp
 * @brief 	This is the test of using levelset to generate body fitted SPH particles.
 * @details	We use this case to test the particle generation and relaxation with a complex geometry (2D).
 *			Before the particles are generated, we clean the sharp corners and other unresolvable surfaces.
 * @author 	Yongchuan Yu and Xiangyu Hu
 */
#include "relative_error_for_consistency.h"
#include "sphinxsys.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Set the file path to the data file.
//----------------------------------------------------------------------
std::string airfoil = "./input/NACA5515_5deg.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real L = 1.0;
Real DL = 5 * L;
Real DL1 = 2 * L;
Real DH = 3 * L;
Real resolution_ref = 0.004; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(-DL1, -DH), Vec2d(DL, DH));
//----------------------------------------------------------------------
//	import model as a complex shape
//----------------------------------------------------------------------
class ImportModel : public MultiPolygonShape
{
  public:
    explicit ImportModel(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addAPolygonFromFile(airfoil, ShapeBooleanOps::add);
    }
};

Vec2d waterblock_halfsize = Vec2d(0.5 * (DL + DL1), DH);
Vec2d waterblock_translation = Vec2d(0.5, 0.0);

class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addABox(Transform(waterblock_translation), waterblock_halfsize, ShapeBooleanOps::add);
        multi_polygon_.addAPolygonFromFile(airfoil, ShapeBooleanOps::sub);
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
    RealBody airfoil(sph_system, makeShared<ImportModel>("Airfoil"));
    // airfoil.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    airfoil.defineBodyLevelSetShape()->cleanLevelSet(1.0)->writeLevelSet(io_environment);
    airfoil.defineParticlesAndMaterial();
    airfoil.generateParticles<ParticleGeneratorLattice>();
    airfoil.addBodyStateForRecording<Real>("Density");

    RealBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    //water_block.defineBodyLevelSetShape()->cleanLevelSet(1.0)->writeLevelSet(io_environment);
    water_block.defineBodyLevelSetShape()->writeLevelSet(io_environment);
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
    InnerRelation airfoil_inner(airfoil);
    InnerRelation water_inner(water_block);
    ComplexRelation airfoil_water_complex(airfoil, {&water_block});
    ComplexRelation water_airfoil_complex(water_block, {&airfoil});
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    SimpleDynamics<RandomizeParticlePosition> random_airfoil_particles(airfoil);
    SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
    relax_dynamics::RelaxationStepInner relaxation_step_inner(airfoil_inner, true);
    relax_dynamics::RelaxationStepInner relaxation_step_inner_water(water_inner, true);

    ReducedQuantityRecording<TotalKineticEnergy> write_airfoil_kinetic_energy(io_environment, airfoil, "Airfoil_Kinetic_Energy");
    ReducedQuantityRecording<TotalKineticEnergy> write_water_kinetic_energy(io_environment, water_block, "Water_Kinetic_Energy");

    // BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
    BodyStatesRecordingToPlt write_real_body_states(io_environment, sph_system.real_bodies_);
    ReloadParticleIO write_real_body_particle_reload_files(io_environment, sph_system.real_bodies_);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    random_airfoil_particles.exec(0.25);
    random_water_particles.exec(0.25);
    relaxation_step_inner.SurfaceBounding().exec();
    relaxation_step_inner_water.SurfaceBounding().exec();
    airfoil.updateCellLinkedList();
    water_block.updateCellLinkedList();
    airfoil_water_complex.updateConfiguration();
    water_airfoil_complex.updateConfiguration();
    //----------------------------------------------------------------------
    //	First output before the simulation.
    //----------------------------------------------------------------------
    write_real_body_states.writeToFile();
    //----------------------------------------------------------------------
    //	Particle relaxation time stepping start here.
    //----------------------------------------------------------------------
    int ite_p = 0;
    while (ite_p < 1000)
    {
        relaxation_step_inner.exec();
        relaxation_step_inner_water.exec();

        write_airfoil_kinetic_energy.writeToFile(ite_p);
        write_water_kinetic_energy.writeToFile(ite_p);

        ite_p += 1;
        if (ite_p % 200 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
            write_real_body_states.writeToFile(ite_p);
        }

        airfoil_water_complex.updateConfiguration();
        water_airfoil_complex.updateConfiguration();
    }
    std::cout << "The physics relaxation process finish !" << std::endl;

    write_real_body_particle_reload_files.writeToFile(0);
    return 0;
}
