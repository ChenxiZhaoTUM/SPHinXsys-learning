/**
 * @file 	airfoil_2d.cpp
 * @brief 	This is the test of using levelset to generate body fitted SPH particles.
 * @details	We use this case to test the particle generation and relaxation with a complex geometry (2D).
 *			Before the particles are generated, we clean the sharp corners and other unresolvable surfaces.
 * @author 	Yongchuan Yu and Xiangyu Hu
 */
#include "sphinxsys.h"
#include "relative_error_for_consistency.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Set the file path to the data file.
//----------------------------------------------------------------------
std::string full_path_to_stl_file = "./input/starfish_3D.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real resolution_ref = 0.03; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vecd(-1.5, -1.5, -0.5), Vecd(2.5, 2.5, 1.0));
//----------------------------------------------------------------------
//	import model as a complex shape
//----------------------------------------------------------------------
class Bunny : public ComplexShape
{
  public:
    explicit Bunny(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeSTL>(full_path_to_stl_file, Vecd(0.0, 0.0, 0.0), 1.0);
    }
};

Vecd water_half_size(1.6, 1.6, 0.25);
Vecd water_transition(0.3825, 0.2972, 0.25);
class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TransformShape<GeometricShapeBox>>(Transform(water_transition), water_half_size, "OuterBoundary");
        subtract<TriangleMeshShapeSTL>(full_path_to_stl_file, Vecd(0.0, 0.0, 0.0), 1.0);
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
    RealBody starfish(sph_system, makeShared<Bunny>("StarFish"));
    starfish.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    //starfish.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(io_environment);
    starfish.defineParticlesAndMaterial();
    starfish.generateParticles<ParticleGeneratorLattice>();
    starfish.addBodyStateForRecording<Real>("Density");

    RealBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    water_block.defineComponentLevelSetShape("OuterBoundary");
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
    InnerRelation starfish_inner(starfish);
    ComplexRelation water_starfish_complex(water_block, {&starfish});
    ComplexRelation starfish_water_complex(starfish, {&water_block});
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    SimpleDynamics<RandomizeParticlePosition> random_starfish_particles(starfish);
    SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
    relax_dynamics::RelaxationStepInner relaxation_step_inner(starfish_inner, true);
    relax_dynamics::RelaxationStepComplex relaxation_step_complex(water_starfish_complex, "OuterBoundary", true);
    starfish.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");
    water_block.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");
    
    ReducedQuantityRecording<TotalKineticEnergy> write_starfish_kinetic_energy(io_environment, starfish, "Starfish_Kinetic_Energy");
    ReducedQuantityRecording<TotalKineticEnergy> write_water_kinetic_energy(io_environment, water_block, "Water_Kinetic_Energy");

    //BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
    BodyStatesRecordingToPlt write_real_body_states(io_environment, sph_system.real_bodies_);
    WriteFuncRelativeErrorSum write_function_relative_error_sum(io_environment, starfish, water_block);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    random_starfish_particles.exec(0.25);
    random_water_particles.exec(0.25);
    relaxation_step_inner.SurfaceBounding().exec();
    relaxation_step_complex.SurfaceBounding().exec();
    starfish.updateCellLinkedList();
    water_block.updateCellLinkedList();
    starfish_water_complex.updateConfiguration();
    water_starfish_complex.updateConfiguration();
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
        relaxation_step_complex.exec();

        write_starfish_kinetic_energy.writeToFile(ite_p);
        write_water_kinetic_energy.writeToFile(ite_p);

        ite_p += 1;
        if (ite_p % 100 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
            write_real_body_states.writeToFile(ite_p);
        }

        starfish_water_complex.updateConfiguration();
        water_starfish_complex.updateConfiguration();
    }
    std::cout << "The physics relaxation process finish !" << std::endl;
    write_function_relative_error_sum.writeToFile(ite_p);

    return 0;
}
