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
std::string full_path_to_stl_file = "./input/bun_zipper_res2.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DX = 0.06;
Real DY = 0.05;
Real DZ = 0.1;
Real resolution_ref = 0.002; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vecd(-DX, -DY, 0), Vecd(DX, DY, DZ));
//----------------------------------------------------------------------
//	import model as a complex shape
//----------------------------------------------------------------------
class Bunny : public ComplexShape
{
  public:
    explicit Bunny(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TriangleMeshShapeSTL>(full_path_to_stl_file, Vecd::Zero(), 1.0);
    }
};

Vecd water_half_size(DX, DY, DZ / 2);
Vecd water_transition(0.0, 0.0, DZ / 2);
class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        add<TransformShape<GeometricShapeBox>>(Transform(water_transition), water_half_size, "OuterBoundary");
        subtract<TriangleMeshShapeSTL>(full_path_to_stl_file, Vecd::Zero(), 1.0);
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
    sph_system.setRunParticleRelaxation(false); // tag to run particle relaxation when no commandline option
#ifdef BOOST_AVAILABLE
    sph_system.handleCommandlineOptions(ac, av);
#endif
    IOEnvironment io_environment(sph_system);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    RealBody starfish(sph_system, makeShared<Bunny>("Bunny"));
    starfish.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    //starfish.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(io_environment);
    starfish.defineParticlesAndMaterial();
    starfish.generateParticles<ParticleGeneratorReload>(io_environment, starfish.getName());
    starfish.addBodyStateForRecording<Real>("Density");

    RealBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    water_block.defineComponentLevelSetShape("OuterBoundary");
    //water_block.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(io_environment);
    water_block.defineParticlesAndMaterial();
    water_block.generateParticles<ParticleGeneratorReload>(io_environment, water_block.getName());
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
    InnerRelation water_inner(water_block);
    ComplexRelation water_starfish_complex(water_block, {&starfish});
    ComplexRelation starfish_water_complex(starfish, {&water_block});
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    InteractionDynamics<ZeroOrderConsistencyInteraction> zero_order_consistency_solid(starfish_inner);
    InteractionDynamics<ZeroOrderConsistencyInteractionComplex> zero_order_consistency_fluid(water_starfish_complex, "OuterBoundary");
    starfish.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");
    water_block.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");

    // BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
    BodyStatesRecordingToPlt write_real_body_states(io_environment, sph_system.real_bodies_);
    //----------------------------------------------------------------------
    //	Particle relaxation time stepping start here.
    //----------------------------------------------------------------------
    sph_system.initializeSystemCellLinkedLists();
    sph_system.initializeSystemConfigurations();
    zero_order_consistency_solid.exec();
    zero_order_consistency_fluid.exec();
    write_real_body_states.writeToFile();

    return 0;
}
