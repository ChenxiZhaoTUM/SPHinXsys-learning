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
std::string zigzag_geo = "./input/zigzag_0.75.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 0.75;
Real DH = 0.75;
Real resolution_ref = 0.05; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(-DL, -DH), Vec2d(DL, DH));
//----------------------------------------------------------------------
//	import model as a complex shape
//----------------------------------------------------------------------
class ImportModel : public MultiPolygonShape
{
  public:
    explicit ImportModel(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addAPolygonFromFile(zigzag_geo, ShapeBooleanOps::add);
    }
};

std::vector<Vecd> createWaterBlockShape()
{
    std::vector<Vecd> water_block_shape;
    water_block_shape.push_back(Vecd(-DL, -DH));
    water_block_shape.push_back(Vecd(-DL, DH));
    water_block_shape.push_back(Vecd(DL, DH));
    water_block_shape.push_back(Vecd(DL, -DH));
    water_block_shape.push_back(Vecd(-DL, -DH));

    return water_block_shape;
}

class WaterBlock : public ComplexShape
{
  public:
    explicit WaterBlock(const std::string &shape_name) : ComplexShape(shape_name)
    {
        MultiPolygon outer_boundary(createWaterBlockShape());
        add<MultiPolygonShape>(outer_boundary, "OuterBoundary");
        ImportModel import_model("InnerBody");
        subtract<ImportModel>(import_model);
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
    RealBody zigzag(sph_system, makeShared<ImportModel>("ZigZag"));
    zigzag.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    //zigzag.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(io_environment);
    zigzag.defineParticlesAndMaterial();
    zigzag.generateParticles<ParticleGeneratorReload>(io_environment, zigzag.getName());
    zigzag.addBodyStateForRecording<Real>("Density");

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
    InnerRelation zigzag_inner(zigzag);
    InnerRelation water_inner(water_block);
    ComplexRelation zigzag_water_complex(zigzag, {&water_block});
    ComplexRelation water_zigzag_complex(water_block, {&zigzag});
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    InteractionDynamics<ZeroOrderConsistencyInteraction> zero_order_consistency_solid(zigzag_inner);
    InteractionDynamics<ZeroOrderConsistencyInteractionComplex> zero_order_consistency_fluid(water_zigzag_complex, "OuterBoundary");
    zigzag.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");
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
