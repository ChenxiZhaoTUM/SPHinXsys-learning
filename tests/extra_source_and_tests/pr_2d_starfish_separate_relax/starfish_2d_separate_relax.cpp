/**
 * @file 	airfoil_2d.cpp
 * @brief 	This is the test of using levelset to generate body fitted SPH particles.
 * @details	We use this case to test the particle generation and relaxation with a complex geometry (2D).
 *			Before the particles are generated, we clean the sharp corners and other unresolvable surfaces.
 * @author 	Yongchuan Yu and Xiangyu Hu
 */
#include "sphinxsys.h"
#include "kernel_summation.h"
#include "kernel_summation.hpp"

using namespace SPH;
//----------------------------------------------------------------------
//	Set the file path to the data file.
//----------------------------------------------------------------------
std::string import_geo = "./input/starfish_sorted.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 1.5;
Real DH = 1.5;
Real resolution_ref = 0.075; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(-DL, -DH), Vec2d(DL, DH));
//----------------------------------------------------------------------
//	import model as a complex shape
//----------------------------------------------------------------------
Vec2d starfish_translation = Vec2d(-0.34, -0.3);
class ImportModel : public MultiPolygonShape
{
  public:
    explicit ImportModel(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addAPolygonFromFile(import_geo, ShapeBooleanOps::add, starfish_translation);
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

class WaterOuter : public ComplexShape
{
  public:
    explicit WaterOuter(const std::string &shape_name) : ComplexShape(shape_name)
    {
        MultiPolygon outer_boundary(createWaterBlockShape());
        add<MultiPolygonShape>(outer_boundary, "OuterBoundaryShape");
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
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    BOOLEAN complex_relaxation(false);
    BOOLEAN geometry_boolean_operation(false);
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    
    RealBody import_body(sph_system, makeShared<ImportModel>("StarFish"));
    import_body.defineBodyLevelSetShape()->writeLevelSet(sph_system);
    import_body.generateParticles<BaseParticles, Lattice>();
    import_body.addBodyStateForRecording<Real>("Density");

    InverseShape<ImportModel> inversed_import("InversedStarfish");
    LevelSetShape inversed_import_level_set(inversed_import, makeShared<SPHAdaptation>(resolution_ref));
    inversed_import_level_set.cleanLevelSet();
    WaterOuter water_shape("WaterShape");
    water_shape.initializeComponentLevelSetShapesByAdaptation(makeShared<SPHAdaptation>(resolution_ref), sph_system);
    water_shape.addAnLevelSetShape(&inversed_import_level_set);
   /* for (size_t i = 0; i != water_shape.getLevelSetShapes().size(); i++)
    {
        water_shape.getLevelSetShapes()[i]->writeLevelSet(sph_system);
    };*/

    RealBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    water_block.defineComponentLevelSetShape("OuterBoundary");
    water_block.generateParticles<BaseParticles, Lattice>();
    water_block.addBodyStateForRecording<Real>("Density");
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation import_inner(import_body);
    InnerRelation water_inner(water_block);
    ContactRelation import_water_contact(import_body, {&water_block});
    ContactRelation water_import_contact(water_block, {&import_body});
    ComplexRelation import_water_complex(import_inner, import_water_contact);
    ComplexRelation water_import_complex(water_inner, water_import_contact);
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    using namespace relax_dynamics;
    SimpleDynamics<RandomizeParticlePosition> random_import_particles(import_body);
    SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
    RelaxationStepLevelSetCorrectionInner relaxation_step_inner(import_inner);
    RelaxationStepWithComplexBounding relaxation_step_inner_water(water_inner, water_shape);

    InteractionDynamics<NablaWVLevelSetCorrectionInner> solid_zero_order_consistency(import_inner);
    InteractionDynamics<NablaWVLevelSetCorrectionComplex> fluid_zero_order_consistency(ConstructorArgs(water_inner, std::string("OuterBoundary")), water_import_contact);
    //InteractionDynamics<NablaWVComplex> fluid_zero_order_consistency(water_inner, water_import_contact);
    import_body.addBodyStateForRecording<Vecd>("KernelSummation");
    water_block.addBodyStateForRecording<Vecd>("KernelSummation");

    SimpleDynamics<NormalDirectionFromBodyShape> solid_normal_direction(import_body);
    InteractionWithUpdate<FluidSurfaceIndicationByDistance> fluid_surface_indicator(water_import_contact);
    water_block.addBodyStateForRecording<int>("Indicator");
    ReducedQuantityRecording<SurfaceKineticEnergy> write_water_kinetic_energy(water_block);
    water_block.addBodyStateForRecording<Real>("ParticleEnergy");

    BodyStatesRecordingToVtp write_real_body_states(sph_system.real_bodies_);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    random_import_particles.exec(0.25);
    random_water_particles.exec(0.25);
    relaxation_step_inner.SurfaceBounding().exec();
    relaxation_step_inner_water.SurfaceBounding().exec();
    import_body.updateCellLinkedList();
    water_block.updateCellLinkedList();
    import_water_complex.updateConfiguration();
    water_import_complex.updateConfiguration();
    //----------------------------------------------------------------------
    //	First output before the simulation.
    //----------------------------------------------------------------------
    write_real_body_states.writeToFile();
    //----------------------------------------------------------------------
    //	Particle relaxation time stepping start here.
    //----------------------------------------------------------------------
    int ite_p = 0;
    while (ite_p < 2000)
    {
        relaxation_step_inner.exec();
        relaxation_step_inner_water.exec();

        solid_normal_direction.exec();
        fluid_surface_indicator.exec();
        write_water_kinetic_energy.writeToFile(ite_p);

        ite_p += 1;
        if (ite_p % 100 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
            write_real_body_states.writeToFile(ite_p);
        }

        import_water_complex.updateConfiguration();
        water_import_complex.updateConfiguration();
    }
    std::cout << "The physics relaxation process finish !" << std::endl;

    solid_zero_order_consistency.exec();
    fluid_zero_order_consistency.exec();
    write_real_body_states.writeToFile(ite_p);

    return 0;
}
