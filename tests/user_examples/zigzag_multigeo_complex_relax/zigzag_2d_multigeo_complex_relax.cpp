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
    sph_system.setRunParticleRelaxation(true); // tag to run particle relaxation when no commandline option
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
    zigzag.generateParticles<ParticleGeneratorLattice>();
    zigzag.addBodyStateForRecording<Real>("Density");

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
    InnerRelation zigzag_inner(zigzag);
    ComplexRelation zigzag_water_complex(zigzag, {&water_block});
    ComplexRelation water_zigzag_complex(water_block, {&zigzag});
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    SimpleDynamics<RandomizeParticlePosition> random_zigzag_particles(zigzag);
    SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
    relax_dynamics::RelaxationStepInner relaxation_step_inner(zigzag_inner, true);
    relax_dynamics::RelaxationStepComplex relaxation_step_complex(water_zigzag_complex, "OuterBoundary", true);
    zigzag.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");
    water_block.addBodyStateForRecording<Vecd>("ZeroOrderConsistencyValue");

    ReducedQuantityRecording<TotalKineticEnergy> write_zigzag_kinetic_energy(io_environment, zigzag, "ZigZag_Kinetic_Energy");
    ReducedQuantityRecording<TotalKineticEnergy> write_water_kinetic_energy(io_environment, water_block, "Water_Kinetic_Energy");

    //BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
    BodyStatesRecordingToPlt write_real_body_states(io_environment, sph_system.real_bodies_);
    WriteFuncRelativeErrorSum write_function_relative_error_sum(io_environment, zigzag, water_block);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    random_zigzag_particles.exec(0.25);
    random_water_particles.exec(0.25);
    relaxation_step_inner.SurfaceBounding().exec();
    relaxation_step_complex.SurfaceBounding().exec();
    zigzag.updateCellLinkedList();
    water_block.updateCellLinkedList();
    zigzag_water_complex.updateConfiguration();
    water_zigzag_complex.updateConfiguration();
    //----------------------------------------------------------------------
    //	First output before the simulation.
    //----------------------------------------------------------------------
    int ite_p = 0;
    write_real_body_states.writeToFile();
    write_zigzag_kinetic_energy.writeToFile(ite_p);
    write_water_kinetic_energy.writeToFile(ite_p);
    //cell_linked_list_recording.writeToFile();
    //----------------------------------------------------------------------
    //	Particle relaxation time stepping start here.
    //----------------------------------------------------------------------
    while (ite_p < 4000)
    {
        relaxation_step_inner.exec();
        relaxation_step_complex.exec();

        ite_p += 1;

        write_zigzag_kinetic_energy.writeToFile(ite_p);
        write_water_kinetic_energy.writeToFile(ite_p);
        
        if (ite_p % 100 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
            write_real_body_states.writeToFile(ite_p);
        }

        zigzag_water_complex.updateConfiguration();
        water_zigzag_complex.updateConfiguration();
    }
    std::cout << "The physics relaxation process finish !" << std::endl;
    write_function_relative_error_sum.writeToFile(ite_p);
    
    return 0;
}