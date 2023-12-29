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
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 1.0;
Real DH = 0.5;
Real resolution_ref = 0.02; /**< Reference resolution. */
BoundingBox system_domain_bounds(Vec2d(0.0, 0.0), Vec2d(DL, 2*DH));
//----------------------------------------------------------------------
//	import model as a complex shape
//----------------------------------------------------------------------
Vec2d waterblock_halfsize = Vec2d(DL/2, DH/2);
Vec2d waterblock_translation = Vec2d(DL/2, DH/2);
class WaterBlock : public MultiPolygonShape
{
  public:
    explicit WaterBlock(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addABox(Transform(waterblock_translation), waterblock_halfsize, ShapeBooleanOps::add);
    }
};
Vec2d airblock_translation = Vec2d(DL/2, 3*DH/2);
class AirBlock : public MultiPolygonShape
{
  public:
    explicit AirBlock(const std::string &import_model_name) : MultiPolygonShape(import_model_name)
    {
        multi_polygon_.addABox(Transform(airblock_translation), waterblock_halfsize, ShapeBooleanOps::add);
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
    RealBody air_block(sph_system, makeShared<AirBlock>("AirBlock"));
    air_block.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(io_environment);
    air_block.defineParticlesAndMaterial();
    air_block.generateParticles<ParticleGeneratorLattice>();

    RealBody water_block(sph_system, makeShared<WaterBlock>("WaterBlock"));
    water_block.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(io_environment);
    water_block.defineParticlesAndMaterial();
    water_block.generateParticles<ParticleGeneratorLattice>();
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation air_inner(air_block);
    InnerRelation water_inner(water_block);
    ComplexRelation air_water_complex(air_block, {&water_block});
    ComplexRelation water_air_complex(water_block, {&air_block});
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    SimpleDynamics<RandomizeParticlePosition> random_air_particles(air_block);
    SimpleDynamics<RandomizeParticlePosition> random_water_particles(water_block);
    relax_dynamics::RelaxationStepInner relaxation_step_inner_air(air_inner);
    relax_dynamics::RelaxationStepInner relaxation_step_inner_water(water_inner);
    //----------------------------------------------------------------------
    //	Define simple file input and outputs functions.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
    
    ReducedQuantityRecording<TotalKineticEnergy> write_air_kinetic_energy(io_environment, air_block, "Air_Kinetic_Energy");
    ReducedQuantityRecording<TotalKineticEnergy> write_water_kinetic_energy(io_environment, water_block, "Water_Kinetic_Energy");

    InteractionDynamics<ZeroOrderConsistency> air_0order_consistency_value(air_water_complex);
    InteractionDynamics<ZeroOrderConsistency> water_0order_consistency_value(water_air_complex);
    air_block.addBodyStateForRecording<Real>("ZeroOrderConsistencyValue");
    water_block.addBodyStateForRecording<Real>("ZeroOrderConsistencyValue");

    WriteFuncRelativeErrorSum write_relative_error_sum_for_consistency(io_environment, air_block, water_block);

    //InteractionDynamics<ZeroOrderConsistencyInner> zigzag_0order_consistency_value(zigzag_inner);
    //zigzag.addBodyStateForRecording<Real>("ZeroOrderConsistencyValueInner");

    //MeshRecordingToPlt cell_linked_list_recording(io_environment, zigzag.getCellLinkedList());
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    random_air_particles.exec(0.25);
    random_water_particles.exec(0.25);
    relaxation_step_inner_air.SurfaceBounding().exec();
    relaxation_step_inner_water.SurfaceBounding().exec();
    air_block.updateCellLinkedList();
    water_block.updateCellLinkedList();
    air_water_complex.updateConfiguration();
    water_air_complex.updateConfiguration();
    //----------------------------------------------------------------------
    //	First output before the simulation.
    //----------------------------------------------------------------------
    write_real_body_states.writeToFile();
    //cell_linked_list_recording.writeToFile();
    //----------------------------------------------------------------------
    //	Particle relaxation time stepping start here.
    //----------------------------------------------------------------------
    int ite_p = 0;
    while (ite_p < 10000)
    {
        relaxation_step_inner_air.exec();
        relaxation_step_inner_water.exec();
        
        air_0order_consistency_value.exec();
        water_0order_consistency_value.exec();
       
        write_air_kinetic_energy.writeToFile(ite_p);
        write_water_kinetic_energy.writeToFile(ite_p);

        ite_p += 1;
        if (ite_p % 100 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
            write_real_body_states.writeToFile(ite_p);
        }

        air_water_complex.updateConfiguration();
        water_air_complex.updateConfiguration();
    }
    std::cout << "The physics relaxation process finish !" << std::endl;
    write_relative_error_sum_for_consistency.writeToFile(ite_p);

    return 0;
}
