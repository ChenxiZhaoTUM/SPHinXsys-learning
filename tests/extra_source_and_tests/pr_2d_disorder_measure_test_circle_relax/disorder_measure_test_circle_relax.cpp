/**
 * @file 	airfoil_2d.cpp
 * @brief 	This is the test of using levelset to generate body fitted SPH particles.
 * @details	We use this case to test the particle generation and relaxation with a complex geometry (2D).
 *			Before the particles are generated, we clean the sharp corners and other unresolvable surfaces.
 * @author 	Yongchuan Yu and Xiangyu Hu
 */
#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "sphinxsys.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real R = 2.0;
Real resolution_ref = R / 20.0;
BoundingBox system_domain_bounds(Vec2d(-R, -R), Vec2d(R, R));

class ImportModel : public MultiPolygonShape
{
  public:
    explicit ImportModel(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        multi_polygon_.addACircle(Vecd(0.0, 0.0), R, 100, ShapeBooleanOps::add);
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
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    RealBody import_body(sph_system, makeShared<ImportModel>("Circle"));
    import_body.defineBodyLevelSetShape()->writeLevelSet(sph_system);
    import_body.generateParticles<BaseParticles, Lattice>();
    import_body.addBodyStateForRecording<Real>("Density");
    //----------------------------------------------------------------------
    //	Define body relation map.
    //	The contact map gives the topological connections between the bodies.
    //	Basically the the range of bodies to build neighbor particle lists.
    //  Generally, we first define all the inner relations, then the contact relations.
    //  At last, we define the complex relaxations by combining previous defined
    //  inner and contact relations.
    //----------------------------------------------------------------------
    InnerRelation import_inner(import_body);
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    using namespace relax_dynamics;
    SimpleDynamics<RandomizeParticlePosition> random_import_particles(import_body);
    RelaxationStepLevelSetCorrectionInner relaxation_step_inner(import_inner);

    InteractionDynamics<NablaWVLevelSetCorrectionInner> solid_zero_order_consistency(import_inner);
    import_body.addBodyStateForRecording<Vecd>("KernelSummation");

    SimpleDynamics<NormalDirectionFromBodyShape> solid_normal_direction(import_body);

    InteractionDynamics<LocalDisorderMeasure> local_disorder_measure(import_inner);
    import_body.addBodyStateForRecording<Real>("FirstDistance");
    import_body.addBodyStateForRecording<Real>("SecondDistance");
    import_body.addBodyStateForRecording<Real>("LocalDisorderMeasureParameter");
    GlobalDisorderMeasure write_global_disorder_measure(import_body);

    BodyStatesRecordingToVtp write_real_body_states(sph_system.real_bodies_);
    // BodyStatesRecordingToPlt write_real_body_states(sph_system.real_bodies_);
    //----------------------------------------------------------------------
    //	Prepare the simulation with cell linked list, configuration
    //	and case specified initial condition if necessary.
    //----------------------------------------------------------------------
    random_import_particles.exec(0.25);
    relaxation_step_inner.SurfaceBounding().exec();
    import_body.updateCellLinkedList();
    //----------------------------------------------------------------------
    //	First output before the simulation.
    //----------------------------------------------------------------------
    solid_zero_order_consistency.exec();
    solid_normal_direction.exec();
    local_disorder_measure.exec();
    write_global_disorder_measure.writeToFile();
    write_real_body_states.writeToFile();
    //----------------------------------------------------------------------
    //	Particle relaxation time stepping start here.
    //----------------------------------------------------------------------
     int ite_p = 0;
     while (ite_p < 10000)
    {
        relaxation_step_inner.exec();
    
        solid_normal_direction.exec();
        local_disorder_measure.exec();

        ite_p += 1;
        if (ite_p % 500 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
            write_real_body_states.writeToFile(ite_p);
        }

        write_global_disorder_measure.writeToFile(ite_p);
    }
     std::cout << "The physics relaxation process finish !" << std::endl;

    solid_zero_order_consistency.exec();
    write_real_body_states.writeToFile(ite_p);

    return 0;
}