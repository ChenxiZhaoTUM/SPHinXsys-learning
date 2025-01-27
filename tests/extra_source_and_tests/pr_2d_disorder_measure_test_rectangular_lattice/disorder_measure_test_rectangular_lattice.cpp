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
Real L = 2.0;
Real H = 2.0;
Real resolution_ref = H / 40.0;
BoundingBox system_domain_bounds(Vec2d(0.0, 0.0), Vec2d(L, H));

class ImportModel : public MultiPolygonShape
{
  public:
    explicit ImportModel(const std::string &shape_name) : MultiPolygonShape(shape_name)
    {
        std::vector<Vecd> shape;
        shape.push_back(Vecd(0.0, 0.0));
        shape.push_back(Vecd(0.0, H));
        shape.push_back(Vecd(L, H));
        shape.push_back(Vecd(L, 0.0));
        shape.push_back(Vecd(0.0, 0.0));
        multi_polygon_.addAPolygon(shape, ShapeBooleanOps::add);
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
    RealBody import_body(sph_system, makeShared<ImportModel>("Rectangular"));
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
    import_body.updateCellLinkedList();
    //----------------------------------------------------------------------
    //	First output before the simulation.
    //----------------------------------------------------------------------
    solid_zero_order_consistency.exec();
    solid_normal_direction.exec();
    //----------------------------------------------------------------------
    //	Particle relaxation time stepping start here.
    //----------------------------------------------------------------------

    local_disorder_measure.exec();
    write_global_disorder_measure.writeToFile(0);
    solid_zero_order_consistency.exec();
    write_real_body_states.writeToFile(0);

    return 0;
}