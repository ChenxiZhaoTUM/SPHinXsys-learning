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
std::string zigzag_geo = "./input/zigzag_modify.dat";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Real DL = 1.5;
Real DH = 1.5;
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

class RelativeError : public LocalDynamics, public GeneralDataDelegateInner, public GeneralDataDelegateContactOnly
{
public:
    RelativeError(ComplexRelation& complex_relation) :
        LocalDynamics(complex_relation.getSPHBody()), 
        GeneralDataDelegateInner(complex_relation.getInnerRelation()), 
        GeneralDataDelegateContactOnly(complex_relation.getContactRelation()),
        pos_(particles_->pos_), mass_(particles_->mass_), rho_(particles_->rho_)
    {
        particles_->registerVariable(error_, "RelativeErrorForConsistency");

        for (size_t k = 0; k != contact_particles_.size(); ++k)
        {
            contact_mass_.push_back(&(contact_particles_[k]->mass_));
        }
    }

    void interaction(size_t index_i, Real dt = 0.0)
    {
        Real f_analytical = sin(pos_[index_i][0] * pos_[index_i][0] + pos_[index_i][1] * pos_[index_i][1]);
        Real f_sph = 0;
        const Neighborhood &inner_neighborhood = inner_configuration_[index_i];
        for (size_t n = 0; n != inner_neighborhood.current_size_; ++n)
        {
            size_t index_j = inner_neighborhood.j_[n];
            Real f_sph_j = sin(pos_[index_j][0] * pos_[index_j][0] + pos_[index_j][1] * pos_[index_j][1]);
            f_sph += f_sph_j * inner_neighborhood.W_ij_[n] * mass_[index_j] / rho_[index_j];
        }


        for (size_t k = 0; k < this->contact_configuration_.size(); ++k)
        {
            StdLargeVec<Real> &contact_mass_k = *(contact_mass_[k]);
            Neighborhood &contact_neighborhood = (*contact_configuration_[k])[index_i];
            for (size_t n = 0; n != contact_neighborhood.current_size_; ++n)
            {
                size_t index_j = contact_neighborhood.j_[n];
                Real f_sph_j = sin(pos_[index_j][0] * pos_[index_j][0] + pos_[index_j][1] * pos_[index_j][1]);
                f_sph += f_sph_j * contact_neighborhood.W_ij_[n] * mass_[index_j] / rho_[index_j];
            }
        }

        error_[index_i] = abs(f_sph - f_analytical) * abs(f_sph - f_analytical) / abs(f_analytical) / abs(f_analytical) * mass_[index_i] / rho_[index_i];
    }

protected:
    StdLargeVec<Real> error_;
    StdLargeVec<Vecd>& pos_;
    StdLargeVec<Real>& mass_, &rho_;
    StdVec<StdLargeVec<Real> *> contact_mass_, contact_rho_;
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
    // zigzag.defineBodyLevelSetShape()->writeLevelSet(io_environment);
    zigzag.defineBodyLevelSetShape()->cleanLevelSet()->writeLevelSet(io_environment);
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

    InteractionDynamics<RelativeError> relative_error_for_zigzag(zigzag_water_complex);
    InteractionDynamics<RelativeError> relative_error_for_water(water_zigzag_complex);
    ReducedQuantityRecording<QuantitySummation<Real>> compute_relative_error_(io_environment, zigzag, "RelativeErrorForConsistency");
    //----------------------------------------------------------------------
    //	Define simple file input and outputs functions.
    //----------------------------------------------------------------------
    BodyStatesRecordingToVtp write_real_body_states(io_environment, sph_system.real_bodies_);
    //MeshRecordingToPlt cell_linked_list_recording(io_environment, zigzag.getCellLinkedList());
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
        ite_p += 1;
        if (ite_p % 100 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps N = " << ite_p << "\n";
            write_real_body_states.writeToFile(ite_p);

            relative_error_for_zigzag.exec();
            relative_error_for_water.exec();
            compute_relative_error_.writeToFile(ite_p);
        }
    }
    std::cout << "The physics relaxation process finish !" << std::endl;
    
    return 0;
}
