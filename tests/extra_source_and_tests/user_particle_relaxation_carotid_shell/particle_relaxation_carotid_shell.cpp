/**
 * @file 	particle_relaxation_single_resolution.cpp
 * @brief 	This is the test of using levelset to generate particles with single resolution and relax particles.
 * @details We use this case to test the particle generation and relaxation for a complex geometry.
 *			Before particle generation, we clean the sharp corners of the model.
 * @author 	Yongchuan Yu and Xiangyu Hu
 */

#include "sphinxsys.h"
using namespace SPH;
//----------------------------------------------------------------------
//	Setting for the geometry.
//	To use this, please commenting the setting for the first geometry.
//----------------------------------------------------------------------
std::string full_path_to_file = "./input/carotid_fluid_geo.stl";
// std::string full_path_to_file = "./input/carotid_solid_geo.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real scaling = pow(10, -3);
Vec3d domain_lower_bound(-6.0 * scaling, -4.0 * scaling, -32.5 * scaling);
Vec3d domain_upper_bound(12.0 * scaling, 10.0 * scaling, 23.5 * scaling);
//----------------------------------------------------------------------
//	Below are common parts for the two test geometries.
//----------------------------------------------------------------------
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 0.1 * scaling;
Real thickness = 1.0 * dp_0;
Real level_set_refinement_ratio = dp_0 / (0.1 * thickness);
//----------------------------------------------------------------------
//	define the imported model.
//----------------------------------------------------------------------
class SolidBodyFromMesh : public ComplexShape
{
public:
    explicit SolidBodyFromMesh(const std::string &shape_name) : ComplexShape(shape_name),
        mesh_shape_(new TriangleMeshShapeSTL(full_path_to_file, translation, scaling))
    {
        //add<ExtrudeShape<TriangleMeshShapeSTL>>(thickness, full_path_to_file, translation, scaling);
        add<TriangleMeshShapeSTL>(full_path_to_file, translation, scaling);
    }

    TriangleMeshShapeSTL* getMeshShape() const
    {
        return mesh_shape_.get();
    }

private:
    std::unique_ptr<TriangleMeshShapeSTL> mesh_shape_;

};

class FromSTLFile;
template <>
class ParticleGenerator<FromSTLFile> : public ParticleGenerator<Surface>
{

    const Real thickness_;
    TriangleMeshShapeSTL* mesh_shape_;
    Shape &initial_shape_;


public:
    explicit ParticleGenerator(SPHBody& sph_body, TriangleMeshShapeSTL* mesh_shape) 
        : ParticleGenerator<Surface>(sph_body),
        thickness_(sph_body.sph_adaptation_->ReferenceSpacing()),
        mesh_shape_(mesh_shape), initial_shape_(sph_body.getInitialShape()) 
    {
        if (!mesh_shape_)
        {
            std::cerr << "Error: Mesh shape is not set!" << std::endl;
            return;
        }

        if (!initial_shape_.isValid())
        {
            std::cout << "\n BaseParticleGeneratorLattice Error: initial_shape_ is invalid." << std::endl;
            std::cout << __FILE__ << ':' << __LINE__ << std::endl;
            throw;
        }
    }

    virtual void initializeGeometricVariables() override
    {
        // Preload vertex positions
        std::vector<Vec3d> vertex_positions;
        int num_vertices = mesh_shape_->getTriangleMesh()->getNumVertices();
        vertex_positions.reserve(num_vertices);
        for (int i = 0; i < num_vertices; ++i)
        {
            vertex_positions.push_back(SimTKToEigen(mesh_shape_->getTriangleMesh()->getVertexPosition(i)));
        }

        // Generate particles at the center of each triangle face
        int num_faces = mesh_shape_->getTriangleMesh()->getNumFaces();
        std::cout << "num_faces calculation = " << num_faces << std::endl;

        for (int i = 0; i < num_faces; ++i)
        {
            Vec3d vertices[3];
            for (int j = 0; j < 3; ++j)
            {
                int vertexIndex = mesh_shape_->getTriangleMesh()->getFaceVertex(i, j);
                vertices[j] = vertex_positions[vertexIndex];
            }

            // Generate particle at the center of this triangle face
            generateParticleAtFaceCenter(vertices);
        }
    }

private:
    void generateParticleAtFaceCenter(const Vec3d vertices[3])
    {
        Vec3d face_center = (vertices[0] + vertices[1] + vertices[2]) / 3.0;
        Vec3d edge1 = vertices[1] - vertices[0];
        Vec3d edge2 = vertices[2] - vertices[0];
        double area = 0.5 * edge1.cross(edge2).norm();

        initializePositionAndVolumetricMeasure(face_center, area);
        initializeSurfaceProperties(initial_shape_.findNormalDirection(face_center), thickness_);
    }
};

//-----------------------------------------------------------------------------------------------------------
//	Main program starts here.
//-----------------------------------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up -- a SPHSystem
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.setRunParticleRelaxation(false); // Tag for run particle relaxation for body-fitted distribution
    sph_system.setReloadParticles(false);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.
    //----------------------------------------------------------------------
    SolidBodyFromMesh solid_body_from_mesh("SolidBodyFromMesh");
    TriangleMeshShapeSTL* mesh_shape = solid_body_from_mesh.getMeshShape();

    RealBody imported_model(sph_system, makeShared<SolidBodyFromMesh>("SolidBodyFromMesh"));
    imported_model.defineAdaptation<SPHAdaptation>(1.15, 1.0);
    //imported_model.defineBodyLevelSetShape(level_set_refinement_ratio)->correctLevelSetSign()->writeLevelSet(sph_system);
    (!sph_system.RunParticleRelaxation() && sph_system.ReloadParticles())
        ? imported_model.generateParticles<SurfaceParticles, Reload>(imported_model.getName())
        : imported_model.generateParticles<SurfaceParticles, FromSTLFile>(mesh_shape);
                       
    if (sph_system.RunParticleRelaxation())
    {
        //----------------------------------------------------------------------
        //	Define body relation map.
        //	The contact map gives the topological connections between the bodies.
        //	Basically the the range of bodies to build neighbor particle lists.
        //  Generally, we first define all the inner relations, then the contact relations.
        //  At last, we define the complex relaxations by combining previous defined
        //  inner and contact relations.
        //----------------------------------------------------------------------
        InnerRelation imported_model_inner(imported_model);
        //----------------------------------------------------------------------
        //	Methods used for particle relaxation.
        //----------------------------------------------------------------------
        using namespace relax_dynamics;
        SimpleDynamics<RandomizeParticlePosition> random_imported_model_particles(imported_model);
        /** A  Physics relaxation step. */
        ShellRelaxationStep relaxation_step_inner(imported_model_inner);
        ShellNormalDirectionPrediction shell_normal_prediction(imported_model_inner, thickness, cos(Pi / 3.75));

        /** Write the body state to Vtp file. */
        BodyStatesRecordingToVtp write_imported_model_to_vtp({imported_model});
        /** Write the particle reload files. */
        ReloadParticleIO write_particle_reload_files(imported_model);
        //----------------------------------------------------------------------
        //	Particle relaxation starts here.
        //----------------------------------------------------------------------
        random_imported_model_particles.exec(0.25);
        relaxation_step_inner.MidSurfaceBounding().exec();
        write_imported_model_to_vtp.writeToFile(0.0);
        imported_model.updateCellLinkedList();
        // write_cell_linked_list.writeToFile(0.0);
        //----------------------------------------------------------------------
        //	Particle relaxation time stepping start here.
        //----------------------------------------------------------------------
        int ite_p = 0;
        while (ite_p < 500)
        {
            relaxation_step_inner.exec();
            ite_p += 1;
            if (ite_p % 50 == 0)
            {
                std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";
                write_imported_model_to_vtp.writeToFile(ite_p);
            }
        }
        std::cout << "The physics relaxation process of imported model finish !" << std::endl;

        shell_normal_prediction.exec();
        write_imported_model_to_vtp.writeToFile(ite_p);
        write_particle_reload_files.writeToFile(0);

        return 0;
    }

    BodyStatesRecordingToVtp write_body_states(sph_system);
    write_body_states.writeToFile();
    return 0;
}
