/**
 * @file 	shell_particle_generation_illustration.cpp
 * @brief
 */

#include "kernel_summation.h"
#include "kernel_summation.hpp"
#include "particle_generation_and_detection.h"
#include "sphinxsys.h"

using namespace SPH;
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
std::string full_path_to_file = "./input/fluid_cylinder_12.stl";
//----------------------------------------------------------------------
//	Basic geometry parameters and numerical setup.
//----------------------------------------------------------------------
Vec3d translation(0.0, 0.0, 0.0);
Real length_scale = 1.0;
Vec3d domain_lower_bound(-1.0 * length_scale, -2.5 * length_scale, -2.5 * length_scale);
Vec3d domain_upper_bound(13.0 * length_scale, 2.5 * length_scale, 2.5 * length_scale);
BoundingBox system_domain_bounds(domain_lower_bound, domain_upper_bound);
Real dp_0 = 3.0 / 30.0;
Vecd buffer_half = Vecd(1.0 * dp_0, 2.5 * length_scale, 2.5 * length_scale);
//----------------------------------------------------------------------
//	Define case dependent body shapes.
//----------------------------------------------------------------------
class ShellShape : public ComplexShape
{
  public:
    explicit ShellShape(const std::string &shape_name) : ComplexShape(shape_name),
                                                         mesh_shape_(new TriangleMeshShapeSTL(full_path_to_file, translation, length_scale))
    {
        add<TriangleMeshShapeSTL>(full_path_to_file, translation, length_scale);
    }

    TriangleMeshShapeSTL *getMeshShape() const
    {
        return mesh_shape_.get();
    }

  private:
    std::unique_ptr<TriangleMeshShapeSTL> mesh_shape_;
};

//----------------------------------------------------------------------
//	Shell particle generation.
//----------------------------------------------------------------------
class FromSTLFile;
template <>
class ParticleGenerator<SurfaceParticles, FromSTLFile> : public ParticleGenerator<SurfaceParticles>
{
    Real mesh_total_area_;
    Real particle_spacing_;
    const Real thickness_;
    Real avg_particle_volume_;
    size_t planned_number_of_particles_;

    TriangleMeshShapeSTL *mesh_shape_;
    Shape &initial_shape_;

  public:
    explicit ParticleGenerator(SPHBody &sph_body, SurfaceParticles &surface_particles, TriangleMeshShapeSTL *mesh_shape)
        : ParticleGenerator<SurfaceParticles>(sph_body, surface_particles),
          mesh_total_area_(0),
          particle_spacing_(sph_body.sph_adaptation_->ReferenceSpacing()),
          thickness_(particle_spacing_),
          avg_particle_volume_(pow(particle_spacing_, Dimensions - 1) * thickness_),
          planned_number_of_particles_(0),
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

    virtual void prepareGeometricData() override
    {

        // Preload vertex positions
        std::vector<std::array<Real, 3>> vertex_positions;
        int num_vertices = mesh_shape_->getTriangleMesh()->getNumVertices();
        vertex_positions.reserve(num_vertices);
        for (int i = 0; i < num_vertices; i++)
        {
            const auto &p = mesh_shape_->getTriangleMesh()->getVertexPosition(i);
            vertex_positions.push_back({Real(p[0]), Real(p[1]), Real(p[2])});
        }

        // Preload face
        std::vector<std::array<int, 3>> faces;
        int num_faces = mesh_shape_->getTriangleMesh()->getNumFaces();
        std::cout << "num_faces calculation = " << num_faces << std::endl;
        faces.reserve(num_faces);
        for (int i = 0; i < num_faces; i++)
        {
            auto f1 = mesh_shape_->getTriangleMesh()->getFaceVertex(i, 0);
            auto f2 = mesh_shape_->getTriangleMesh()->getFaceVertex(i, 1);
            auto f3 = mesh_shape_->getTriangleMesh()->getFaceVertex(i, 2);
            faces.push_back({f1, f2, f3});
        }

        // Calculate total volume
        std::vector<Real> face_areas(num_faces);
        for (int i = 0; i < num_faces; ++i)
        {
            Vec3d vertices[3];
            for (int j = 0; j < 3; ++j)
            {
                const auto &pos = vertex_positions[faces[i][j]];
                vertices[j] = Vec3d(pos[0], pos[1], pos[2]);
            }

            Real each_area = calculateEachFaceArea(vertices);
            face_areas[i] = each_area;
            mesh_total_area_ += each_area;
        }

        Real number_of_particles = mesh_total_area_ * thickness_ / avg_particle_volume_ + 0.5;
        planned_number_of_particles_ = int(number_of_particles);
        std::cout << "planned_number_of_particles calculation = " << planned_number_of_particles_ << std::endl;

        // initialize a uniform distribution between 0 (inclusive) and 1 (exclusive)
        std::mt19937_64 rng;
        std::uniform_real_distribution<Real> unif(0, 1);

        // Calculate the interval based on the number of particles.
        Real interval = planned_number_of_particles_ / (num_faces + TinyReal); // if planned_number_of_particles_ >= num_faces, every face will generate particles
        if (interval <= 0)
            interval = 1; // It has to be lager than 0.

        for (int i = 0; i < num_faces; ++i)
        {
            Vec3d vertices[3];
            for (int j = 0; j < 3; ++j)
            {
                const auto &pos = vertex_positions[faces[i][j]];
                vertices[j] = Vec3d(pos[0], pos[1], pos[2]);
            }

            Real random_real = unif(rng);
            if (random_real <= interval && base_particles_.TotalRealParticles() < planned_number_of_particles_)
            {
                // Generate particle at the center of this triangle face
                // generateParticleAtFaceCenter(vertices);

                // Generate particles on this triangle face, unequal
                int particles_per_face = std::max(1, int(planned_number_of_particles_ * (face_areas[i] / mesh_total_area_)));
                generateParticlesOnFace(vertices, particles_per_face);
            }
        }

        std::cout << "Shell particle generation finish!" << std::endl;
    }

  private:
    Real calculateEachFaceArea(const Vec3d vertices[3])
    {
        Vec3d edge1 = vertices[1] - vertices[0];
        Vec3d edge2 = vertices[2] - vertices[0];
        Real area = 0.5 * edge1.cross(edge2).norm();
        return area;
    }

    void generateParticleAtFaceCenter(const Vec3d vertices[3])
    {
        Vec3d face_center = (vertices[0] + vertices[1] + vertices[2]) / 3.0;

        addPositionAndVolumetricMeasure(face_center, avg_particle_volume_ / thickness_);
        addSurfaceProperties(initial_shape_.findNormalDirection(face_center), thickness_);
    }

    void generateParticlesOnFace(const Vec3d vertices[3], int particles_per_face)
    {
        for (int k = 0; k < particles_per_face; ++k)
        {
            Real u = static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX);
            Real v = static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX);

            if (u + v > 1.0)
            {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            Vec3d particle_position = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2];

            addPositionAndVolumetricMeasure(particle_position, avg_particle_volume_ / thickness_);
            addSurfaceProperties(initial_shape_.findNormalDirection(particle_position), thickness_);
        }
    }
};
//-----------------------------------------------------------------------------------------------------------
//	Main program starts here.
//-----------------------------------------------------------------------------------------------------------
int main(int ac, char *av[])
{
    //----------------------------------------------------------------------
    //	Build up the environment of a SPHSystem with global controls.
    //----------------------------------------------------------------------
    SPHSystem sph_system(system_domain_bounds, dp_0);
    sph_system.handleCommandlineOptions(ac, av)->setIOEnvironment();
    //----------------------------------------------------------------------
    //	Creating body, materials and particles.cd
    //----------------------------------------------------------------------

    ShellShape body_from_mesh("BodyFromMesh");
    TriangleMeshShapeSTL *mesh_shape = body_from_mesh.getMeshShape();
    SolidBody shell_body(sph_system, makeShared<ShellShape>("ShellBody"));
    shell_body.defineBodyLevelSetShape(2.0)->correctLevelSetSign()->writeLevelSet(sph_system);
    shell_body.defineMaterial<Solid>();
    shell_body.generateParticles<SurfaceParticles, FromSTLFile>(mesh_shape);
    //----------------------------------------------------------------------
    //	SPH Particle relaxation section
    //----------------------------------------------------------------------
    /** check whether run particle relaxation for body fitted particle distribution. */
    InnerRelation shell_inner(shell_body);
    BodyAlignedBoxByCell inlet_detection_box(shell_body,
                                             makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(0.0, 0.0, 0.0)), buffer_half));
    BodyAlignedBoxByCell outlet_detection_box(shell_body,
                                              makeShared<AlignedBoxShape>(xAxis, Transform(Vec3d(12.0 * length_scale, 0.0, 0.0)), buffer_half));
    //----------------------------------------------------------------------
    //	Methods used for particle relaxation.
    //----------------------------------------------------------------------
    using namespace relax_dynamics;
    /** A  Physics relaxation step. */
    SurfaceRelaxationStep relaxation_step_inner(shell_inner);
    ShellNormalDirectionPrediction shell_normal_prediction(shell_inner, dp_0 * 1.0);

    // here, need a class to switch particles in aligned box to ghost particles (not real particles)
    SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> inlet_particles_detection(inlet_detection_box);
    SimpleDynamics<ParticlesInAlignedBoxDetectionByCell> outlet_particles_detection(outlet_detection_box);

    /** Write the body state to Vtp file. */
    BodyStatesRecordingToVtp write_shell_to_vtp({shell_body});
    write_shell_to_vtp.addToWrite<Vecd>(shell_body, "NormalDirection");
    BodyStatesRecordingToVtp write_all_bodies_to_vtp({sph_system});
    /** Write the particle reload files. */
    ReloadParticleIO write_particle_reload_files({&shell_body});
    ParticleSorting particle_sorting_shell(shell_body);
    //----------------------------------------------------------------------
    //	Particle relaxation starts here.
    //----------------------------------------------------------------------
    relaxation_step_inner.getOnSurfaceBounding().exec();
    write_shell_to_vtp.writeToFile(0.0);
    shell_body.updateCellLinkedList();
    //----------------------------------------------------------------------
    //	Particle relaxation time stepping start here.
    //----------------------------------------------------------------------
    int ite_p = 0;
    while (ite_p < 3000)
    {
        relaxation_step_inner.exec();
        ite_p += 1;
        if (ite_p % 500 == 0)
        {
            std::cout << std::fixed << std::setprecision(9) << "Relaxation steps for the imported model N = " << ite_p << "\n";
            write_shell_to_vtp.writeToFile(ite_p);
        }
    }
    std::cout << "The physics relaxation process of imported model finish !" << std::endl;

    shell_normal_prediction.smoothing_normal_exec();

    inlet_particles_detection.exec();
    particle_sorting_shell.exec();
    shell_body.updateCellLinkedList();
    outlet_particles_detection.exec();
    particle_sorting_shell.exec();
    shell_body.updateCellLinkedList();

    write_all_bodies_to_vtp.writeToFile(ite_p);
    write_particle_reload_files.writeToFile(0);

    return 0;
}
