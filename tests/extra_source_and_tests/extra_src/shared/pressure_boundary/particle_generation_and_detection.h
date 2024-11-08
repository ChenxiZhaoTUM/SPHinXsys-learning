#ifndef PARTICLE_GENERATION_AND_DETECTION_H
#define PARTICLE_GENERATION_AND_DETECTION_H

#include "base_body.h"
#include "base_particle_generator.h"
#include "all_geometries.h"

#include "base_fluid_dynamics.h"
#include "particle_reserve.h"

#include "base_relax_dynamics.h"
#include "particle_smoothing.hpp"
#include "relax_stepping.hpp"

#include <mutex>

namespace SPH
{
class FromSTLFile;
template <>
class ParticleGenerator<SurfaceParticles, FromSTLFile> : public ParticleGenerator<SurfaceParticles>
{
    Real mesh_total_area_;
    Real particle_spacing_;
    const Real thickness_;
    Real avg_particle_volume_;
    size_t planned_number_of_particles_;

    TriangleMeshShapeSTL* mesh_shape_;
    Shape &initial_shape_;

public:
    explicit ParticleGenerator(SPHBody &sph_body, SurfaceParticles &surface_particles, TriangleMeshShapeSTL* mesh_shape) 
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
                const auto& pos = vertex_positions[faces[i][j]];
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
        Real interval = planned_number_of_particles_ / (num_faces + TinyReal);  // if planned_number_of_particles_ >= num_faces, every face will generate particles
        if (interval <= 0)
            interval = 1; // It has to be lager than 0.

        for (int i = 0; i < num_faces; ++i)
        {
            Vec3d vertices[3];
            for (int j = 0; j < 3; ++j)
            {
                const auto& pos = vertex_positions[faces[i][j]];
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

        addPositionAndVolumetricMeasure(face_center, avg_particle_volume_/thickness_);
        addSurfaceProperties(initial_shape_.findNormalDirection(face_center), thickness_);
    }

    void generateParticlesOnFace(const Vec3d vertices[3], int particles_per_face)
    {
        for (int k = 0; k < particles_per_face; ++k)
        {
            Real u = static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX);
            Real v = static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX);

            if (u + v > 1.0) {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            Vec3d particle_position = (1 - u - v) * vertices[0] + u * vertices[1] + v * vertices[2];

            addPositionAndVolumetricMeasure(particle_position, avg_particle_volume_/thickness_);
            addSurfaceProperties(initial_shape_.findNormalDirection(particle_position), thickness_);
        }
    }
};

template <class BodyRegionType, typename AlignedShapeType>
class BaseAlignedRegion : public BodyRegionType
{
public:
    BaseAlignedRegion(RealBody &real_body, AlignedShapeType &aligned_shape)
        : BodyRegionType(real_body, aligned_shape), aligned_shape_(aligned_shape){};
    BaseAlignedRegion(RealBody& real_body, SharedPtr<AlignedShapeType> aligned_shape_ptr)
        : BodyRegionType(real_body, aligned_shape_ptr), aligned_shape_(*aligned_shape_ptr.get()){};
    virtual ~BaseAlignedRegion(){};
    AlignedShapeType &getAlignedShape() { return aligned_shape_; };

protected:
    AlignedShapeType &aligned_shape_;
};

template <typename AlignedShapeType>
using BodyAlignedRegionByCell = BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>;

namespace relax_dynamics
{
/**
 * @class DisposerOutflowDeletion
 * @brief Delete particles who ruing out the computational domain.
 */
class ParticlesInAlignedBoxDetectionByCell : public BaseLocalDynamics<BodyPartByCell>
{
  public:
    ParticlesInAlignedBoxDetectionByCell(BodyAlignedBoxByCell &aligned_box_part);
    virtual ~ParticlesInAlignedBoxDetectionByCell(){};

    void update(size_t index_i, Real dt = 0.0);

  protected:
    std::mutex mutex_switch_to_buffer_; /**< mutex exclusion for memory conflict */
    Vecd *pos_;
    AlignedBoxShape &aligned_box_;
};

template <typename AlignedShapeType>
class ParticlesInAlignedRegionDetectionByCell : public BaseLocalDynamics<BodyPartByCell>
{
  public:
      ParticlesInAlignedRegionDetectionByCell(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part)
          : BaseLocalDynamics<BodyPartByCell>(aligned_region_part),
          pos_(particles_->getVariableDataByName<Vecd>("Position")),
          aligned_shape_(aligned_region_part.getAlignedShape()) {};
    virtual ~ParticlesInAlignedRegionDetectionByCell(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        mutex_switch_to_ghost_.lock();
        while (aligned_shape_.checkInBounds(pos_[index_i]) && index_i < particles_->TotalRealParticles())
        {
            particles_->switchToBufferParticle(index_i);
        }
        mutex_switch_to_ghost_.unlock();
    }

  protected:
    std::mutex mutex_switch_to_ghost_; /**< mutex exclusion for memory conflict */
    Vecd *pos_;
    AlignedShapeType &aligned_shape_;
};

using DeleteParticlesInBox = ParticlesInAlignedRegionDetectionByCell<AlignedBoxShape>;

class OnSurfaceBounding : public LocalDynamics
{
  public:
    OnSurfaceBounding(RealBody &real_body_);
    virtual ~OnSurfaceBounding(){};
    void update(size_t index_i, Real dt = 0.0);

  protected:
    Vecd *pos_;
    Shape *shape_;
};

class SurfaceRelaxationStep : public BaseDynamics<void>
{
  public:
    explicit SurfaceRelaxationStep(BaseInnerRelation &inner_relation);
    virtual ~SurfaceRelaxationStep(){};
    virtual void exec(Real dt = 0.0) override;
    SimpleDynamics<OnSurfaceBounding> &getOnSurfaceBounding() { return on_surface_bounding_; };

  protected:
    RealBody &real_body_;
    BaseInnerRelation &inner_relation_;
    InteractionDynamics<RelaxationResidue<Inner<>>> relaxation_residue_;
    ReduceDynamics<RelaxationScaling> relaxation_scaling_;
    SimpleDynamics<PositionRelaxation> position_relaxation_;
    SimpleDynamics<OnSurfaceBounding> on_surface_bounding_;
};
} // namespace relax_dynamics

} // namespace SPH
#endif PARTICLE_GENERATION_AND_DETECTION_H