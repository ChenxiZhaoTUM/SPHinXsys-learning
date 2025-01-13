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