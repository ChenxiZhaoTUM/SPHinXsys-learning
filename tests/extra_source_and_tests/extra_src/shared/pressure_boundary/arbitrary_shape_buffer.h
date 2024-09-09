/* @file 	  arbitrary_shape_buffer.h

 */

#ifndef ARBITRARY_SHAPE_BUFFER_H
#define ARBITRARY_SHAPE_BUFFER_H

#include "all_simbody.h"
#include "base_geometry.h"
#include "base_body_part.h"
#include "all_particle_dynamics.h"
#include "geometric_shape.h"
#include "level_set_shape.h"
#include "transform_shape.h"
#include "particle_reserve.h"

namespace SPH
{
class BaseAlignedShape
{
protected:
    const int alignment_axis_;

public:
    explicit BaseAlignedShape(int upper_bound_axis)
        : alignment_axis_(upper_bound_axis) {};
    virtual ~BaseAlignedShape(){};

    virtual bool checkInBounds(const Vecd &probe_point) = 0;
    //virtual bool checkUpperBound(const Vecd &probe_point) = 0;
    //virtual bool checkLowerBound(const Vecd &probe_point) = 0;
    //virtual bool checkNearUpperBound(const Vecd &probe_point, Real threshold) = 0;
    //virtual bool checkNearLowerBound(const Vecd &probe_point, Real threshold) = 0;
    //virtual Vecd getUpperPeriodic(const Vecd &probe_point) = 0;
    //virtual Vecd getLowerPeriodic(const Vecd &probe_point) = 0;
    int AlignmentAxis() { return alignment_axis_; };
};

class AlignedBoxShape02 : public TransformShape<GeometricShapeBox>, public BaseAlignedShape
{
  public:
    /** construct directly */
    template <typename... Args>
    explicit AlignedBoxShape02(int upper_bound_axis, const Transform &transform, Args &&... args)
        : TransformShape<GeometricShapeBox>(transform, std::forward<Args>(args)...),
          BaseAlignedShape(upper_bound_axis){};
    /** construct from a shape already has aligned boundaries */
    template <typename... Args>
    explicit AlignedBoxShape02(int upper_bound_axis, const Shape &shape, Args &&... args)
        : TransformShape<GeometricShapeBox>(
              Transform(Vecd(0.5 * (shape.bounding_box_.second_ + shape.bounding_box_.first_))),
              0.5 * (shape.bounding_box_.second_ - shape.bounding_box_.first_), std::forward<Args>(args)...),
          BaseAlignedShape(upper_bound_axis){};
    virtual ~AlignedBoxShape02(){};

    Vecd HalfSize() { return halfsize_; }
    bool checkInBounds(const Vecd &probe_point) override;
    bool checkUpperBound(const Vecd &probe_point);
    bool checkLowerBound(const Vecd &probe_point);
    bool checkNearUpperBound(const Vecd &probe_point, Real threshold);
    bool checkNearLowerBound(const Vecd &probe_point, Real threshold); 
    Vecd getUpperPeriodic(const Vecd &probe_point);
    Vecd getLowerPeriodic(const Vecd &probe_point);
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

using BodyAlignedBoxByCell02 = BaseAlignedRegion<BodyRegionByCell, AlignedBoxShape02>;


namespace relax_dynamics
{
template <typename AlignedShapeType>
class ParticlesInAlignedRegionDetectionByCell : public BaseLocalDynamics<BodyPartByCell>, public DataDelegateSimple
{
  public:
      ParticlesInAlignedRegionDetectionByCell(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part)
          : BaseLocalDynamics<BodyPartByCell>(aligned_region_part),
          DataDelegateSimple(aligned_region_part.getSPHBody()),
          pos_(*particles_->getVariableDataByName<Vecd>("Position")),
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
    StdLargeVec<Vecd> &pos_;
    AlignedShapeType &aligned_shape_;
};

using DeleteParticlesInBox = ParticlesInAlignedRegionDetectionByCell<AlignedBoxShape02>;
} // namespace relax_dynamics

namespace fluid_dynamics
{
template <typename AlignedShapeType>
class EmitterInflowInjectionArb : public BaseLocalDynamics<BodyPartByParticle>, public DataDelegateSimple
{
  public:
      EmitterInflowInjectionArb(BaseAlignedRegion<BodyRegionByParticle, AlignedShapeType>& aligned_region_part, ParticleBuffer<Base>& buffer);
    virtual ~EmitterInflowInjectionArb(){};

    void update(size_t original_index_i, Real dt = 0.0);

  protected:
    std::mutex mutex_switch_to_real_; /**< mutex exclusion for memory conflict */
    Fluid &fluid_;
    StdLargeVec<size_t> &original_id_;
    StdLargeVec<size_t> &sorted_id_;
    StdLargeVec<Vecd> &pos_;
    StdLargeVec<Real> &rho_, &p_;
    ParticleBuffer<Base> &buffer_;
    AlignedShapeType &aligned_shape_;
};

template <typename AlignedShapeType>
class DisposerOutflowDeletionArb : public BaseLocalDynamics<BodyPartByCell>, public DataDelegateSimple
{
  public:
      DisposerOutflowDeletionArb(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part);
    virtual ~DisposerOutflowDeletionArb(){};

    void update(size_t index_i, Real dt = 0.0);

  protected:
    std::mutex mutex_switch_to_buffer_; /**< mutex exclusion for memory conflict */
    StdLargeVec<Vecd> &pos_;
    AlignedShapeType &aligned_shape_;
};

template <typename AlignedShapeType>
class DisposerOutflowDeletionWithWindkesselArb: public fluid_dynamics::DisposerOutflowDeletionArb<AlignedShapeType>
{
  public:
      DisposerOutflowDeletionWithWindkesselArb(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part, const std::string& body_part_name);
    virtual ~DisposerOutflowDeletionWithWindkesselArb(){};

    void update(size_t index_i, Real dt = 0.0);

  protected:
    StdLargeVec<Real> &Vol_;
    Real &flow_rate_;
};
} // namespace fluid_dynamics
} // namespace SPH
#endif // ARBITRARY_SHAPE_BUFFER_H
