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
#include "bidirectional_buffer.h"

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

using BodyAlignedBoxByCell02 = BaseAlignedRegion<BodyRegionByCell, AlignedBoxShape>;


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

using DeleteParticlesInBox = ParticlesInAlignedRegionDetectionByCell<AlignedBoxShape>;
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
      explicit DisposerOutflowDeletionArb(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part)
          : BaseLocalDynamics<BodyPartByCell>(aligned_region_part),
          DataDelegateSimple(aligned_region_part.getSPHBody()),
          pos_(*particles_->getVariableDataByName<Vecd>("Position")),
          aligned_shape_(aligned_region_part.getAlignedShape()) {};
    virtual ~DisposerOutflowDeletionArb(){};

    void update(size_t index_i, Real dt = 0.0)
    {
        mutex_switch_to_buffer_.lock();
        while (aligned_shape_.checkUpperBound(pos_[index_i]) && index_i < particles_->TotalRealParticles())
        {
            particles_->switchToBufferParticle(index_i);
        }
        mutex_switch_to_buffer_.unlock();
    }

  protected:
    std::mutex mutex_switch_to_buffer_; /**< mutex exclusion for memory conflict */
    StdLargeVec<Vecd> &pos_;
    AlignedShapeType &aligned_shape_;
};

template <typename AlignedShapeType, typename TargetPressure, class ExecutionPolicy = ParallelPolicy>
class BidirectionalBufferArb
{
  protected:
    TargetPressure target_pressure_;

    class TagBufferParticles : public BaseLocalDynamics<BodyPartByCell>, public DataDelegateSimple
    {
      public:
        TagBufferParticles(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part)
            : BaseLocalDynamics<BodyPartByCell>(aligned_region_part),
              DataDelegateSimple(aligned_region_part.getSPHBody()),
              pos_(*particles_->getVariableDataByName<Vecd>("Position")),
              aligned_shape_(aligned_region_part.getAlignedShape()),
              buffer_particle_indicator_(*particles_->registerSharedVariable<int>("BufferParticleIndicator"))
        {
            particles_->addVariableToSort<int>("BufferParticleIndicator");
        };
        virtual ~TagBufferParticles(){};

        virtual void update(size_t index_i, Real dt = 0.0)
        {
            buffer_particle_indicator_[index_i] = aligned_shape_.checkInBounds(pos_[index_i]) ? 1 : 0;
        };

      protected:
        StdLargeVec<Vecd> &pos_;
        AlignedShapeType &aligned_shape_;
        StdLargeVec<int> &buffer_particle_indicator_;
    };

    class Injection : public BaseLocalDynamics<BodyPartByCell>, public DataDelegateSimple
    {
      public:
        Injection(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part, ParticleBuffer<Base> &particle_buffer,
                  TargetPressure &target_pressure)
            : BaseLocalDynamics<BodyPartByCell>(aligned_region_part),
              DataDelegateSimple(aligned_region_part.getSPHBody()),
              particle_buffer_(particle_buffer),
              aligned_shape_(aligned_region_part.getAlignedShape()),
              fluid_(DynamicCast<Fluid>(this, particles_->getBaseMaterial())),
              original_id_(particles_->ParticleOriginalIds()),
              pos_n_(*particles_->getVariableDataByName<Vecd>("Position")),
              rho_n_(*particles_->getVariableDataByName<Real>("Density")),
              p_(*particles_->getVariableDataByName<Real>("Pressure")),
              previous_surface_indicator_(*particles_->getVariableDataByName<int>("PreviousSurfaceIndicator")),
              buffer_particle_indicator_(*particles_->getVariableDataByName<int>("BufferParticleIndicator")),
              target_pressure_(target_pressure)
        {
            particle_buffer_.checkParticlesReserved();
        };
        virtual ~Injection(){};

        void update(size_t index_i, Real dt = 0.0)
        {
            if (aligned_shape_.checkUpperBound(pos_n_[index_i]) && buffer_particle_indicator_[index_i] == 1)
            {
                mutex_switch_to_real_.lock();
                particle_buffer_.checkEnoughBuffer(*particles_);
                particles_->createRealParticleFrom(index_i);
                mutex_switch_to_real_.unlock();

                /** Periodic bounding. */
                pos_n_[index_i] = aligned_shape_.getUpperPeriodic(pos_n_[index_i]);
                Real sound_speed = fluid_.getSoundSpeed(rho_n_[index_i]);
                p_[index_i] = target_pressure_(p_[index_i]);
                rho_n_[index_i] = p_[index_i] / pow(sound_speed, 2.0) + fluid_.ReferenceDensity();
                previous_surface_indicator_[index_i] = 1;
            }
        }

      protected:
        std::mutex mutex_switch_to_real_;
        ParticleBuffer<Base> &particle_buffer_;
        AlignedShapeType &aligned_shape_;
        Fluid &fluid_;
        StdLargeVec<size_t> &original_id_;
        StdLargeVec<Vecd> &pos_n_;
        StdLargeVec<Real> &rho_n_, &p_;
        StdLargeVec<int> &previous_surface_indicator_, &buffer_particle_indicator_;

      private:
        TargetPressure &target_pressure_;
    };

  public:
    BidirectionalBufferArb(BaseAlignedRegion<BodyRegionByCell, AlignedShapeType>& aligned_region_part, ParticleBuffer<Base> &particle_buffer)
        : target_pressure_(*this), tag_buffer_particles(aligned_region_part),
          injection(aligned_region_part, particle_buffer, target_pressure_){};
    virtual ~BidirectionalBufferArb(){};

    SimpleDynamics<TagBufferParticles, ExecutionPolicy> tag_buffer_particles;
    SimpleDynamics<Injection, ExecutionPolicy> injection;
};

template <typename AlignedShapeType>
using NonPrescribedPressureBidirectionalBufferArb = BidirectionalBufferArb<AlignedShapeType, NonPrescribedPressure>;
} // namespace fluid_dynamics
} // namespace SPH
#endif // ARBITRARY_SHAPE_BUFFER_H
